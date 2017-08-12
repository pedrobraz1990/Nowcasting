import datetime as dt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
from pykalman import KalmanFilter
import numpy.ma as ma
import scipy.linalg as linalg
import scipy as sp
import statsmodels.api as sm
import statsmodels.tsa as tsa
import os.path
import pickle
import scipy as sp
from scipy.optimize import minimize
import time

import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy.linalg import det

from scipy.stats import norm
from scipy.stats import chi2

from pykalman import KalmanFilter as pykalman_KF
from numpy import ma

from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Semaphore

# pd.set_option('max_rows', 20)
pd.set_option('max_rows', 100)

dataDetailsSheet = "Plan2"

picklesDir = "./Pickles/"


# priorsTable = pd.read_pickle('./PerformanceTestPickles/priorsTable')
Z = pd.read_pickle('./PerformanceTestPickles/Z')
T = pd.read_pickle('./PerformanceTestPickles/T')
Q = pd.read_pickle('./PerformanceTestPickles/Q')
H = pd.read_pickle('./PerformanceTestPickles/H')
R = pd.read_pickle('./PerformanceTestPickles/R')
y = pd.read_pickle('./PerformanceTestPickles/y')
coefsIndex = pd.read_pickle('./PerformanceTestPickles/coefsIndex').values


def MH(priorsTable, Z, T, Q, H, R, y, coefsIndex, n, name):

    print("Started {name}".format(name=name))

    y = np.array(y)
    Z = np.array(Z)
    H = np.array(H)
    T = np.array(T)
    Q = np.array(Q)
    R = np.array(R)


    coefs = []

    alphaPostValues = np.zeros(n)

    accept = np.zeros(n)

    initialCoefs = pd.Series(0.1, index=coefsIndex)
    initialCoefs = pd.Series(priorsTable['initialValues'], index=coefsIndex)


    # Tentativa de fazer com transformacao exponencial
    initialCoefs[priorsTable["distribution"] == 'Chi'] = np.log(initialCoefs[priorsTable["distribution"] == 'Chi'])

    temp = initialCoefs.copy()

    temp[priorsTable["distribution"] == 'Chi'] = np.exp(temp)[priorsTable["distribution"] == 'Chi']

    lastLogPost = posteriori(priorsTable, temp, Z, T, Q, H, R, y, coefsIndex)

    #     lastLogPost = posteriori(priorsTable,initialCoefs,Z, T, Q, H, R,y, coefsIndex)

    #     print(lastLogPost)

    alphaPostValues[0] = lastLogPost

    coefs.append(initialCoefs)

    disturbances = []

    for i in range(1, n):

        disturbance = np.multiply(np.random.randn(coefsIndex.shape[0]),
                                  priorsTable["randomWalkVariance"])

        disturbances.append(disturbance)

        temp = coefs[i - 1] + disturbance

        temp2 = temp.copy()

        temp2[priorsTable["distribution"] == 'Chi'] = np.exp(temp2)[priorsTable["distribution"] == 'Chi']

        newLogPost = posteriori(priorsTable, temp2, Z, T, Q, H, R, y, coefsIndex)

        #         disturbance[priorsTable["distribution"]=='Chi'] = np.exp(disturbance)[priorsTable["distribution"]=='Chi']

        #         temp = coefs[i-1] + disturbance

        #         temp2 = temp

        #         temp2[priorsTable["distribution"]=='Chi'] = np.exp(temp)[priorsTable["distribution"]=='Chi']

        #         newLogPost = posteriori(priorsTable,temp2,Z, T, Q, H, R,y, coefsIndex)

        #         print(newLogPost)

        alphaPostValues[i] = newLogPost

        logAlpha = newLogPost - lastLogPost

        #         print(logAlpha)

        alpha = np.exp(logAlpha)

        #         print(alpha)

        r = np.min([1, alpha])

        u = np.random.uniform()

        if u < r:
            #             print('pos')
            accept[i] = 1
            lastLogPost = newLogPost
            coefs.append(temp)
        else:
            #             print('no')
            accept[i] = 0
            coefs.append(coefs[i - 1])
            #             coefs.append(temp)

    coefs = pd.DataFrame(coefs)

    coefs = coefs.T
    coefs[priorsTable["distribution"] == 'Chi'] = np.exp(coefs)[priorsTable["distribution"] == 'Chi']
    coefs = coefs.T

    # name = './BayesianResults/posterior_n{n!s}_{name}'.format(n=n,name=name)
    coefs.reset_index(drop=True,inplace=True)
    coefs.to_pickle(name)
    return {
        'posterior': coefs,
        'posteriorValues': pd.DataFrame(alphaPostValues),
        #         'posterior' : post[burn:],
        #             'post'
        'accept': accept,
        'disturbances': pd.DataFrame(disturbances),
        #            'rs' : pd.DataFrame(rs),
        #            'priors': pd.DataFrame(priors)}
    }

def getPdfValue(value, dist, mean, variance):
    if dist == "Normal":
        return norm.pdf(value, mean, variance)
    if dist == "Chi":
        return chi2.pdf(value, mean)

def posteriori(priorsTable, theta, Z, T, Q, H, R, y, coefsIndex):
    # OBS: RETURNS THE LOG VALUE

    post = 0
    for i in range(0, priorsTable.shape[0]):
        #         print(theta[i])
        #         print(priorsTable.iloc[i])
        #         print(getPdfValue(value = theta[i],
        #                             dist = priorsTable.iloc[i]["distribution"],
        #                             mean = priorsTable.iloc[i]["mean"],
        #                             variance = priorsTable.iloc[i]["variance"],))

        post += np.log(getPdfValue(value=theta[i],
                                   dist=priorsTable.iloc[i]["distribution"],
                                   mean=priorsTable.iloc[i]["mean"],
                                   variance=priorsTable.iloc[i]["variance"], ))

    # ll = wrapper(coefs,Z.copy(), T.copy(), Q.copy(), H.copy(), R.copy(), y.copy(),coefsIndex)['ll']
    # ll = wrapper2(theta, Z.copy(), T.copy(), Q.copy(), H.copy(), R.copy(), y.copy(), coefsIndex)['ll']
    ll = hardWrapper(theta, Z.copy(), T.copy(), Q.copy(), H.copy(), R.copy(), y.copy(), coefsIndex)['ll']
    #     ll = wrapper3(coefs,Z.copy(), T.copy(), Q.copy(), H.copy(), R.copy(), y.copy(),coefsIndex)
    #     print(ll)
    #     ll = ll['ll']
    #     return ll

    #     ll = np.exp(ll/1000)
    #     ll = np.exp(ll)
    #     print(post)
    return post + ll

def hardWrapper(coefs, Z, T, Q, H, R, y, coefsIndex):
    # coefs = pd.Series(coefs, index=coefsIndex)

    # Z first column
    for i in range(0,20+1):
        Z[i,0] = coefs[i]

    # Now the GDP Multiplications
    Z[20, 1] = 2 * coefs[20]
    Z[20, 2] = 3 * coefs[20]
    Z[20, 3] = 2 * coefs[20]
    Z[20, 4] = 1 * coefs[20]


    # A
    T[0,0] = coefs[21]

    # The quarterly alpha
    T[25,25] = coefs[22]

    # Monthly alphas
    coefi = 23
    for i in range(5,24+1):
        T[i,i] = coefs[coefi]
        coefi += 1

    # Q's
    Q[0, 0] = coefs[43]
    Q[5, 5] = coefs[44]
    Q[6, 6] = coefs[45]
    Q[7, 7] = coefs[46]
    Q[8, 8] = coefs[47]
    Q[9, 9] = coefs[48]

    coefi = 49
    for i in range(10, 25 + 1):
        Q[i, i] = coefs[coefi]
        coefi += 1

    # H Diag
    temp = np.ones(H.shape[0]) * coefs[65]
    H = np.diag(temp)

    m = Z.shape[1]
    a1 = np.zeros((m))
    P1 = np.ones((m, m)) * 0.5
    nStates = m

    # They already should be numpy arrays at this point
    # y = np.array(y)
    # Z = np.array(Z)
    # H = np.array(H)
    # T = np.array(T)
    # Q = np.array(Q)
    # R = np.array(R)

    return KalmanFilter2(
        y=y.T,
        nStates=nStates,
        Z=Z,
        H=H,
        T=T,
        Q=Q,
        a1=a1,
        P1=P1,
        R=R,
        export=True)

def wrapper2(coefs, Z, T, Q, H, R, y, coefsIndex):
    coefs = pd.Series(coefs, index=coefsIndex)

    for i in range(0, coefs.size):
        Z.replace(coefs.index[i], coefs.iloc[i], inplace=True)
        T.replace(coefs.index[i], coefs.iloc[i], inplace=True)
        Q.replace(coefs.index[i], coefs.iloc[i], inplace=True)
        H.replace(coefs.index[i], coefs.iloc[i], inplace=True)

    # Z.replace(coefsIndex[i],coefs[i],inplace=True)
    #         T.replace(coefsIndex[i],coefs[i],inplace=True)
    #         Q.replace(coefsIndex[i],coefs[i],inplace=True)
    #         H.replace(coefsIndex[i],coefs[i],inplace=True)

    Z.replace("2*BZGDYOY% Index_loading", 2 * coefs.loc["BZGDYOY% Index_loading"], inplace=True)
    Z.replace("3*BZGDYOY% Index_loading", 3 * coefs.loc["BZGDYOY% Index_loading"], inplace=True)

    m = Z.shape[1]
    a1 = np.zeros((m))
    P1 = np.ones((m, m)) * 0.5
    nStates = m

    y = np.array(y)
    Z = np.array(Z)
    H = np.array(H)
    T = np.array(T)
    Q = np.array(Q)
    R = np.array(R)

    return KalmanFilter2(
        y=y.T,
        nStates=nStates,
        Z=Z,
        H=H,
        T=T,
        Q=Q,
        a1=a1,
        P1=P1,
        R=R,
        export=True)

def KalmanFilter2(y, nStates, Z, H, T, Q, a1, P1, R, export=False):
# Only receives np arrays

    p = y.shape[1]
    n = y.shape[0]
    m = nStates


    yhat = np.empty((n, p))
    Z = np.array(Z.astype(float))  # (PxM) we'll drop t
    H = np.array(H.astype(float))
    T = np.array(T.astype(float))  # Should be M x M
    Q = np.array(Q.astype(float))  # (RxR)
    a = np.empty((n + 1, p + 1, m))  # each alpha t,i is mx1
    a[0, 0, :] = np.array(a1.astype(float)).ravel()  # TODO Check a1 dimension
    P = np.empty((n + 1, p + 1, m, m))
    P[0, 0, :, :] = np.array(P1.astype(float))
    v = np.empty((n, p))
    F = np.empty((n, p))
    K = np.empty((n, p, m))
    ZT = Z.T  # To avoid transposing it several times
    TT = T.T  # To avoid transposing it several times
    R = np.array(R)  # (MxR)
    RT = R.T
    ll = 0


    for t in range(0, n):
        ind = ~np.isnan(y[t, :])
        templl = 0
        pst = 0
        for i in range(0, p):  # later on change to Pt
            if ind[i]:
                v[t, i] = y[t, i] - Z[i, :].reshape((1, m)).dot(a[t, i, :].T)  # a should be mx1
                F[t, i] = Z[i, :].reshape((1, m)).dot(P[t, i, :, :]).dot(Z[i, :]) + H[i, i]
                K[t, i, :] = P[t, i, :, :].dot(Z[i, :]) * F[t, i] ** (-1)
                a[t, i + 1, :] = a[t, i, :] + K[t, i, :] * v[t, i]
                P[t, i + 1, :, :] = P[t, i, :, :] - (K[t, i, :] * F[t, i]).reshape((m, 1)).dot(
                    K[t, i].reshape((1, m)))
            else:
                # Setting all Z's to zeros
                v[t, i] = np.zeros(v[t, i].shape)
                F[t, i] = H[
                    i, i]
                K[t, i, :] = np.zeros(K[t, i, :].shape)
                a[t, i + 1, :] = a[t, i, :] + K[t, i, :] * v[t, i]
                P[t, i + 1, :, :] = P[t, i, :, :] - (K[t, i, :] * F[t, i]).reshape(
                    (m, 1)).dot(K[t, i].reshape((1, m)))
            if F[t,i] != 0:
                templl += np.log(F[t,i]) + (v[t,i] ** 2) / F[t,i]
                pst += 1

        ll+= pst * np.log(2*np.pi) + templl


        a[t + 1, 0, :] = T.dot(a[t, i + 1, :])
        P[t + 1, 0, :, :] = T.dot(P[t, i + 1]).dot(TT) + R.dot(Q).dot(RT)
        # yhat[t,:] = Z.dot(a[t,1,:]) # ERRADO

        if export:
            yhat[t, :] = Z.dot(a[t, 0, :])

    ll *= -0.5
#     ll = np.exp(ll)

    if export:
        states = pd.DataFrame(a[:, 0, :])
        yhat = pd.DataFrame(yhat)
        y = pd.DataFrame(y)
        return {'states' : states,
                'yhat' : yhat,
                'prediction': yhat,
                'y' : y,
                'll' : ll}
    else:
        return ll

def loadDf(pubDate):
    filename = picklesDir + pubDate.strftime('%Y-%m-%d') + ".pickle"
    return pickle.load(open(filename, "rb"))

def dumpResults(priorsTable, Z, T, Q, H, R, cutY, coefsIndex, n, filename1, filename2):
    start_time = time.time()
    posterior = MH(priorsTable, Z, T, Q, H, R, cutY.T, coefsIndex, n, filename1)['posterior']

    cutY = np.array(cutY)

    temp = hardWrapper(posterior.median(), Z.copy(), T.copy(), Q.copy(), H.copy(), R.copy(), cutY.T.copy(), coefsIndex)
    temp['prediction'].columns = y.T.columns
    temp['prediction'].index = y.T.index
    pickle.dump(temp, open(filename2, "wb"))
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

##### GENERATE THE STYLISED CALENDAR ####
    yn = y.T
    calendar = pd.read_excel("Nowcasting.xlsx", sheetname=dataDetailsSheet)
    calendar = calendar[["Code", "PubLag"]]
    calendar.set_index("Code", inplace=True)

    newDfs = []

    # For each Series
    for serie in yn.columns:
        #     Create new df
        newDf = pd.DataFrame(yn[serie])
        newDf.reset_index(inplace=True)
        cols = list(newDf.columns)
        cols[0] = "refDate"
        newDf.columns = cols
        newDf["pubDate"] = newDf["refDate"] + dt.timedelta(int(calendar.loc[serie]))
        newDf.set_index(["refDate", "pubDate"], inplace=True)
        newDfs.append(newDf)

    yn = pd.concat(newDfs, axis=1)
    yn = yn.groupby(level="refDate").fillna(method='ffill')

    pubDates = yn.index.get_level_values("pubDate").unique().sort_values()

##### END - GENERATE THE STYLISED CALENDAR ####

    start_time = time.time()

    picklesDir = "./BayesianResults/Pickles/"
    priorsTable = pd.read_excel("./BayesianResults/PriorsTable3.xlsx")
    Z = np.array(Z)
    H = np.array(H)
    T = np.array(T)
    Q = np.array(Q)
    R = np.array(R)

    n = 50


    pool = Pool(processes=10)

    listOfResults = {}
    for pubDate in pubDates[13:1400]:
        filename1 = picklesDir + pubDate.strftime('%Y-%m-%d') + "_posteriori.pickle"
        filename2 = picklesDir + pubDate.strftime('%Y-%m-%d') + "_results.pickle"
        if not os.path.exists(filename2):
            print(pubDate)
            cutY = yn[yn.index.get_level_values("pubDate") <= pubDate].groupby(level="refDate").last()
            cutY = cutY.reindex(y.T.index)
            # pool.apply_async(func=dumpResults,
            #                  args=(priorsTable, Z, T, Q, H, R, cutY, coefsIndex, n, filename1, filename2))
            dumpResults(priorsTable, Z, T, Q, H, R, cutY, coefsIndex, n, filename1, filename2)


    pool.close()
    pool.join()


    print("--- FINAL: %s seconds ---" % (time.time() - start_time))
    print("Done !")
