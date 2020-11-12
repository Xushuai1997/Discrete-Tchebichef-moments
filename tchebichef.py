import numpy as np


def tch_ploy(N):  # tchebichef polynomials
    T = np.zeros((N, N))
    for x in range(0, N):
        T[0, x] = 1 / np.sqrt(N)
    for t in range(1, N-1):
        T[t, 0] = -np.sqrt((N - t) / (N + t)) * np.sqrt((2 * t + 1) / (2 * t - 1)) * T[t - 1, 0]
        T[t, 1] = (1 + t * (1 + t) / (1 - N)) * T[t, 0]
        T[t, N - 2] = (-1)**t * T[t, 1]
        T[t, N - 1] = (-1)**t * T[t, 0]
        for x in range(2, int(N / 2)):
            g1 = (-t * (t + 1) - (2 * x - 1) * (x - N - 1) - x) / (x * (N - x))
            g2 = ((x - 1) * (x - N - 1)) / (x * (N - x))
            T[t, x] = g1 * T[t, x-1] + g2 * T[t, x-2]
            T[t, N-1-x] = (-1)**t * T[t, x]

    return T


def tch_moment(img, tn, tm):
    tmoment = np.dot(np.dot(tn, img), tm.T)
    return tmoment


def inv_tch_moment(tmoment, Tn, Tm, order):
    M_ = tmoment[0:order, 0:order]
    H1, W1 = M_.shape
    W, _ = Tn.shape
    H, _ = Tm.shape

    M1 = np.pad(M_, ((0, H - H1), (0, W - W1)), 'constant', constant_values=(0, 0))
    X = Tn.T.dot(M1.dot(Tm))

    return X


def quat_tch_moment(img):   # quaternion tchebichef moments
    H, W, _ = img.shape
    tn = tch_ploy(H)
    tm = tch_ploy(W)
    r_tmoment = tch_moment(img[:,:,0], tn, tm)
    g_tmoment = tch_moment(img[:,:,1], tn, tm)
    b_tmoment = tch_moment(img[:,:,2], tn, tm)
    qua_moment = np.ones((H, W, 4))
    A_0 = (r_tmoment + g_tmoment + b_tmoment) / np.sqrt(3)
    A_1 = (g_tmoment - b_tmoment) / (-np.sqrt(3))
    A_2 = (b_tmoment - r_tmoment) / (-np.sqrt(3))
    A_3 = (r_tmoment - g_tmoment) / (-np.sqrt(3))
    for p in range(H):
        for q in range(W):
             qua_moment[p, q] = [A_0[p,q], A_1[p,q], A_2[p,q], A_3[p,q]]
    return qua_moment


def inv_qua_tch_moment(QTm, order):  # inverse quaternion tchebichef moments
    H, W, _= QTm.shape
    tn = tch_ploy(H)
    tm = tch_ploy(W)
    A0 = QTm[:, :, 0]
    A1 = QTm[:, :, 1]
    A2 = QTm[:, :, 2]
    A3 = QTm[:, :, 3]
    a1 = (-1/np.sqrt(3))*(inv_tch_moment(A0, tn, tm, order) + inv_tch_moment(A2, tn, tm, order) - inv_tch_moment(A3, tn, tm, order))
    a2 = (-1 / np.sqrt(3)) * (inv_tch_moment(A0, tn, tm, order) - inv_tch_moment(A1, tn, tm, order) + inv_tch_moment(A3, tn, tm, order))
    a3 = (-1 / np.sqrt(3)) * (inv_tch_moment(A0, tn, tm, order) + inv_tch_moment(A1, tn, tm, order) - inv_tch_moment(A2, tn, tm, order))
    reimg = np.zeros((H, W, 3), np.int)
    for x in range(H):
        for y in range(W):
            t = [a1[y, x], a2[y, x], a3[y, x]]
            reimg[y, x] = np.around(np.asarray(t))
    return reimg


if __name__=='__main__':
    tn = tch_ploy(8)
    tm = tch_ploy(8)
    print(np.dot(tn, tm.T).astype(int))