import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

y = np.load("EM_data.npy")
N = y.size

K = [4, 16, 64, 256, 1024]

Px = np.zeros((len(K), max(K)))
Params = {}
MinD = None
for (l, k) in enumerate(K):
    plt.figure(l)
    plt.scatter(y.real , y.imag, color='blue', label='Received data')

    C = np.random.choice(y.flatten(), size=k)
    # plt.scatter(C.real, C.imag, color='red', label='Initial cluster centres')

    # K-means algorithm
    j = np.zeros(N)
    for it in range(100):
        j = np.argmin(np.abs(y-C), axis=1)
        d = np.arange(k).reshape(-1,1)==j
        C = (d@y/np.maximum(np.sum(d, axis=1), 1).reshape(-1,1)).flatten()
    plt.scatter(C.real, C.imag, color='red', label='K-mean centres')

    Px = np.sum(d, axis=1)/np.sum(d)
    mean = Px@C
    sigma2 = Px@((C.real-mean.real)**2 + (C.imag-mean.imag)**2)
    delta_r = 1
    delta_i = 0
    qam = np.array([i+q for i in np.arange(-np.sqrt(k)+1, np.sqrt(k), 2) for q in 1j*np.arange(-np.sqrt(k)+1, np.sqrt(k), 2)])
    for it in range(30):
        ## E-Step: Calculate auxiliary distribution Q_{X|Y}
        Pxy = Px*ss.norm.pdf(y.real, C.real, np.sqrt(sigma2/2))*ss.norm.pdf(y.imag, C.imag, np.sqrt(sigma2/2))
        Qx_y = Pxy/np.sum(Pxy, axis=1).reshape(-1,1)
        
        # M-Step: Optimize parameters C, Px, sigma2, delta
        Px = np.sum(Qx_y, axis=0)/N
        temp = delta_r
        C = C/(delta_r+1j*delta_i)
        delta_r = np.sum(np.sum(Qx_y*(C.real*(y.real+delta_i*C.imag)+C.imag*(y.imag+delta_i*C.real))))/np.sum(np.sum(Qx_y*np.abs(C)**2))
        delta_i = np.sum(np.sum(Qx_y*(C.imag*(-y.real+temp*C.real)+C.real*(y.imag-temp*C.imag))))/np.sum(np.sum(Qx_y*np.abs(C)**2))
        sigma2 = np.sum(np.sum(Qx_y*((y.real-delta_r*C.real+delta_i*C.imag)**2+(y.imag-delta_r*C.imag-delta_i*C.real)**2)))/N
        C = (delta_r+1j*delta_i)*qam

    plt.scatter(C.real, C.imag, color='green', label='EM-QAM')
    plt.title(f"{k}-QAM")
    Params[k]=(Px, delta_r+1j*delta_i, sigma2)

    j = np.argmin(np.abs(y-C), axis=1)
    temp = np.sum(np.abs(y.flatten()-C[j]))
    if (MinD is None or temp<MinD) and np.sum(Qx_y==0)==0:
        MinD = temp
        MinK = k

print("Px =", Params[MinK][0])
print("delta =", Params[MinK][1])
print("sigma2 =", Params[MinK][2])
print(f"{MinK}-QAM was selected as the correct constellation")
plt.legend()
plt.show()

# def awgn_pY(y, X, pX, sigma2):
#     # reshape X as row-vector and y as column-vector for broadcasting
#     X = X.reshape((1, X.size))
#     pX = pX.reshape((1, pX.size))
#     y = y.reshape((y.size, 1))

#     # calculate pY
#     pY = np.sum( pX * ss.norm.pdf(y, X, np.sqrt(sigma2)), axis=1 )
#     return pY 

# plt.figure(l+1)
# yrange = np.linspace(np.min(y.real), np.max(y.real), 1000)
# py = awgn_pY(yrange, MinC.real, Px, sigma2/2)
# plt.plot(yrange, py)
# plt.show()