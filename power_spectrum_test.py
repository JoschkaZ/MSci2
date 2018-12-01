# IMPORTS
import importlib
import numpy as np
import utils
from os import listdir
from os.path import isfile, join
importlib.reload(utils)
import matplotlib.pyplot as plt


box = utils.read_box(r'delta_T_v3_no_halos_z007.20_nf0.715123_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb014.04_Pop-1_256_300Mpc')


# %% VISUALIZE A BOX
utils.show_box(box)

#%% POWER SPECTRUM
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def raPsd2d(img, res, show=False):
    # get averaged power spectral density of image with resolution res


    #compute power spectrum
    imgf = np.fft.fft2(img)
    imgfs = np.fft.fftshift(imgf)
    imgfsp = (np.abs(imgfs) / (1.*256*256)) **2.
    print(np.shape(imgfsp))

    S = np.zeros(128)
    C = np.zeros(128)
    for i in range(256):
        for j in range(256):

            i2 = i - (256.-1)/2.
            j2 = j - (256.-1)/2.

            r = int(np.round(np.sqrt(i2**2+j2**2)))


            if r <= 127:
                S[r] += imgfsp[i][j]
                C[r] += 1

    for i in range(128):
        S[i] = S[i] / C[i]



    '''
    # compute radially averaged power spectrum
    #X,Y = mesgrid() # meshgrid(-dimMax/2:dimMax/2-1, -dimMax/2:dimMax/2-1)
    [X,Y] = np.meshgrid(range(-128,128), range(-128,128))
    print(X,Y)
    theta = np.zeros((256,256))
    rho = np.zeros((256, 256))

    for i in range(256):
        for j in range(256):
            r,t = cart2pol(X[i][j],Y[i][j])
            rho[i][j] = np.round(r)
            theta[i][j] = t

    i = np.zeros(np.floor(256/2)+1)
    for r in range(0, np.floor(256/2+1)):
        i[r+1] =


    Hi, I found this to be faster, hope it helps, S.
    %% code for fast radial averaging:
    [X,Y] = np.meshgrid(range(-128,128), range(-128,128))
    rr=np.sqrt(X**2+Y**2);
    ri=np.floor(rr)+1;
    N=numel(ri);
    [~,~,iir]=unique(ri); %find unique values
    Qt =sparse(1:N,iir,ones(N,1));

    tic;Pf=(rr(:).'*Qt).';toc %radial average
    '''



    if show == True:
        print('Original')
        plt.imshow(np.log(np.abs(img)), cmap='hot', interpolation='nearest')
        plt.show()
        print('Fourier')
        plt.imshow(np.log(np.abs(imgf)), cmap='hot', interpolation='nearest')
        plt.show()
        print('Fourier + Shift')
        plt.imshow(np.log(np.abs(imgfs)), cmap='hot', interpolation='nearest')
        plt.show()
        print('Fourier + Shift + Squared')
        plt.imshow(np.log(np.abs(imgfsp)), cmap='hot', interpolation='nearest')
        plt.show()
        #print('Theta')
        #plt.imshow(theta, cmap='hot', interpolation='nearest')
        #plt.show()
        #print('Rho')
        #plt.imshow(rho, cmap='hot', interpolation='nearest')
        #plt.show()

    return S


S = raPsd2d(box[0], (256,256), show=True)
#S = raPsd2d(np.random.random((256,256)), (256,256))
#S = raPsd2d(np.random.normal(1,0.1,(256,256)), (256,256))

plt.plot(S[3::])
plt.yscale('log')
plt.show()
