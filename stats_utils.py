import numpy as np

def crossraPsd2d(img1,img2,show=False):
    s1 = len(img1)
    s2 = len(img2)

    img1 = np.array(img1) - np.mean(img1)
    img2 = np.array(img2) - np.mean(img2)
    #a = np.random.randint(2, size=(10,10))
    #k = [[1,1,1],[1,1,1],[1,1,1]]

    tensor_a = tf.constant(img1, tf.float32)
    tensor_k = tf.constant(img2, tf.float32)
    conv = tf.nn.convolution(
    tf.reshape(tensor_a, [1, s1, s1, 1]),
    tf.reshape(tensor_k, [s2, s2, 1, 1]),
    #use_cudnn_on_gpu=True,
    padding='SAME')
    conv = tf.Session().run(conv)
    conv = np.reshape(conv,(s1,s1))


    k = np.ones((s1,s1))
    #print(k)
    tensor_a = tf.constant(k, tf.float32)
    tensor_k = tf.constant(k, tf.float32)
    convc = tf.nn.convolution(
    tf.reshape(tensor_a, [1, s1, s1, 1]),
    tf.reshape(tensor_k, [s1, s1, 1, 1]),
    #use_cudnn_on_gpu=True,
    padding='SAME')
    convc = tf.Session().run(convc)
    convc = np.reshape(convc,(s1,s1))

    conv = conv / convc

    imgf = np.fft.fft2(conv)
    imgfs = np.fft.fftshift(imgf)
    S = np.zeros(128)
    Sconv = np.zeros(128)
    C = np.zeros(128)
    k_list = []
    for i in range(256):
        for j in range(256):

            i2 = i - (256.-1)/2.
            j2 = j - (256.-1)/2.

            r = int(np.round(np.sqrt(i2**2+j2**2)))


            if r <= 127:
                S[r] += imgfs[i][j]
                Sconv[r] += conv[i][j]
                C[r] += 1

    for i in range(128):
        k = i*(1.*2*np.pi/300)
        if C[i] == 0:
            S[i] = 0
            Sconv[i] = 0
        else:
            #print(k**2 * S[i] / C[i])
            S[i] = np.real(k**2 * S[i] / C[i])
            Sconv[i] = np.real(k**0 * Sconv[i] / C[i])

        k_list.append(k)

    if show == True:
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(conv)
        plt.show()
        plt.imshow(convc)
        plt.show()

    #S,k_list = raPsd2d(conv,s1,show=show)



    return S,Sconv,k_list


def xps(img1, img2, show=False):
    # get averaged power spectral density of image with resolution res
    img_width = img1.shape[0]
    #print('img width',int(img_width/2))

    #compute power spectrum
    imgf1 = np.fft.fft2(img1)
    imgfs1 = np.fft.fftshift(imgf1)
    imgf2 = np.fft.fft2(img2)
    imgfs2 = np.fft.fftshift(imgf2)
    imgfs2 = np.conjugate(imgfs2)
    imgfsp = imgfs1 * imgfs2
    #print(np.shape(imgfsp))

    S = np.zeros(int(img_width/2))
    C = np.zeros(int(img_width/2))
    k_list = []
    for i in range(int(img_width)):
        for j in range(int(img_width)):

            i2 = i - (float(img_width)-1)/2.
            j2 = j - (float(img_width)-1)/2.

            r = int(np.round(np.sqrt(i2**2+j2**2)))


            if r <= (int(img_width/2)-1):
                S[r] += imgfsp[i][j]
                C[r] += 1
    physical_width = (img_width/256)*300
    for i in range(int(img_width/2)):
        k = i*(1.*2*np.pi/physical_width)
        if C[i] == 0:
            S[i] = 0
        else:
            S[i] = k**2 * S[i] / C[i]

        k_list.append(k)

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

    return S,k_list


def raPsd2d(img, res, show=False):
    # get averaged power spectral density of image with resolution res
    img_width = img.shape[0]
    #print('img width',int(img_width/2))

    #compute power spectrum
    imgf = np.fft.fft2(img)
    imgfs = np.fft.fftshift(imgf)
    imgfsp = (np.abs(imgfs)) **2.
    #print(np.shape(imgfsp))

    S = np.zeros(int(img_width/2))
    C = np.zeros(int(img_width/2))
    k_list = []
    for i in range(int(img_width)):
        for j in range(int(img_width)):

            i2 = i - (float(img_width)-1)/2.
            j2 = j - (float(img_width)-1)/2.

            r = int(np.round(np.sqrt(i2**2+j2**2)))


            if r <= (int(img_width/2)-1):
                S[r] += imgfsp[i][j]
                C[r] += 1
    physical_width = (img_width/256)*300
    for i in range(int(img_width/2)):
        k = i*(1.*2*np.pi/physical_width)
        if C[i] == 0:
            S[i] = 0
        else:
            S[i] = k**2 * S[i] / C[i]

        k_list.append(k)

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

    return S,k_list



def produce_average_ps(slices):
    img_width = slices.shape[1]
    #print('width',img_width)
    PS = np.zeros(int(img_width/2)-1) #the first value of the PS is always zero so ignore that
    N = len(slices)
    #print('len',N)
    values_list = [ [] for i in range(int(img_width/2)-1) ] #list of 127 empty lists
    std_list = []
    S0 = []
    S1 = []

    for i in range(N):
        slc = slices[i]
        S,k_list = raPsd2d(slc,(int(img_width),int(img_width)))
        S = S[1:] #ignore the first element in PS becuase its always zero
        #k_list = k_list[1:]
        #S0.append(S[0])
        #S1.append(S[1])
        PS = np.add(PS,S)
        for j in range(len(S)):
            values_list[j].append(S[j])
    PS = PS / N
    #print(len(k_list))
    k_list = k_list[1:]
    #print(len(k_list))
    #print(S0)
    #print(S1)

    for k in range(len(values_list)):
        std = np.std(values_list[k])
        std_list.append(std)
    return PS,std_list,k_list



def compare_ps(real_PS,fake_PS,ps_std):
    diff = np.subtract(real_PS,fake_PS)
    for i in range(len(diff)):
        diff[i] = (1.*diff[i])/ps_std[i]
    mod_diff2 = diff**2
    tot_diff = np.sum(mod_diff2)
    return tot_diff



def get_pk_hist(slices):
    N = len(slices)
    count_list = []

    for i in range(N):
        counts = []
        slc = slices[i]
        for j in range(2,np.shape(slc)[0]-2): #dont consider edge rows
            for k in range(2,np.shape(slc)[1]-2): #dont consider edge columns
                middle = slc[j,k] #middle cell that we are considering
                largest = True #set middle cell to be the largest out of neighbours
                done = False
                for p in range(-2,3):
                    if done == True: break
                    for q in range(-2,3):
                        if done == True: break
                        if p != 0 or q != 0: #dont compare with middle cell
                            neighbour = slc[j+p,k+q]
                            if neighbour >= middle:##################what if the peak is more than a pixek big
                                largest = False
                                done = True
                if largest == True:
                    counts.append(middle)
        count_list.append(len(counts))
    return count_list


def get_peak_vs_brightness(slices):
    #print('slices shape',slices.shape)
    N = len(slices)
    #print('N',N)
    brightness_list = []

    for i in range(N):
        slc = slices[i]
        #print('middle',slc[1,1])
        #print('middle',slc[1,1][0])
        #print('slc shape',slc.shape)
        for j in range(2,np.shape(slc)[0]-2): #dont consider edge rows
            for k in range(2,np.shape(slc)[1]-2): #dont consider edge columns
                middle = slc[j,k] #middle cell that we are considering
                largest = True #set middle cell to be the largest out of neighbours
                done = False
                for p in range(-2,3):
                    if done == True: break
                    for q in range(-2,3):
                        if done == True: break
                        if p != 0 or q != 0: #dont compare with middle cell
                            neighbour = slc[j+p,k+q]
                            if neighbour >= middle:
                                largest = False
                                done = True
                if largest == True:
                    brightness_list.append(middle[0])
    return brightness_list


def get_pixel_val(slices):
    print(slices.shape)
    N = len(slices)
    elem = [] #list of elements to use for histogram

    for i in range(N):
        slc = slices[i]
        for j in range(np.shape(slc)[0]):
            for k in range(np.shape(slc)[1]):
                elem.append(slc[j,k][0])

    return elem
