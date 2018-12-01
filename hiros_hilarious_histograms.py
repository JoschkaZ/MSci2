import pickle
import numpy as np
import matplotlib.pyplot as plt
import pyfits
import pylab as py
from scipy import fftpack, ndimage

slices = pickle.load(open("slices.pkl","rb"))

#produce histograms of pixel values
N = 10 #produce histograms from N real images
incr = int((1. * np.shape(slices)[0]) / N)
elem = [] #list of elements to use for histogram

for i in range(1,N+1):
    slc = slices[i*incr -1]
    for j in range(np.shape(slc)[0]):
        for k in range(np.shape(slc)[1]):
            elem.append(slc[j,k])

plt.hist(elem, bins=100)
plt.show()


#produce peak count
N = 1000 #find peak counts for N real images
incr = int((1. * np.shape(slices)[0]) / N)
count_list = []

for i in range(1,N+1):
    if i%50==0: print(i)
    counts = []
    slc = slices[i*incr -1]
    for j in range(2,np.shape(slc)[0]-2): #dont consider edge rows
        for k in range(2,np.shape(slc)[1]-2): #dont consider edge columns
            middle = slc[j,k] #middle cell that we are considering
            largest = True #set middle cell to be the largest out of neighbours
            done = False
            for p in range(-2,3):
                if done == True: break
                for q in range(-2,3):
                    if done == True: break
                    if p != 0 and q != 0: #dont compare with middle cell
                        neighbour = slc[j+p,k+q]
                        if neighbour > middle:
                            largest = False
                            done = True

            if largest == True:
                counts.append(middle)

    count_list.append(len(counts))

plt.hist(count_list, bins=100)
plt.show()
