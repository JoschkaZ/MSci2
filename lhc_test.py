import pyDOE
import matplotlib.pyplot as plt


lh = pyDOE.lhs(2, samples=20, criterion='centermaximin')
#“center” or “c”: center the points within the sampling intervals
#“maximin” or “m”: maximize the minimum distance between points, but place the point in a randomized location within its interval
#“centermaximin” or “cm”: same as “maximin”, but centered within the intervals
#“correlation” or “corr”: minimize the maximum correlation coefficient



#%
x = []
y = []
for entry in lh:
    x.append(entry[0])
    y.append(entry[1])


plt.scatter(x,y)
plt.show()

#%%
import numpy as np
import pickle as pkl

data = []

img1 = np.array([[0,1],[2,3]])
img2 = np.array([[0,0],[0,0]])

for fake in range(100):

    for z in range(7,12):
        if z == 7:
            img = np.repeat(np.repeat(img1,128, axis=0), 128, axis=1)
        else:
            img = np.repeat(np.repeat(img2,128, axis=0), 128, axis=1)
        #print(img)

        data.append([img, 'something_z'+str(z) + '_something'])


pkl.dump(data, open("faketest_images.pkl", "wb"))
