import os
os.getcwd()
os.chdir(r"C:\Users\Joschka\github\MSci2")
#%%
# IMPORTS
import importlib
import numpy as np
import matplotlib.pyplot as plt
import utils
from os import listdir
from os.path import isfile, join
importlib.reload(utils)


# %% READ A BOX
box = utils.read_box(r'delta_T_v3_no_halos_z007.20_nf0.715123_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb014.04_Pop-1_256_300Mpc')
# %% VISUALIZE A BOX
utils.show_box(box)


# %% MAKE A MOVIE
#utils.box_to_movie(box)


# %% CHANGE A PARAMETER
utils.change_parameter('ZETA_X', '4.0e56')
#utils.change_parameter('ZETA_X', 'default')

# %% RUN COMMANDS IN SHELL
commands = [
'make',
'./drive_logZscroll_Ts'
]

utils.change_parameter('ZETA_X', 'default')
utils.run_commands(commands)
utils.change_parameter('ZETA_X', '3.0e56')
utils.run_commands(commands)


#%%

mypath = 'C:\\'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)

#%% try chaning parameter

utils.change_parameter('drive_zscroll_noTs ZSTART', '9')
utils.change_parameter('drive_zscroll_noTs ZEND', '9')
utils.change_parameter('RANDOM_SEED', '123')




#%% get boxes

boxnames = utils.get_delta_T_boxes(mypath='C:\Outputs\Outputs')
print(boxnames)
print(len(boxnames))
#%%
utils.boxes_to_list_of_slices(boxnames, mypath='C:\Outputs\Outputs')


#%%
import dcgan_21cm
import random
importlib.reload(utils)
importlib.reload(dcgan_21cm)
img1 = [[1,1,1], [1,1,1], [1,1,1]]
img2 = [[1,1,1], [1,1,1], [1,1,1]]

img1 = box[0]
img2 = box[128]

#img1 = np.array(img1) - np.mean(img1)
#img2 = np.array(img2) - np.mean(img2)

'''
img1 = np.zeros((256,256))
for x in range(256):
    for y in range(256):
        r = np.sqrt(1.*(x-128)**2+(y-128)**2)
        r = x
        #img1[x][y] = np.sin(50*r/150. + np.random.normal(0,0)) + np.sin(300*r/150. + np.random.normal(0,0))
        img1[x][y] = np.random.normal(0,1)


img2 = np.zeros((256,256))

for x in range(256):
    for y in range(256):
        r = np.sqrt(1.*(x-128)**2+(y-128)**2)
        r = x
        img2[x][y] = np.sin(50*r/150. + np.random.normal(0,0)) + np.sin(300*r/150. + np.random.normal(0,0))
        img2[x][y] = np.random.normal(0,1)
'''

#img1 = img2
print(img1)
S = dcgan_21cm.crossraPsd2d(img1,img2,show=True)
#test = dcgan_21cm.raPsd2d(img1, 256,show=True)
#%%
print(S)
plt.plot(S[0][0:])

plt.show()
plt.plot(S[1][0:])
plt.show()



#%%
labels = [1, 1, 2, 1, 3, 2]
labels.index(2)
[x for x in labels==1]

#%%
[[5]]*10


#%
for i in range(5,10):
    print(i)



#%%


a = np.array([[1,2],[3,4]])
print((a-5)/2)
b = {5: 'a', 8: 'b'}
print(max(b.keys()))

#%%

fig, axs = plt.subplots(9, 2,figsize=(4,18), dpi=250)
#fig = plt.figure(figsize=(20, 6))

y = [[0,1],[2,3]]
x = [[0,10],[30,30]]
for i in range(9):
    for j in range(2):
        if j==0 and i==0:
            axs[i,j].imshow(x)
        else:
            axs[i,j].imshow(y)
        axs[i,j].set_title('Label')
        axs[i,j].axis('off')

plt.show()


#%%
import numpy as np
x = np.random.normal(0, 0.05, size=(4,4,4))
y = np.random.normal(0, 0.05, size=(4,4,4))
print(x)
print(y)
print(x+y)



#%%
import matplotlib.pyplot as plt


f = plt.figure(1)
f.plot([1,2,3])

g = plt.figure(2)
g.plot([3,2,1])
g.show()


f.plot([2,3,4])
f.show()
