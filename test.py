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
