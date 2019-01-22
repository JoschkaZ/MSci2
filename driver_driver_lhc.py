# IMPORTS
import importlib
import numpy as np
import utils
import datetime
importlib.reload(utils)
import time
import pyDOE

# SETTINGS
lhc_space = [
['ZETA_X', 2.0e56, 2.0e56, 'log'],
['test1', 0, 5, 'lin'],
['test2', 1, 5, 'log']
#can add more parameters here...
]
n_samples = 3
z_range = [11, 7, -1]


# get lh coordinates
lhc_dim = len(lhc_space)
lh = pyDOE.lhs(lhc_dim, samples=n_samples, criterion='centermaximin')
print('original lh coordinates:')
print(lh)

# scale the lh coordinates appropriately
lh_scaled = []
for sample in lh:
    new_sample = []
    for i_param in range(len(lhc_space)):
        if lhc_space[i_param][3] == 'log':
            new = (lhc_space[i_param][2]/lhc_space[i_param][1])**(sample[i_param])*lhc_space[i_param][1]
        elif lhc_space[i_param][3] == 'lin':
            new = (lhc_space[i_param][2]-lhc_space[i_param][1])*(sample[i_param])+lhc_space[i_param][1]
        new_sample.append(new)
    lh_scaled.append(new_sample)
lh_scaled = np.array(lh_scaled)
print('scaled lh coordinates')
print(lh_scaled)


#%%
# loop over the lh samples and run the driver for all z
SEED = -1
for sample in lh_scaled:
    SEED += 1
    print('###################################################################')
    print('###################################################################')
    print('###################################################################')
    print('using seed ', SEED)
    print('using settings ' sample)
    start_time = time.time()

    #clean box folder
    utils.clear_box_directory()

    #remove compiled files
    utils.cd_to_programs()
    commands = ['make clean']
    utils.run_commands(commands)

    #change some parameters
    utils.change_parameter('ZETA_X', sample[0]) #NOTE make sure the indices are right
    utils.change_parameter('RANDOM_SEED', str(SEED))
    utils.change_parameter('drive_zscroll_noTs ZSTART', z_range[0]) # 8
    utils.change_parameter('drive_zscroll_noTs ZEND', z_range[1])
    utils.change_parameter('drive_zscroll_noTs ZSTEP', z_range[2])

    #run driver
    commands = ['make', './drive_zscroll_noTs']
    utils.run_commands(commands)

    #rename and zip all delta_T_boxes #NOTE all parameters must be included in param_string
    utils.cd_to_boxes()
    box_names = utils.get_delta_T_boxes()
    param_string = 'SEED'+str(SEED)+'_ZSTART'+str(ZSTART)+'_ZEND'+str(ZEND)+'_ZSTEP'+str(ZSTEP)+'_ZETA_X'+str(ZETA_X)
    box_names = utils.rename_boxes(box_names, param_string) # box name is: parameter_string + '_' + old_boxname

    #name archive:  archive name is: data + '_' + parameter_string
    archive_name = str(datetime.datetime.now()).replace(' ','_').replace(':','#').replace('.','#')+'_'+param_string
    utils.zip_boxes(box_names, archive_name)

    print('ITERATION FINISHED')
    print('time taken: ', time.time() - start_time, 's')
