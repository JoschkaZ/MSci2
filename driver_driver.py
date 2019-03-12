# IMPORTS
import importlib
import numpy as np
import utils
import datetime
importlib.reload(utils)
import time


#zip_name = 'single_redshift'


seeds = range(100)
ZSTART = 7
ZEND = 7
ZSTEP = -1
ZETA_X = 'default'

ZETA_Xs = np.logspace(2.0e56/6., 2.0e56*3.3333, 5)
print('ZETA_X: ', ZETA_Xs)

for SEED in seeds:
    print('###################################################################')
    print('###################################################################')
    print('###################################################################')
    start_time = time.time()

    print('using seed ', SEED)

    #clean box folder
    utils.clear_box_directory()

    #remove compiled files
    utils.cd_to_programs()
    commands = ['make clean']
    utils.run_commands(commands)

    for ZETA_X in ZETA_Xs:
        print('ZETA_X: ', ZETA_X)
        #change some parameters
        utils.change_parameter('ZETA_X', ZETA_X)
        utils.change_parameter('RANDOM_SEED', str(SEED))
        utils.change_parameter('drive_zscroll_noTs ZSTART', ZSTART) # 8
        utils.change_parameter('drive_zscroll_noTs ZEND', ZEND)
        utils.change_parameter('drive_zscroll_noTs ZSTEP', ZSTEP)

        #run driver
        #commands = ['make', './drive_logZscroll_Ts']
        commands = ['make', './drive_zscroll_noTs']
        utils.run_commands(commands)

    #rename and zip all delta_T_boxes
    # box name is: parameter_string + '_' + old_boxname
    utils.cd_to_boxes()
    box_names = utils.get_delta_T_boxes()
    box_names_density = utils.get_deltax_boxes()

    param_string = 'SEED'+str(SEED)+'_ZSTART'+str(ZSTART)+'_ZEND'+str(ZEND)+'_ZSTEP'+str(ZSTEP)+'_ZETA_X'+str(ZETA_X)
    box_names = utils.rename_boxes(box_names, param_string)
    box_names_density = utils.rename_boxes(box_names_density, param_string)

    #name archive
    # archive name is: data + '_' + parameter_string
    archive_name = str(datetime.datetime.now()).replace(' ','_').replace(':','#').replace('.','#')+'_'+param_string
    utils.zip_boxes(box_names, archive_name)

    archive_name_density = 'density_'+str(datetime.datetime.now()).replace(' ','_').replace(':','#').replace('.','#')+'_'+param_string
    utils.zip_boxes(box_names_density, archive_name_density)

    print('ITERATION FINISHED')
    print('T = ', time.time() - start_time, 's')
