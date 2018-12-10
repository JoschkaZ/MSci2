# IMPORTS
import importlib
import numpy as np
import utils
import datetime
importlib.reload(utils)


#zip_name = 'single_redshift'

seeds = range(1)
ZSTART = 12
ZEND = 11
ZSTEP = -1
ZETA_X = 'default'

for SEED in seeds:
    print('###################################################################')
    print('###################################################################')
    print('###################################################################')

    print('using seed ', SEED)

    #clean box folder
    utils.clear_box_directory()

    #remove compiled files
    utils.cd_to_programs()
    commands = ['make clean']
    utils.run_commands(commands)

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
    utils.cd_to_boxes()
    box_names = utils.get_delta_T_boxes()
    param_string = 'SEED'+str(SEED)+'_ZSTART'+str(ZSTART)+'_ZEND'+str(ZEND)+'_ZSTEP'+str(ZSTEP)+'_ZETA_X'+str(ZETA_X)
    box_names = utils.rename_boxes(box_names, param_string)

    #archive_name = zip_name + '_' + str(SEED) #DEFINE ARCHIVE NAME
    archive_name = str(datetime.datetime.now())+'_'+param_string
    #print('ARCHIVE_NAME', archive_name)
    utils.zip_boxes(box_names, archive_name)


    '''
    next steps after driver has finished:
    1. process boxes
    2. downlaod boxes
    3. repeat for different settings
    '''
