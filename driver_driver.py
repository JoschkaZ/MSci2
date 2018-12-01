# IMPORTS
import importlib
import numpy as np
import utils
importlib.reload(utils)


zip_name = 'single_redshift'

seeds = range(100)
for seed in seeds:
    print('using seed ', seed)

    #clean box folder
    utils.clear_box_directory()

    #remove compiled files
    utils.cd_to_programs()
    commands = ['make clean']
    utils.run_commands(commands)

    #change some parameters
    utils.change_parameter('ZETA_X', 'default')
    utils.change_parameter('RANDOM_SEED', str(seed))
    utils.change_parameter('drive_zscroll_noTs ZSTART', 9)
    utils.change_parameter('drive_zscroll_noTs ZEND', 9)

    #run driver
    #commands = ['make', './drive_logZscroll_Ts']
    commands = ['make', './drive_zscroll_noTs']
    utils.run_commands(commands)

    #zip all delta_T_boxes
    utils.cd_to_boxes()
    box_names = utils.get_delta_T_boxes()

    archive_name = zip_name + '_' + str(seed) #DEFINE ARCHIVE NAME
    print('ARCHIVE_NAME', archive_name)
    utils.zip_boxes(box_names, archive_name)


    '''
    next steps after driver has finished:
    1. process boxes
    2. downlaod boxes
    3. repeat for different settings
    '''
