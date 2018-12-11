from sys import platform
import os
import numpy as np
import matplotlib.pyplot as plt
from sys import platform
from os import listdir
from os.path import isfile, join
import pickle as pkl
#from power_spectrum_test import raPsd2d
print(5)



#%%




def get_path():
    if platform == "darwin":
        return r'/Users/HirokazuKatori/Desktop/Msci Project'
    elif platform == "linux":
        user = get_user()
        return r'/home/' + user + r'/21cmFAST-master'
    else:
        return r'C:\21cmFAST\21cmFAST-master'




def get_user():
    linpath = os.getcwd()
    user = linpath.split('/')[2]
    return user

def what_platform():
    print(platform)
    return platform

def read_box(filename, verbose=1, mypath = ''):

    PATH = get_path()

    if mypath == '':
        if platform == "darwin": # its mac
            boxpath = PATH + '/Boxes/' +  filename
        elif platform == "linux":
            boxpath = PATH + '/Boxes/' + filename
        else: # its windows
                boxpath = PATH + '\\Boxes\\' +  filename
    else:

        if platform == "darwin": # its mac
            boxpath = mypath + '/' + filename
        elif platform == "linux":
            boxpath = mypath + '/' + filename
        else: # its windows
            boxpath = mypath + '\\' + filename


    dtype='f'
    fd=open(boxpath,'rb')
    read_data=np.fromfile(fd,dtype)
    fd.close()

    dim = int(np.round(len(read_data)**(1./3)))
    row = []
    rc = 0
    layer = []
    lc = 0
    box = []
    bc = 0

    for i,n in enumerate(read_data):

        row.append(n)
        rc += 1

        if rc == dim:
            layer.append(row)
            lc += 1
            row = []
            rc = 0

            if lc == dim:
                box.append(layer)
                bc += 1
                layer = []
                lc = 0

    box = np.array(box)

    if verbose==1:
        print('read box @ ', boxpath)
        print('with dimensions ', box.shape)
        print('..........')
    return box

def show_box(box):
    plt.imshow(box[0], cmap='hot', interpolation='nearest')
    plt.show()
    return 1

def box_to_movie(box, verbose=1):

    PATH = get_path()
    savedirectory = PATH + '\\output_movie\\'
    if verbose==1: 'making movie'
    for layer in range(len(box)):
        print('saving frame #', layer)
        plt.imshow(box[layer], cmap='hot', interpolation='nearest')
        plt.savefig(savedirectory + str(layer) + '.png', format='png')

    print('..........')
    return 1

def get_parameter_paths(parameter_name, new_value):

    if parameter_name == 'ZETA_X':

        f1 = 'Parameter_files'
        f2 = 'HEAT_PARAMS.H'
        if new_value == 'default': new_value = '2.0e56'

    elif parameter_name == 'RANDOM_SEED':

        f1 = 'Parameter_files'
        f2 = 'INIT_PARAMS.H'
        if new_value == 'default': new_value = '1'

    elif parameter_name == 'drive_zscroll_noTs ZSTART':

        f1 = 'Programs'
        f2 = 'drive_zscroll_noTs.c'
        parameter_name = 'ZSTART'
        if new_value == 'default': new_value = '12'

    elif parameter_name == 'drive_zscroll_noTs ZEND':

        f1 = 'Programs'
        f2 = 'drive_zscroll_noTs.c'
        parameter_name = 'ZEND'
        if new_value == 'default': new_value = '6'

    elif parameter_name == 'drive_zscroll_noTs ZSTEP':

        f1 = 'Programs'
        f2 = 'drive_zscroll_noTs.c'
        parameter_name = 'ZSTEP'
        if new_value == 'default': new_value = '-0.2'

    else:
        print('WARNING - PARAMETER_NAME NOT RECOGNIZED!')
        return 0,0,0
    return f1,f2,new_value,parameter_name

def change_parameter(parameter_name, new_value, verbose=1):

    #ZETA_X (double) (2.0e56) // 2e56 ~ 0.3 X-ray photons per stellar baryon

    # get path to file
    PATH = get_path()

    f1,f2,new_value,parameter_name = get_parameter_paths(parameter_name, new_value)
    if f1 == 0: return 0

    # get path to parameter file
    if platform == "darwin": # its a mac !
        filepath = PATH + '/' + f1 + '/' + f2
    elif platform == "linux": # its a linux !!
        filepath = PATH + '/' + f1 + '/' + f2
    else: # its a windows !!!!
        filepath = PATH + '\\' + f1 + '\\' + f2


    if verbose == 1: print('Changing paramter ', parameter_name, ' to ', new_value)

    # read old text
    with open(filepath, "r") as f:
         old = f.readlines() # read everything in the file

    # create new text
    new = []
    found_it = False
    for line in old:
        #print(line)
        if '#define ' + parameter_name + ' (' in line:
            found_it = True
            #print('CHANGING LINE')
            #print(line)
            if len(line.split('(')) == 3: #its #define name (type) (value)
                newline = '('.join(line.split('(')[0:2]) + '(' + str(new_value) + ')' + ')'.join(line.split(')')[2::])
            else: #its #define name (value)
                newline = line.split('(')[0] + '(' + str(new_value) + ')' + ')'.join(line.split(')')[1::])
            #print(newline)
        else:
            newline=line
        new.append(newline)
    if found_it == False: print('WARNING - COULD NOT FIND ROW WITH PARAMETER TO BE CHANGED!')

    #print('#####PRINTING NEW FILE#####')
    #for line in new:
    #    print(line)
    #    1

    # save new text
    '''
    print(filepath[0:-3]+'_mod.H')
    with open(filepath[-3:-1]+'_mod.H', 'w') as f:
        for line in new:
            print(line)
            f.write("%s\n" % line)
    '''

    print('writing to filepath ', filepath)
    new = ''.join(new)
    text_file = open(filepath, "w")
    text_file.write(new)
    text_file.close()




    if verbose == 1: print('Parameter file has been modified')

    # r'C:\21cmFAST\21cmFAST-master'
    #:\21cmFAST\21cmFAST-master\Parameter_files

    return 1

def run_commands(commands, verbose=1):
    for command in commands:
        print('Running command: ' + command)
        os.system(command)
    return 1

def cd_to_boxes():
    user = get_user()
    os.chdir(r'/home/' + user + r'/21cmFAST-master/Boxes')
def cd_to_programs():
    user = get_user()
    os.chdir(r'/home/' + user + r'/21cmFAST-master/Programs')
def cd_to_python():
    user = get_user()
    os.chdir(r'/home/' + user + r'/MSci-Project-21cm')

def clear_box_directory(verbose=1):
    if verbose == 1: print('clearing box directory')
    user = get_user()
    commands = ['rm /home/'+user+r'/21cmFAST-master/Boxes/*']
    run_commands(commands, verbose=verbose)
    return 1

def get_delta_T_boxes(verbose=1, mypath=''):
    if verbose == 1: print('getting delta_t box names')

    if mypath == '':
        user = get_user()
        mypath = r'/home/' + user + r'/21cmFAST-master/Boxes'
    boxfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    temp = []
    for boxfile in boxfiles:
        if boxfile[0:8] == 'delta_T_':
            temp.append(boxfile)

    return temp

def rename_boxes(box_names, param_string, verbose=1):
    if verbose==1: print('renaming boxes')
    user = get_user()

    commands = [] #mv /home/user/Files/filename1.ext /home/user/Files/filename2.ext
    new_box_names = []
    for box_name in box_names:
        new_box_name = param_string + '_' + box_name
        new_box_names.append(new_box_name)
        commands.append(
        'mv '+r'/home/'+user+r'/21cmFAST-master/Boxes/'+box_name+ r' /home/'+user+r'/21cmFAST-master/Boxes/'+new_box_name)
    run_commands(commands)

    return new_box_names


def zip_boxes(box_names, archive_name, verbose=1):
    user = get_user()

    #get box names
    box_names = '\n'.join(box_names)

    #write box names to out.txt
    text_file = open(r'/home/' + user + r'/21cmFAST-master/Boxes/out.txt', "w")
    text_file.write(box_names)
    text_file.close()

    if verbose==1: print('zipping boxes')


    print('archive name ', archive_name)
    # zip boxes
    commands = [
    'zip '+ archive_name +' -@ < out.txt', #zip all files listed in out.txt
    'mv '+archive_name+'.zip ' + '/home/'+user+r'/Outputs/'] # move archive.zip
    run_commands(commands)

    return 1

def boxes_to_list_of_slices(box_names, limit=None, mypath=''):


    if mypath == '':
        user = get_user()
        mypath =  r'/home/' + user + r'/Outputs'
    slices = []

    if limit== None: limit = len(box_names)
    for boxname in box_names[0:limit]:

        box = read_box(boxname, mypath=mypath)

        for i in range(0,255,5):
            slice = box[i,:,:]
            slices.append(slice)
        for i in range(0,255,5):
            slice = box[:,i,:]
            slices.append(slice)
        for i in range(0,255,5):
            slice = box[:,:,i]
            slices.append(slice)

    pkl.dump(slices, open("slices.pkl", "wb"))



if __name__ == '__main__':
    print(1)
