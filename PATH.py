from sys import platform

def get_path():
    if platform == "darwin":
        return r'/Users/HirokazuKatori/Desktop/Msci Project'
    elif platform == "linux":
        return r'/home/jz8415/21cmFAST-master'
    else:
        return r'C:\21cmFAST\21cmFAST-master'
