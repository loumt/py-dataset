
import os
import shutil

def mkdir(file):
    if not os.path.exists(file):
        os.mkdir(file)


def copy(source, target):
    shutil.copy(source, target)