import os.path
from os import removedirs,remove
from shutil import rmtree

from utils.log_tool import *

def rm(p):
    if os.path.exists(p):
        log(f"[REMOVE]{p}")
        if os.path.isdir(p):
            rmtree(p)
        if os.path.isfile(p):
            remove(p)
    else:
        log(F"[NOT EXIST]{p}")

def clear():
    log("----------------clear------------------------")

    #清理label集合
    rm('data/labels.txt')
    rm('data/labels/train.txt')
    rm('data/labels/val.txt')
    rm('data/labels/test.txt')
    rm('data/images')
    rm('data')
    log("=>>> CLEAN COMPLETE!!!")


if __name__ == "__main__":
    clear()