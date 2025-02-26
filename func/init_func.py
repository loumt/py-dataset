import os
from utils.file_tool import mkdir,copy
from utils.log_tool import *


def init_folder():
    #初始化文件夹
    mkdir("data")
    mkdir("data/images")
    mkdir("data/images/train")
    mkdir("data/images/test")
    mkdir("data/images/val")
    mkdir("data/labels")
    mkdir("weights")

def init_dataset(opt):
    log(F"[INIT DATA] 总量:{opt.total} 比例:训练集({opt.percent_train}),验证集({opt.percent_val}),测试集({opt.percent_test})")

    num_all_img = os.listdir(F"{opt.dataset}")
    num_all_img_size = len(num_all_img)
    if num_all_img_size < opt.total:
        log(F"[ERROR]dateset len < total:{opt.total}")
        quit()


    ftrain = open('data/labels/train.txt', 'w+', encoding='utf-8')
    fval = open('data/labels/val.txt', 'w+', encoding='utf-8')
    ftest = open('data/labels/test.txt', 'w+', encoding='utf-8')
    flabels = open('data/labels.txt', 'w+', encoding='utf-8')

    index = 0
    with open(opt.label, encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            flabels.write(line)
            image_name = line.split(' ')[0]
            if index < num_all_img_size * opt.percent_train:
                ftrain.write(line)
                copy(F'{opt.dataset}/{image_name}', "data/images/train")
            elif index < num_all_img_size * (opt.percent_train + opt.percent_val):
                fval.write(line)
                copy(F'{opt.dataset}/{image_name}', "data/images/val")
            else:
                ftest.write(line)
                copy(F'{opt.dataset}/{image_name}', "data/images/test")
            index += 1
        f.close()

    log(f"[INFO] Complete!!")


def init_charset(opt):
    log(f'[INFO] init charset ......')
    with open(opt.label, encoding='utf-8-sig') as f:
        lines = f.readlines()
        log(f'\t[INFO]{opt.label} len: {len(lines)}')
        f.close()

    char = []
    for line in lines:
        char.append(line.strip().split(".jpg ")[1])

    with open('data/chinese.txt', 'w+', encoding='utf-8') as f:
        char = ''.join(char)
        char = set(char)
        char = list(char)
        char.sort()
        char = ''.join(char)
        log(f'\t[INFO] char len : {len(char)}')
        f.write(char)
