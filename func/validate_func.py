import os
from utils.log_tool import *

# 目录
parentImageDir = 'data/images'
parentLabelDir = 'data/labels'

trainName = 'train'
valName = 'val'
testName = 'test'

def valid(name):
    if not os.path.exists(f'{parentImageDir}/{name}'):
        log(f'\t[ERROR] Not Exist => {parentImageDir}/{name}')
        quit()

    _set = set(os.listdir(f'{parentImageDir}/{name}'))
    log(f'\t[INFO] {name}: {len(_set)}个')

    with open(f'{parentLabelDir}/{name}.txt', encoding='utf-8-sig') as f:
        labels = f.readlines()
        labels = {label.replace('\n', '').split(' ')[0] for label in labels}

        if labels.issubset(_set):
            log(f'\t[INFO] 所有标签对应的图片都存在于目录({name})中。')
            return True
        else:
            # 找出不在_set中的labels
            missing_files = labels - _set
            log(f'\t[ERROR] 以下标签对应的图片不存在于目录({name})中：')
            for missing_file in missing_files:
                log(missing_file)
            return False

def valid_data():
    return [valid(trainName), valid(valName), valid(testName)]
