

import argparse
from func.validate_func import valid_data
from center.CRNNDataset import get_dataloader
from utils.log_tool import *


def start(opt):
    log(r'[INFO] 验证 ......')
    train_ok, val_ok, test_ok = valid_data()
    if not train_ok & val_ok & test_ok:
        log(f'[ERROR] get error from ready operation , please check it !')
        quit()

    log(r'[INFO] 获取数据集加载器 ......')
    loader = get_dataloader(r'data/images/train', r'data/labels/train.txt', opt.batch_size, True)

    log(r'[INFO] 创建神经网络 ......')
    for epoch in range(1, opt.epochs + 1):
        print(epoch)

def parse_args():
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument('--epochs', type=int, default=1, help='训练批次')
    parser.add_argument('--batch-size', type=int, default=20, help='批次大小')
    return parser.parse_args();

if __name__ == "__main__":
    opt = parse_args();
    start(opt)