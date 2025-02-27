import argparse
from func.init_func import init_folder, init_dataset,init_charset


def start(opt):
    # 初始化文件夹
    init_folder()

    #初始化数据集
    init_dataset(opt)

    #初始化字符集
    init_charset(opt)

def parse_args():
    parser = argparse.ArgumentParser(description="Ready.")
    parser.add_argument('--dataset', type=str, default='D:/DataSet/Train/images', help='数据集文件夹')
    parser.add_argument('--label', type=str, default='D:/DataSet/Train/labels.txt', help='数据标签集合')
    parser.add_argument('--percent-train', type=float, default=0.85, help='训练文件占比')
    parser.add_argument('--percent-val', type=float, default=0.10, help='验证文件占比')
    parser.add_argument('--percent-test', type=float, default=0.05, help='测试文件占比')
    parser.add_argument('--total', type=int, default=500, help='准备数据集大小')
    return parser.parse_args();


if __name__ == "__main__":
    opt = parse_args();
    start(opt)