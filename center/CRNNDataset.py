from torch.utils.data import Dataset, DataLoader, DataChunk
from PIL import Image
import os
from utils.img_tool import ResizeAndNormalize
from torchvision import transforms

class CRNNDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.image_dict = self.read_image()
        self.image_name = [filename for filename, _ in self.image_dict.items()]


    def __getitem__(self, index):
        # 图片路径
        image_path = os.path.join(self.image_path, self.image_name[index])

        # 图片标签
        label = self.image_dict.get(self.image_name[index])

        # 图片处理-1.转换为灰度图片
        image = Image.open(image_path).convert("L")

        # 图片处理-2.对图片进行resize和归一化操作
        transform = ResizeAndNormalize((1024, 32))

        return transform(image), label

    def __len__(self):
        return len(self.image_dict)

    def read_image(self):
        dir = {}
        with open(self.label_path, encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                k = line.strip().replace('\n', '').split(r' ')[0]
                v = line.strip().replace('\n', '').split(r' ')[1]
                dir[k] = v
            f.close()
        return dir

def get_dataloader(image_path, label_path, batch_size, shuffle):
    # 图片转换形式
    traindata_transfomer = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor格式
        transforms.Resize(60),  # 调整图像大小，调整为高度或宽度为60像素，另一边按比例调整
        transforms.RandomCrop(48),  # 裁剪图片，随机裁剪成高度和宽度均为48像素的部分
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行归一化处理。对每个通道执行了均值为0.5、标准差为0.5的归一化操作
    ])
    valdata_transfomer = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor格
        transforms.Resize(48),  # 调整图像大小，调整为高度或宽度为48像素，另一边按比例调整
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    dataset = CRNNDataset(image_path, label_path)
    # dataset 数据集对象，必须实现__len__和__getitem__方法
    # batch_size 每个批次中的样本数。如果设置为 None，则必须指定 batch_sampler
    # shuffle 是否在每个 epoch 开始时打乱数据集。通常在训练时设置为 True，但在验证或测试时设置为 False
    # sampler 定义从数据集中采样的策略。如果指定了 sampler，则 shuffle 参数会被忽略
    # batch_sampler 类似于 sampler，但返回的是一个批次的索引列表，而不是单个样本的索引。如果指定了 batch_sampler，则 batch_size, shuffle, sampler 和 drop_last 参数将被忽略
    # num_workers使用多少个子进程来加载数据。设置为 0 表示在主进程中加载数据（单线程）。增加此值可以加速数据加载，特别是在数据集较大或读取速度较慢的情况下
    # collate_fn 自定义函数，用于合并样本列表以形成一个批次。默认情况下，PyTorch 提供了一个简单的 default_collate 函数，它可以处理大多数情况。如果需要自定义批次合并逻辑，可以提供自己的 collate_fn
    # pin_memory如果设置为 True，则 DataLoader 将把返回的张量复制到 CUDA 固定内存中，这可以加快数据传输到 GPU 的速度。
    # drop_last  如果数据集大小不能被 batch_size 整除，则最后一个批次可能会小于 batch_size。如果设置为 True，则会丢弃这个不完整的批次
    # timeout设置等待工作进程生成下一个批次的时间（秒）。如果超时，则会抛出 TimeoutError 异常。
    # worker_init_fn在每个工作进程启动时调用的初始化函数。这对于设置随机种子或其他初始化任务非常有用。
    # multiprocessing_context控制多进程上下文的行为。可以是 'spawn', 'fork', 'forkserver' 等。
    # generator用于生成随机数的生成器。可以用于控制数据集打乱顺序时使用的随机种子。
    # prefetch_factor每个工作进程在队列中预取的批次数量。仅在 num_workers > 0 时有效。
    # persistent_workers如果设置为 True，则在 DataLoader 被销毁之前，工作进程不会关闭。这可以提高性能，尤其是在多个 epoch 中使用相同的 DataLoader 时。
    loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle)
    return loader
