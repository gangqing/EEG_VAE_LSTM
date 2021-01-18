import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from Config import Config

"""
数据加载和处理，生成训练样本：
1、99标签之前的数据(全0标签数据)作为训练数据；
2、99标签之后的数据(包括99标签数据)作为测试数据；
3、按维度进行Z标准化；
"""


def show(data, marker):
    plt.subplot(2, 1, 1)
    plt.plot([i for i in range(len(data))], data)

    plt.subplot(2, 1, 2)
    plt.plot([i for i in range(len(marker))], marker)

    plt.show()


class Data:
    def __init__(self, config: Config):
        self.config = config
        data, self.source_marker = self.load_data()
        # 归一化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.source_data = (data - self.mean) / self.std
        # 划分VAE数据集
        self.vae_train_data, self.vae_test_data = self.vae_data()
        # 划分LSTM数据集
        self.lstm_train_data, self.lstm_train_marker, self.lstm_test_data, self.lstm_test_marker = self.lstm_data()

    def load_data(self):
        """
        加载数据
        :return:
        data: [-1, 22]
        marker:
        0 : 无操作;
        1 - 5 ：对应手指操作;
        91 ：between;
        92 ：end;
        99 ：start
        """
        file = sio.loadmat(self.config.simple_path)
        data = file["o"]["data"][0][0]  # [-1, 22]
        data = np.array(data).astype(np.float)

        marker = file["o"]["marker"][0][0]  # [-1, 1]
        marker = np.array(marker).astype(np.int)
        return data, marker

    def vae_data(self):
        """
        vae数据集
        :return: [-1, 128, 22]
        """
        dim = self.config.simple_dim
        ds = self.target_ds([0])
        num_example = len(ds) // dim
        data = []
        for i in range(num_example):
            data.append(ds[i * dim: i * dim + dim])
        index = int(len(data) * 0.8)
        return data[: index], data[index:]

    def target_ds(self, target=None):
        # 返回正常数据
        if target is None:
            target = [0, 1, 2, 3, 4, 5]
        simple = []
        for i, marker in enumerate(self.source_marker):
            if marker in target:
                simple.append(self.source_data[i])
        return simple

    def lstm_data(self):
        """
        :return: data : [-1, 12, 128, 22]; marker : [-1, 12]
        """
        dim = self.config.simple_dim  # 128
        frame = self.config.frame  # 12
        data = []
        marker = []
        for i in range(dim):  # 数据开始位置
            sub_num_example = (len(self.source_data) - i) // dim  # 每128个数据组成一个子样本
            num_example = sub_num_example // frame  # 每12个子样本组成一个样本
            for j in range(num_example):  # 遍历样本数
                index_start = frame * dim * j + i
                index_end = frame * dim * (j + 1) + i
                ds = self.source_data[index_start: index_end]
                ds = np.reshape(ds, [frame, dim, self.config.num_dim])
                data.append(ds)  # [-1, 12, 128, 22]

                mk = self.source_marker[index_start: index_end]
                mk = np.max(np.reshape(mk, [frame, dim]), axis=1)
                marker.append(mk)  # [-1, 12]
        # 划分训练集和数据集， 只包含0标签的数据作为训练集，包含了非0标签的数据作为训练集
        train_data = []
        train_marker = None
        test_data = None
        test_marker = None
        for index, value_list in enumerate(marker):
            if np.max(value_list) == 0:
                train_data.append(data[index])
            else:
                test_data = data[index + 1:]
                test_marker = marker[index + 1:]
                train_marker = marker[:index]
                break
        return train_data, train_marker, test_data, test_marker

    def to_value(self, data):
        return data * self.std + self.mean


if __name__ == '__main__':
    config = Config()
    ds = Data(config)
