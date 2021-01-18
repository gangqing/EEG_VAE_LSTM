import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from Config import Config

"""
数据加载和处理，生成训练样本：
1、按照标签进行数据分割；
2、指定大小(128)截取数据样本；
3、按维度进行Z标准化；
"""


def show(data, marker):
    plt.subplot(2, 1, 1)
    plt.plot([i for i in range(len(data))], data)

    plt.subplot(2, 1, 2)
    plt.plot([i for i in range(len(marker))], marker)

    plt.savefig("test1.png")
    # plt.show()


class Data:
    def __init__(self, cfg: Config):
        self.config = cfg
        file_path = "data/5F-SubjectB-160311-5St-SGLHand-HFREQ.mat"
        file = sio.loadmat(file_path)
        dataes = file["o"]["data"][0][0]  # [-1, 22]
        marker = file["o"]["marker"][0][0]  # [-1, 1]
        dataes, marker = self.deal_data(dataes, marker)
        dataes = np.array(dataes).astype(np.float)
        self.mean = np.mean(dataes, axis=None)
        self.std = np.std(dataes, axis=None)
        dataes = (dataes - self.mean) / self.std
        # marker : uint8
        # 0 : 无操作
        # 1 - 5 ：对应手指操作
        # 91 ：between
        # 92 ：end
        # 99 ：start
        marker = np.array(marker).astype(np.int)
        # show(dataes, marker)
        # 按照marker划分数据
        labels = []
        data = []
        current = None
        start_index = 0
        targets = [0, 1, 2, 3, 4, 5]
        for index, m in enumerate(marker):
            if m not in targets:
                continue
            if m != current:
                if current is not None:
                    data.append(dataes[start_index: index])
                    labels.append(current)
                start_index = index
                current = m
        self.simple, self.marker = self.get_simple(labels, data)

    def deal_data(self, data, marker):
        # 舍弃99标签以前的数据(包括99标签) 、 92号以后的标签(包括91号)
        index_99 = np.max(np.where(marker == 99)[0])
        index_92 = np.min(np.where(marker == 91)[0])
        marker = marker[index_99 + 1: index_92]
        data = data[index_99 + 1: index_92]
        # 舍弃91号数据
        index_91 = np.where(marker == 91)[0]
        data = np.delete(data, index_91, axis=0)
        marker = np.delete(marker, index_91, axis=0)
        return data, marker

    def to_value(self, ds):
        return ds * self.std + self.mean

    def get_simple(self, labels, data):
        """
        生成样本
        :param labels: [-1, 1]
        :param data: [-1, -1, 22]
        :return: simple: [-1, 128, 22], marker : [-1, ]
        """
        length_size = self.config.simple_dim
        simple = []
        marker = []
        for i, label in enumerate(labels):
            ds = data[i]
            simple_num = len(ds) // length_size
            ds = ds[:length_size * simple_num]
            ds = np.reshape(ds, [-1, length_size, 22])  # [-1, 128, 22]
            simple.extend(ds)
            marker.extend(np.tile(np.array(label), [simple_num]))
        return simple, marker

    def get_train_ds(self):
        # 返回正常数据
        simple = []
        for i, marker in enumerate(self.marker):
            if marker == 0:
                simple.append(self.simple[i])
            # else:
            #     break
        return simple

    def get_test_ds(self):
        # 返回包含异常的数据
        simple = []
        for i, marker in enumerate(self.marker):
            if marker != 0:
                simple = self.simple[i:]
                break
        return simple


if __name__ == '__main__':
    config = Config()
    ds = Data(config)
    train_ds = ds.get_test_ds()
    # targets = [0, 1, 2, 3, 4, 5, 91, 92, 99]
    # for value in targets:
    #     data = ds.get_train_ds([value])
    #     mean = np.mean(data)
    #     std = np.std(data)
    #     print("{value}, mean:".format(value=value), mean)
    #     print("std : ", std)
    #     print("")
