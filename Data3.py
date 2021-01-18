import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from Config import Config

"""
数据加载和处理，生成训练样本：
1、99标签之前的数据(全0标签数据)作为训练数据， 去掉前1502条0标签数据；
2、99标签之后的数据(包括99标签数据)作为测试数据；
3、按照论文一样生成VAE训练数据，每隔一个数据点就生成一条样本；
4、生成LSTM数据，每隔128个数据点生成一个子样本，连继的子样本生成样本；
5、按维度进行Z标准化；
"""


def show(data, marker):
    plt.subplot(2, 1, 1)
    plt.plot([i for i in range(len(data))], data)

    plt.subplot(2, 1, 2)
    plt.plot([i for i in range(len(marker))], marker)

    plt.show()


class Data:
    def __init__(self, cfg: Config):
        self.config = cfg
        self.vae_train_data_path = "data/vae_train_data.dat"
        self.vae_test_data_path = "data/vae_test_data.dat"
        self.lstm_train_data_path = "data/lstm_train_data.dat"
        self.lstm_train_marker_path = "data/lstm_train_marker.dat"
        self.lstm_test_data_path = "data/lstm_test_data.dat"
        self.lstm_test_marker_path = "data/lstm_test_marker.dat"
        self.source_data = None  # [-1, 22]
        self.source_marker = None  # [-1, 1]
        self.mean = None
        self.std = None
        self.vae_train_data = None  # [-1, 128, 22]
        self.vae_test_data = None  # [-1, 128, 22]
        self.lstm_train_data = None  # [-1, 12, 128, 22]
        self.lstm_train_marker = None  # [-1, 12]
        self.lstm_test_data = None  # [-1, 12, 128, 22]
        self.lstm_test_marker = None  # [-1, 12]
        # todo
        self.create_data()

    def create_data(self):
        data, self.source_marker = self.load_mat_data()
        # 归一化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.source_data = (data - self.mean) / self.std
        # sub_simple, sub_marker = self.sub_simple()
        # 划分VAE数据集
        self.vae_train_data, self.vae_test_data = self.vae_data()
        # 划分LSTM数据集
        self.lstm_train_data, self.lstm_train_marker, self.lstm_test_data, self.lstm_test_marker = self.lstm_data()
        # 保存数据
        # self.save_data()

    def sub_simple(self):
        cfg = self.config
        num_examples = len(self.source_data) - cfg.simple_dim + 1
        sub_simple = []
        sub_marker = []
        for i in range(num_examples):
            simple = self.source_data[i: i + cfg.simple_dim]  # [128, 22]
            sub_simple.append(simple)  # [num_examples, 128, 22]
            marker = self.source_marker[i: i + cfg.simple_dim]  # [128, 1]
            sub_marker.append(marker)  # [num_examples, 128, 1]
        return np.array(sub_simple), np.array(sub_marker)

    def load_mat_data(self):
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
        target_data = self.target_ds([0])
        num_examples = len(target_data) - dim + 1
        data = []  # len : 158621
        for i in range(num_examples):
            data.append(target_data[i: i + dim])
        index = int(len(data) * 0.6)
        return data[: index], data[index:]

    def target_ds(self, target=None):
        # 返回正常数据
        if target is None:
            target = [0, 1, 2, 3, 4, 5]
        simple = []
        for i, marker in enumerate(self.source_marker):
            if i < 1502:
                continue
            if marker in target:
                simple.append(self.source_data[i])
            else:
                break
        return simple

    def lstm_data(self):
        """
        :return: data : [-1, 12, 128, 22]; marker : [-1, 12]
        """
        train_data = None
        train_marker = None
        test_data = None
        test_marker = None
        # 划分训练集和测试集，使用marker为0的数据作为训练集
        for i, marker in enumerate(self.source_marker):  # marker : [1,]
            if marker != 0:
                train_data = self.source_data[1502: i]  # [-1, 22]
                train_marker = self.source_marker[1502: i]  # [-1,]
                test_data = self.source_data[i:]
                test_marker = self.source_marker[i:]
                break
        # 训练集处理
        train_data, train_marker = self.win_simple_from_data(train_data, train_marker, True)
        print("lstm_train_num_examples:", len(train_data))
        # 测试集处理
        test_data, test_marker = self.win_simple_from_data(test_data, test_marker, False)
        print("lstm_test_num_examples:", len(test_data))
        return train_data, train_marker, test_data, test_marker

    def win_simple_from_data(self, data, marker, is_train_data=True):
        """
        窗口滑动生成样本
        :param is_train_data:
        :param marker: [-1,]
        :param data: [-1, 22]
        :return: [-1, 12, 128, 22], [-1, 12]
        """
        dim = self.config.simple_dim  # 128
        simple_data = []
        simple_marker = []
        for i in range(1):
            # 子样本数，每128个数据组成一个子样本
            sub_num_example = (len(data) - i) // dim
            sub_simple = []
            sub_marker = []
            for j in range(sub_num_example):  # 遍历子样本数
                sub_data = data[dim * j + i: dim * (j + 1) + i]  # [128, 22]
                sub_simple.append(sub_data)  # [-1, 128, 22]
                sub_marker.append(marker[dim * j + i: dim * (j + 1) + i])
            simple, sub_marker = self.win_simple_from_sub_simple(sub_simple, sub_marker)  # [-1, 12, 128, 22]
            if i == 0:
                simple_data = simple
                simple_marker = sub_marker
            else:
                simple_data = np.concatenate((simple_data, simple), axis=0)
                simple_marker = np.concatenate((simple_marker, sub_marker), axis=0)
            if not is_train_data:
                break
        return simple_data, simple_marker

    def win_simple_from_sub_simple(self, sub_data, sub_marker):
        """
        在子样本中，窗口滑动生成样本
        :param sub_marker: [-1, 128]
        :param sub_data:  [-1, 128, 22]
        :return: [-1, 12, 128, 22], [-1, 12]
        """
        frame = self.config.frame  # 12
        simple = []
        marker = []
        # 样本，每12个子样本组成一个样本
        num_example = len(sub_data) - frame + 1
        for j in range(num_example):  # 遍历样本数
            simple_data = sub_data[j: j + frame]  # [12, 128, 22]
            simple.append(simple_data)  # [-1, 12, 128, 22]
            marker.append(sub_marker[j: j + frame])  # [-1, 12, 128, 1]
        marker = np.reshape(marker, [-1, frame, self.config.simple_dim])
        return np.array(simple), np.max(marker, axis=2)

    def to_value(self, data):
        if self.mean is None:
            self.get_param()
        return data * self.std + self.mean

    def get_param(self):
        data, source_marker = self.load_mat_data()
        # 归一化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def save_data(self):
        np.array(self.vae_train_data).tofile(self.vae_train_data_path, sep=",", format="%f")
        np.array(self.vae_test_data).tofile(self.vae_test_data_path, sep=",", format="%f")
        np.array(self.lstm_train_data).tofile(self.lstm_train_data_path, sep=",", format="%f")
        np.array(self.lstm_train_marker).tofile(self.lstm_train_marker_path, sep=",", format="%f")
        np.array(self.lstm_test_data).tofile(self.lstm_test_data_path, sep=",", format="%f")
        np.array(self.lstm_test_marker).tofile(self.lstm_test_marker_path, sep=",", format="%f")

    def get_vae_train_data(self):
        if self.vae_train_data is None:
            self.vae_train_data = np.reshape(np.fromfile(self.vae_train_data_path, dtype=np.float, sep=","),
                                             [-1, self.config.simple_dim, self.config.num_dim])
        return self.vae_train_data

    def get_vae_test_data(self):
        if self.vae_test_data is None:
            self.vae_test_data = np.reshape(np.fromfile(self.vae_test_data_path, dtype=np.float, sep=","),
                                            [-1, self.config.simple_dim, self.config.num_dim])
        return self.vae_test_data

    def get_lstm_train_data(self):
        if self.lstm_train_data is None:
            self.lstm_train_data = np.reshape(np.fromfile(self.lstm_train_data_path, dtype=np.float, sep=","),
                                              [-1, self.config.frame, self.config.simple_dim, self.config.num_dim])
        return self.lstm_train_data

    def get_lstm_train_marker(self):
        if self.lstm_train_marker is None:
            self.lstm_train_marker = np.reshape(np.fromfile(self.lstm_train_marker_path, dtype=np.float, sep=","),
                                                [-1, self.config.frame])
        return self.lstm_train_marker

    def get_lstm_test_data(self):
        if self.lstm_test_data is None:
            self.lstm_test_data = np.reshape(np.fromfile(self.lstm_test_data_path, dtype=np.float, sep=","),
                                             [-1, self.config.frame, self.config.simple_dim, self.config.num_dim])
        return self.lstm_test_data

    def get_lstm_test_marker(self):
        if self.lstm_test_marker is None:
            self.lstm_test_marker = np.reshape(np.fromfile(self.lstm_test_marker_path, dtype=np.float, sep=","),
                                               [-1, self.config.frame])
        return self.lstm_test_marker


if __name__ == '__main__':
    config = Config()
    ds = Data(config)
