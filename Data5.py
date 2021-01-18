import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from Config import Config
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
基于Data4的基础上，做以下改动：
1、舍弃99标签以前的数据(包括99标签) 、 91号以后的标签(包括91号)
2、全局Z标准化；
3、去掉异常数据，z值大于4.0或者小于-4.0的值当作是异常值；
"""


def show(data, marker, path):
    plt.subplot(2, 1, 1)
    plt.plot([i for i in range(len(data))], data)

    plt.subplot(2, 1, 2)
    plt.plot([i for i in range(len(marker))], marker)

    # plt.show()
    plt.savefig(path)


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
        data, marker = self.load_mat_data()
        # show(data, np.reshape(marker, [-1]), "source.png")
        data, self.source_marker = self.deal_data(data, marker)
        # boxcox

        # 归一化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.source_data = (data - self.mean) / self.std  # [-1, 22]
        # show(data, np.reshape(self.source_marker, [-1]), None)
        # 划分VAE数据集
        train_data, train_marker, test_data, test_marker = self.train_test_data()
        self.vae_train_data = self.vae_data(train_data)
        # 划分LSTM数据集
        self.lstm_train_data, self.lstm_test_data, self.lstm_test_marker = self.lstm_data(train_data, test_data, test_marker)
        # 保存数据
        # self.save_data()

    def deal_data(self, data, marker):
        # 舍弃99标签以前的数据(包括99标签) 、 91号以后的标签(包括91号)
        index_99 = np.max(np.where(marker == 99)[0])
        index_91 = np.min(np.where(marker == 91)[0])
        marker = marker[index_99 + 1: index_91]
        data = data[index_99 + 1: index_91]
        # 舍弃91号数据
        # index_91 = np.where(marker == 91)[0]
        # data = np.delete(data, index_91, axis=0)
        # marker = np.delete(marker, index_91, axis=0)
        return data, marker

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

    def delete_anomaly(self, data, marker):
        """
        :param data: [-1, 22]
        :param marker: [-1, 1]
        :return:
        """
        delete_id = []
        for ids, x in enumerate(data):
            is_anomaly = np.logical_or.reduce([x > self.config.anomaly_threshold, x < -self.config.anomaly_threshold])
            if True in is_anomaly:
                delete_id.append(ids)
        data = np.delete(data, delete_id, axis=0)
        marker = np.delete(marker, delete_id, axis=0)
        return data, marker

    def vae_data(self, train_data):
        """
        vae数据集
        :return: [-1, 128, 22]
        """
        dim = self.config.simple_dim
        num_examples = len(train_data) - dim + 1
        data = []  # len : 158621
        for i in range(num_examples):
            data.append(train_data[i: i + dim])
        return np.array(data)

    def train_test_data(self):
        # 返回正常数据
        true_marker = [0]
        train_data = None
        train_marker = None
        test_data = None
        test_marker = None
        for i, marker in enumerate(self.source_marker):
            if marker not in true_marker:
                train_data = self.source_data[:i]
                train_marker = self.source_marker[:i]
                test_data = self.source_data[i:]
                test_marker = self.source_marker[i:]
                break
        train_data, train_marker = self.delete_anomaly(train_data[:2406], train_marker[:2406])
        test_data, test_marker = self.delete_anomaly(test_data, test_marker)
        # show(train_data, train_marker, "vae_trian.png")
        show(test_data[:2500], test_marker[:2500], "image/vae_test.png")
        return train_data, train_marker, test_data, test_marker

    def lstm_data(self, train_data, test_data, test_marker):
        """
        :param train_data:  [-1, 22]
        :param test_data:  [-1, 22]
        :param test_marker:  [-1, 1]
        :return:
        """
        # 训练数据
        train_data = self.vae_data(train_data)  # [-1, 64, 22]
        train_data, _ = self.W(train_data)  # [-1, 12, 64, 22]
        # 测试集处理
        test_data, test_marker = self.lstm_test_ds(test_data, test_marker)
        return train_data, test_data, test_marker

    def lstm_test_ds(self, data, marker):
        # 生成子样本
        dim = self.config.simple_dim  # 64
        sub_simple = []
        sub_marker = []
        num = len(data) // dim
        for i in range(num):
            sub_simple.append(data[i * dim: (i + 1) * dim])
            sub_marker.append(marker[i * dim: (i + 1) * dim])
        # 生成样本
        frame = self.config.frame  # 12
        simple = []
        marker = []
        # 样本，每12个子样本组成一个样本
        num_example = len(sub_simple) // frame
        for j in range(num_example):  # 遍历样本数
            simple.append([sub_simple[j * frame + i] for i in range(frame)])  # [12, 128, 22]
            marker.append([sub_marker[j * frame + i] for i in range(frame)])  # [-1, 12, 128, 1]
        marker = np.reshape(marker, [-1, frame, dim])  # [-1, 12, 128]
        return np.array(simple), np.max(marker, axis=2)

    def w(self, data, marker):
        """
        窗口滑动生成样本
        :param marker: [-1,]
        :param data: [-1, 22]
        :return: [-1, 128, 22], [-1, 128]
        """
        dim = self.config.simple_dim  # 128
        simple_data = []
        simple_marker = []
        num = len(data) - dim + 1
        for i in range(num):
            simple_data.append(data[i: i + dim])
            simple_marker.append(marker[i: i + dim])
        return np.array(simple_data), np.array(simple_marker)

    def W(self, w_data, w_marker=None):
        """
        :param w_marker: [-1, 128]
        :param w_data:  [-1, 128, 22]
        :return: [-1, 12, 128, 22], [-1, 12]
        """
        dim = self.config.simple_dim  # 128
        frame = self.config.frame  # 12
        simple = []
        marker = []
        # 样本，每12个子样本组成一个样本
        num_example = len(w_data) - dim * frame + 1
        for j in range(num_example):  # 遍历样本数
            sub_simple = [w_data[j + i * dim] for i in range(frame)]  # [12, 128, 22]
            simple.append(sub_simple)  # [-1, 12, 128, 22]
            if w_marker is not None:
                sub_marker = [w_marker[j + i * dim] for i in range(frame)]  # [12, 128, 1]
                marker.append(sub_marker)  # [-1, 12, 128, 1]
        if w_marker is not None:
            marker = np.reshape(marker, [-1, frame, dim])  # [-1, 12, 128]
            marker = np.max(marker, axis=2)
        return np.array(simple), marker

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
