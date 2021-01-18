from tensorflow import keras
import tensorflow as tf
import numpy as np
import scipy.io as sio
import random


class Data:
    def __init__(self):
        self.source_data = None  # [-1, 64, 22]
        self.source_marker = None  # [-1, 1]
        self.mean = None
        self.std = None
        self.create_data()

    def create_data(self):
        data, marker = self.load_mat_data()
        data, marker = self.deal_data(data, marker)
        # 归一化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        data = (data - self.mean) / self.std  # [-1, 22]
        #
        length = 64
        num = len(data) // length
        new_data = []
        new_marker = []
        for i in range(num):
            new_data.append(data[i * length: i * length + length])  # [num, length, 22]
            new_marker.append(marker[i * length: i * length + length])  # [num, length, 1]
        self.source_data = np.array(new_data)
        self.source_marker = np.min(new_marker, axis=(1, 2))

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
        simple_path = "data/5F-SubjectB-160311-5St-SGLHand-HFREQ.mat"
        file = sio.loadmat(simple_path)
        data = file["o"]["data"][0][0]  # [-1, 22]
        data = np.array(data).astype(np.float)

        marker = file["o"]["marker"][0][0]  # [-1, 1]
        marker = np.array(marker).astype(np.int)
        return data, marker


class Model:
    def __init__(self):
        self.model = None
        self.init_net()

    def init_net(self):
        inputs = keras.Input([64, 22], dtype=tf.float32, name="inputs")  # [-1, 64, 22]
        t1 = keras.layers.Dense(32)(inputs)  # [-1, 64, 32]

        t3 = keras.layers.Conv1D(16, 3, 1, padding="same", activation=tf.nn.relu, name="t_conv1")(t1)  # [-1, 64, 16]
        t3 = keras.layers.AvgPool1D(pool_size=3, strides=2, padding="same")(t3)  # [-1, 32, 16]
        t3 = keras.layers.Dropout(0.2)(t3)
        t4 = keras.layers.Conv1D(32, 3, 1, padding="same", activation=tf.nn.relu, name="t_conv2")(t3)  # [-1,, 16, 128]
        t4 = keras.layers.AvgPool1D(pool_size=3, strides=2, padding="same")(t4)  # [-1, 32, 16]
        t4 = keras.layers.Dropout(0.2)(t4)
        t5 = keras.layers.Conv1D(64, 3, 1, padding="same", activation=tf.nn.relu, name="t_conv3")(t4)  # [-1, 8, 256]
        t5 = keras.layers.AvgPool1D(pool_size=3, strides=2, padding="same")(t5)  # [-1, 32, 16]
        t5 = keras.layers.Dropout(0.2)(t5)
        t5 = keras.layers.Conv1D(128, 3, 1, padding="same", activation=tf.nn.relu, name="t_conv4")(t5)  # [-1, 4, 512]
        t5 = keras.layers.AvgPool1D(pool_size=3, strides=2, padding="same")(t5)  # [-1, 32, 16]
        t5 = keras.layers.Dropout(0.2)(t5)
        # t5 = keras.layers.Conv1D(256, 3, 1, padding="same", activation=tf.nn.relu, name="t_conv5")(t5)  # [-1, 2, 1024]
        # t5 = keras.layers.AvgPool1D(pool_size=3, strides=2, padding="same")(t5)  # [-1, 32, 16]
        # t5 = keras.layers.Dropout(0.2)(t5)
        # t5 = keras.layers.Conv1D(512, 3, 2, padding="same", activation=tf.nn.relu, name="t_conv6")(t5)  # [-1, 1, 2048]
        # t5 = keras.layers.AvgPool1D(pool_size=3, strides=2, padding="same")(t5)  # [-1, 32, 16]
        # t5 = keras.layers.Dropout(0.2)(t5)
        t6 = keras.layers.Flatten(name="t_flatten")(t5)  # [-1, 2048]
        outputs = keras.layers.Dense(6, activation=tf.nn.softmax, name="h_dense2")(t6)  # [-1, 6]

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                           loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])

    def train(self, data, marker):
        """
        :param data: [-1, length, 22]
        :param marker: # [-1, 1]
        :return:
        """
        history = self.model.fit(data,
                                 marker,
                                 batch_size=1000,
                                 epochs=100,
                                 validation_split=0.1)
        # self.model.save(filepath="models/classify_models.h5")

    def test(self, ds):
        pass


def separate_train_and_val_set(n_win):
    n_train = int(np.floor((n_win * 0.9)))  # 训练集样本量
    idx_train = random.sample(range(n_win), n_train)  # 随机抽取样本作为训练集，不排序
    idx_val = list(set(idx_train) ^ set(range(n_win)))  # 其余的作为验证集，排序
    return idx_train, idx_val


if __name__ == '__main__':
    data = Data()
    ds = data.source_data  # [21231, 64, 22]
    marker = data.source_marker  # [21231,]
    count = 13182 - 1400
    index_0 = np.where(marker == 0)[0][:-1400]
    ds = np.delete(ds, index_0, axis=0)
    marker = np.delete(marker, index_0, axis=0)
    mask = np.unique(marker)
    for m in mask:
        t = np.sum(marker == m)
        print(m, t, float(t)/len(marker))

    id_x = random.sample(range(len(marker)), len(marker))
    ds = ds[id_x]
    marker = marker[id_x]
    # idx_train, idx_val = separate_train_and_val_set(len(marker))
    # train_data = ds[idx_train]
    # train_marker = marker[idx_train]
    # test_data = ds[idx_val]
    # test_marker = marker[idx_val]

    model = Model()
    model.train(ds, marker)
