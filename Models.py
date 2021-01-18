import tensorflow as tf
from tensorflow import keras
import numpy as np
from Config import Config
import matplotlib.pyplot as plt


def simple(mean, log_var):
    std = tf.exp(0.5 * log_var)
    eps = tf.random.normal(shape=tf.shape(log_var), dtype=log_var.dtype)
    return mean + std * eps


def math_pccd(x, y):
    """
    pearson相关系数(皮尔逊相关系数) : [-1, 1]之间，0代表无相关，-1代表负相关，1代表正相关
    :return [0, 1], 返回值越大越相似
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    d = np.sum((x - mean_x) * (y - mean_y)) / (
            np.sqrt(np.sum(np.square(x - mean_x))) * np.sqrt(np.sum(np.square(y - mean_y))))
    s = np.abs(d)  # 取绝对值，返回值在[0, 1]区间
    return s


def load_model(path):
    try:
        decoder_model = tf.keras.models.load_model(path)
        if decoder_model is None:
            assert "no model to loading!"
        return decoder_model
    except:
        assert "no model to loading!"


class Encoder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inputs = None
        self.model = None

    def init_net(self):
        activation = tf.nn.elu
        self.inputs = keras.Input(shape=[self.cfg.simple_dim, 22], dtype=tf.float32, name="vae_inputs")  # [None, 128, 22]
        base_filters = 32
        x = self.inputs
        for i in range(4):
            # [-1, 64, 22] -> [-1, 32, 32] -> [-1, 16, 64] -> [-1, 8, 128] -> [-1, 4, 256]
            x = keras.layers.Conv1D(filters=base_filters,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    activation=activation)(x)
            base_filters *= 2
        encoded_signal = keras.layers.Flatten()(x)  # 256 * 8
        encoded_signal = keras.layers.Dense(units=self.cfg.vae_size * 4,
                                            activation=activation)(encoded_signal)  # [-1, 1024]
        mean = keras.layers.Dense(units=self.cfg.vae_size)(encoded_signal)  # [-1, 128]
        log_var = keras.layers.Dense(units=self.cfg.vae_size)(encoded_signal)  # [-1, 128]
        self.model = keras.Model(inputs=self.inputs, outputs=[mean, log_var])

    def __call__(self, x):
        return self.model(x)

    def get_inputs(self):
        return self.inputs

    def save(self):
        self.model.save(filepath=self.cfg.encoder_save_path)

    def load(self):
        if self.model is None:
            self.model = load_model(self.cfg.encoder_save_path)

    def predict(self, inputs):
        return self.model.predict(inputs)


class Decoder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None

    def init_net(self):
        activation = tf.nn.elu
        simple = keras.Input(shape=[self.cfg.vae_size], dtype=tf.float32)
        dense_0 = keras.layers.Dense(units=self.cfg.vae_size * 4)(simple)  # [-1, 1024]
        dense_1 = keras.layers.Dense(units=256 * 4, activation=activation)(dense_0)  # [-1, 256 * 8]
        x = keras.layers.Reshape([4, 256])(dense_1)  # [-1, 8, 256]

        base_filters = 128
        for i in range(3):
            # [-1, 4, 256] -> [-1, 8, 128] -> [-1, 16, 64] -> [-1, 32, 32]
            x = keras.layers.Conv1DTranspose(filters=base_filters,
                                             kernel_size=3,
                                             strides=2,
                                             padding="same",
                                             activation=activation)(x)
            base_filters //= 2
        y = keras.layers.Conv1DTranspose(filters=22,
                                         kernel_size=3,
                                         strides=2,
                                         padding="same",
                                         activation=None)(x)  # [-1, 128, 22]
        self.model = keras.Model(inputs=simple, outputs=y)

    def __call__(self, x):
        return self.model(x)

    def save(self):
        self.model.save(filepath=self.cfg.decoder_save_path)

    def load(self):
        if self.model is None:
            self.model = load_model(self.cfg.decoder_save_path)

    def predict(self, mean):
        return self.model.predict(mean)


class VAE:
    def __init__(self, c: Config):
        self.config = c
        self.encoder = Encoder(c)
        self.decoder = Decoder(c)
        self.vae = None

    def init_net(self):
        self.encoder.init_net()
        self.decoder.init_net()
        inputs = self.encoder.get_inputs()
        mean, log_var = self.encoder(inputs)
        z = simple(mean, log_var)
        outputs = self.decoder(z)
        self.vae = keras.Model(inputs=inputs, outputs=outputs)
        self.vae.summary()
        mse_loss = tf.reduce_mean(tf.square(inputs - outputs))
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.pow(mean, 2) - tf.exp(log_var))
        self.vae.add_loss(mse_loss + kl_loss)

    def train(self, inputs, test_ds):
        self.init_net()
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate_vae))
        callbacks = tf.keras.callbacks.TensorBoard(log_dir=self.config.log_dir)
        history = self.vae.fit(inputs,
                               batch_size=self.config.vae_batch_size,
                               epochs=self.config.vae_epoch,
                               callbacks=callbacks,
                               validation_split=0.1)
        print("history:", history.history)
        self.vae.save(filepath=self.config.vae_save_path)
        self.encoder.save()
        self.decoder.save()
        outputs = self.vae.predict(inputs)
        print("train_pccd:", math_pccd(inputs, outputs))

        outputs = self.vae.predict(test_ds)
        print("test_pccd:", math_pccd(test_ds, outputs))

    def test(self, test_ds):
        self.encoder.load()
        mean = self.encoder.predict(test_ds)
        self.decoder.load()
        y = self.decoder.predict(mean)
        print("test_pccd:", math_pccd(test_ds, y))
        return y

    def predict_encoder(self, test_ds):
        """
        预测中间隐藏层的值 -- mean
        :param test_ds: [-1, 128, 22]
        :return: [-1, 128]
        """
        self.encoder.load()
        mean, log_var = self.encoder.predict(test_ds)
        return mean

    def predict_decoder(self, mean):
        """
        :param mean: [-1, 128]
        :return:
        """
        self.decoder.load()
        return self.decoder.predict(mean)


class LSTM:
    def __init__(self, vae: VAE, c: Config):
        self.vae = vae
        self.config = c
        self.model = None

    def init_lstm_net(self):
        cfg = self.config
        inputs = tf.keras.layers.Input([cfg.frame, self.config.vae_size])  # [-1, 12, 128]
        LSTM1 = tf.keras.layers.LSTM(self.config.num_hidden_units_lstm, return_sequences=True)(
            inputs)  # [-1, 12, 1024]
        LSTM2 = tf.keras.layers.LSTM(self.config.num_hidden_units_lstm, return_sequences=True)(
            LSTM1)  # [-1, 12, 1024]
        outputs = tf.keras.layers.LSTM(self.config.vae_size, return_sequences=True, activation=None)(
            LSTM2)  # [-1, 12, 128]
        self.model = tf.keras.Model(inputs, outputs)
        self.model.summary()
        # todo
        loss_sum = tf.reduce_sum(tf.square(outputs[:-1] - inputs[1:]), axis=2)
        loss = tf.reduce_mean(tf.sqrt(loss_sum))
        self.model.add_loss(loss)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate_lstm))

    def vae_encoder(self, data):
        """
        通过vae的encoder生成lstm的数据
        :param data: [-1, 12, 128, 22]
        :return: [-1, 12, 128]
        """
        cfg = self.config
        data = np.reshape(data, [-1, cfg.simple_dim, cfg.num_dim])
        mean, _ = self.vae.predict_encoder(data)  # [-1, 128]
        return np.reshape(mean, [-1, cfg.frame, cfg.vae_size])  # [-1, 12, 256]

    def train(self, lstm_train_data):
        means = self.vae_encoder(lstm_train_data)  # [-1, 12, 128]
        self.init_lstm_net()
        callbacks = tf.keras.callbacks.TensorBoard(log_dir=self.config.log_dir)
        self.model.fit(means,
                       batch_size=self.config.lstm_batch_size,
                       epochs=self.config.lstm_epoch,
                       callbacks=callbacks,
                       validation_split=0.1)
        self.model.save(filepath=self.config.lstm_save_path)

    def test(self, x):
        """
        :param x: [12, 128, 22]
        :param y: [11, 128, 22]
        :return:  # [11,]
        """
        # x -> vae_encoder -> vae_mean
        vae_mean = self.vae.predict_encoder(x)  # [12, 128]
        # vae_mean -> lstm -> lstm_mean
        vae_mean = np.reshape(vae_mean, [-1, self.config.frame, self.config.vae_size])  # [1, 12, 128]
        if self.model is None:
            self.model = load_model(self.config.lstm_save_path)
        lstm_mean = self.model.predict(vae_mean)  # [1, 12, 128]
        lstm_mean = np.reshape(lstm_mean[:, :-1, :], [-1, self.config.vae_size])  # [11, 128]
        # lstm_mean -> vae_decoder -> predict
        lstm_predict = self.vae.predict_decoder(lstm_mean)  # [11, 128, 22]
        # 计算mse
        mse = np.sqrt(np.sum(np.square(lstm_predict - x[1:]), axis=(1, 2)))  # [11,]
        # mse大于某个阈值，那么该数值是异常值
        return mse
