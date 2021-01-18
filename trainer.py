from Models import VAE, LSTM
from Data5 import Data
from Config import Config
import numpy as np


def train_vae(cfg, ds: Data):
    vae = VAE(cfg)
    simple = ds.get_vae_train_data()
    test_ds = ds.get_vae_test_data()
    vae.train(np.array(simple), np.array(test_ds))


def test_vae(cfg, ds: Data):
    vae = VAE(cfg)
    test_ds = ds.get_vae_test_data()  # [-1, 128, 22]
    vae.test(np.array(test_ds))


def train_lstm(cfg, ds: Data):
    vae = VAE(cfg)
    lstm = LSTM(vae, cfg)
    lstm.train(ds.get_lstm_train_data())


def test(cfg, ds: Data):
    vae = VAE(cfg)
    lstm = LSTM(vae, cfg)
    test_data = ds.get_lstm_test_data()  # [-1, 12, 128, 22]
    test_marker = ds.get_lstm_test_marker()  # [-1, 12]
    num_example = len(test_data)
    test_lstm_recons_error = []
    for i in range(num_example):
        data = test_data[i]  # [12, 128, 22]
        mse = lstm.test(data)  # [11,]
        test_lstm_recons_error.append(mse)  # [-1, 11]
    threshold_list = np.linspace(np.amin(test_lstm_recons_error), np.amax(test_lstm_recons_error), 200,
                                 endpoint=True)
    threshold_list = np.flip(threshold_list)
    for threshold in threshold_list:
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        true_marker = [0]
        for i, mse in enumerate(test_lstm_recons_error):
            marker = test_marker[i]
            for j in range(cfg.frame - 1):
                is_true_anomaly = marker[j + 1] in true_marker
                is_pre_anomaly = mse[j] > threshold
                if is_true_anomaly and is_pre_anomaly:
                    TP += 1
                elif is_true_anomaly and not is_pre_anomaly:
                    FN += 1
                elif not is_true_anomaly and is_pre_anomaly:
                    FP += 1
                else:
                    TN += 1
        # 准确率
        P = 0 if TP + FP == 0 else float(TP) / (TP + FP)
        # 召回率
        R = 0 if TP + FN == 0 else float(TP) / (TP + FN)
        F1 = 0 if P + R == 0 else float(2 * P * R) / (P + R)
        print("阈值：", threshold)
        print("精准率:", P)
        print("召回率:", R)
        print("F1:", F1)
        print(float(TP + TN) / (TP + FN + FP + TN))


if __name__ == '__main__':
    config = Config()
    data = Data(config)
    train_vae(config, data)
    train_lstm(config, data)
    test(config, data)
    # test_vae(config, data)

