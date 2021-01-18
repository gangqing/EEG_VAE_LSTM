
class Config:
    def __init__(self):
        self.vae_save_path = "models/{name}/vae_model.h5".format(name=self.get_name())
        self.encoder_save_path = "models/{name}/vae_encoder_model.h5".format(name=self.get_name())
        self.decoder_save_path = "models/{name}/vae_decoder_model.h5".format(name=self.get_name())
        self.lstm_save_path = "models/{name}/lstm_model.h5".format(name=self.get_name())
        self.log_dir = "logdir/{name}".format(name=self.get_name())
        self.simple_path = "data/5F-SubjectB-160311-5St-SGLHand-HFREQ.mat"
        # data params
        self.frame = 12
        self.simple_dim = 64
        self.num_dim = 22
        self.anomaly_threshold = 4.0
        # VAE params
        self.vae_epoch = 200
        self.vae_batch_size = 100
        self.vae_size = 128
        self.learning_rate_vae = 0.0002
        # LSTM params
        self.lstm_epoch = 80
        self.lstm_batch_size = 100
        self.num_hidden_units_lstm = 64
        self.learning_rate_lstm = 0.0001
        # 阈值
        self.sigma = 0.8

    def get_name(self):
        return "test05"

