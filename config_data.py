####################5############################
data_name = 'dianping'
out_path = '/content/drive/MyDrive/DASR-WGAN/log/dianping'
save_path = '/content/drive/MyDrive/DASR-WGAN/src/dianping/Final_Model_augmentation_edge/model.pt'
save_path_2 = '/content/drive/MyDrive/DASR-WGAN/src/dianping/Final_Model_augmentation_edge/model_1.pt'
save_path_bri = '/content/drive/MyDrive/DASR-WGAN/src/dianping/Final_Model_augmentation_edge/model_2.pt'
num_users = 10182 #count
num_items = 8627 #count 

num_bri_user_start = 8486
num_soc_user = 5296
num_common_user = 1303
num_train_neg = 1

num_all_user_id = 14175 #count
num_eva_neg = 1000

mf_dim = 32
learning_rate = 0.01
weight_decay = 0

train_epoch = 2000
topk = 10

