import numpy as np 
from collections import defaultdict
from tqdm import tqdm
import random

infor_train_data_path = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/infor_train.npy', allow_pickle = True).tolist()
eva_infor_test_data_path = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/infor_test.npy', allow_pickle = True).tolist()
social_ratings = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/soc_ratings.npy', allow_pickle = True).tolist()
social_links = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/social_links.npy', allow_pickle = True).tolist()

item_count = 8626

print(min(infor_train_data_path.keys()), max(infor_train_data_path.keys()))
print(min(social_ratings.keys()), max(social_ratings.keys()))

eva_social_rating_test_data = defaultdict(list)
for user, friends in tqdm(social_ratings.items()):
    if user not in infor_train_data_path.keys():
        for i in social_ratings[user]:
            eva_social_rating_test_data[user].append(i)
        for _ in range(1000):
            j = random.randint(0,item_count)
            if j in eva_social_rating_test_data[user]:
                j = random.randint(0,item_count)
            eva_social_rating_test_data[user].append(j)

np.save('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/soc_test_1000.npy', eva_social_rating_test_data)
print('done!')
  
          


