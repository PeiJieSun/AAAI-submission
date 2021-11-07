import torch
import torch.utils.data as data
import numpy as np 
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import config_data as conf
import random

infor_train_data_path = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/infor_train.npy', allow_pickle = True).tolist()
eva_infor_test_data_path = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/infor_test.npy', allow_pickle = True).tolist()
social_ratings = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/soc_ratings.npy', allow_pickle = True).tolist()
social_links = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/social_links.npy', allow_pickle = True).tolist()
soc_test = np.load('/content/drive/MyDrive/DASR-WGAN/data/dianping/version3/soc_test_1000.npy', allow_pickle = True).tolist()


infor_fake = np.load('/content/drive/MyDrive/DASR-WGAN/src/dianping/Final_Model_augmentation_edge/data/edge_modification/infor_fake.npy', allow_pickle = True).tolist()
social_fake = np.load('/content/drive/MyDrive/DASR-WGAN/src/dianping/Final_Model_augmentation_edge/data/edge_modification/fake_links_2.npy', allow_pickle = True).tolist()




def load_all():
    max_user, max_item, max_user_soc = 0, 0, 0

    ##################################################################
    '''
    information domain
    '''
    ##################################################################

    infor_hash_data = set()
    infor_rating_train_data = []
    infor_item_dict  = defaultdict(set)
    infor_user_dict = infor_train_data_path
    all_user = set()
    infor_user = set()
    for user, items in tqdm(infor_train_data_path.items()):
        all_user.add(user)
        infor_user.add(user)
        for item in items:
            infor_hash_data.add((user, item))
            infor_rating_train_data.append([user, item])
            infor_item_dict[item].add(user)  


    eva_infor_rating_test_data = []
    for user, items in tqdm(eva_infor_test_data_path.items()):
        for item in items:
            eva_infor_rating_test_data.append([user, item])


    eva_common_rating_test_data = []
    common_rating_dict = defaultdict(set)
    for user, friends in social_links.items():
        if user in eva_infor_test_data_path:
            for i in eva_infor_test_data_path[user]:
                eva_common_rating_test_data.append([user, i])
                common_rating_dict[user].add(i)


    social_link_dict = social_links
    link_hash_data = set()
    link_train_data = []
    soc_user = set()
    for user, friends in social_link_dict.items():
        all_user.add(user)
        soc_user.add(user)
        for friend in friends:
            all_user.add(friend)
            soc_user.add(friend)
            link_hash_data.add((user, friend))
            link_train_data.append([user, friend])


    eva_social_rating_test_data = []
    for user, friends in tqdm(social_ratings.items()):
        if user not in common_rating_dict:
            for i in social_ratings[user]:
                if i <= 8626:
                    eva_social_rating_test_data.append([user, i])


    # infor_fake_dict = infor_fake
    # social_fake_dict = social_fake
    # infor_fake_item_dict = defaultdict(set)
    # for user,items in infor_fake.items():
    #     for i in items:
    #         infor_fake_item_dict[i].add(user)


    eva_soc_test = soc_test
    eva_soc_test_ground_truth = social_ratings

    print('all user count:', len(all_user))
    print('max  user id:', max(all_user))
    print('infor user count:', len(infor_user))
    print('max infor user id:', max(infor_user))
    print('common user count:', len(common_rating_dict))
    print('soc user count:', len(soc_user))
    print('max soc user id:', max(soc_user))


    #import sys;sys.exit(0)
    return infor_hash_data, infor_rating_train_data, eva_infor_rating_test_data,\
     infor_user_dict, infor_item_dict, eva_common_rating_test_data, social_link_dict, eva_social_rating_test_data,\
     link_train_data, link_hash_data, eva_soc_test, eva_soc_test_ground_truth, common_rating_dict
        
####################edge modification################
def load_corrupt_edge():
    infor_train_data_path_cp = infor_train_data_path
    social_links_cp = social_links
    for _ in range(1000):
        u = random.randint(0, 10181)
        if len(infor_train_data_path_cp[u]) > 0:
            u_value = list(infor_train_data_path_cp[u])
            u_value_len = len(u_value)
            i = random.randint(0,u_value_len-1)
            try:
                infor_train_data_path_cp[u].remove(u_value[i])
            except:
                import pdb;pdb.set_trace()


    for _ in range(1000):
        u = random.randint(0, 10181)
        i = random.randint(0, 8626)
        if i not in infor_train_data_path_cp[u]:
            infor_train_data_path_cp[u].add(i)

    ########add edges in social domain
    for _ in range(500):
        u = random.randint(8486, 14174)
        if len(social_links_cp[u]) > 0:
            u_value = list(social_links_cp[u])
            u_value_len = len(u_value)
            i = random.randint(0,u_value_len-1)
            try:
                social_links_cp[u].remove(u_value[i])
            except:
                pdb.set_trace()

    for _ in range(500):
        u1 = random.randint(8486, 14174)
        u2 = random.randint(8486, 14174)
        if u2 not in social_links_cp[u1]:
            social_links_cp[u1].add(u2)

    infor_fake_item_dict_cp = defaultdict(set)
    for user,items in infor_train_data_path_cp.items():
        for i in items:
            infor_fake_item_dict_cp[i].add(user)
    
    return infor_train_data_path_cp, social_links_cp, infor_fake_item_dict_cp

################node mask################
def load_corrupt_node_mask():
    infor_train_data_path_cp = infor_train_data_path
    social_links_cp = social_links
    for _ in range(1000):
        u = random.randint(0, 10181)
        if len(infor_train_data_path_cp[u]) > 0:
            u_value = list(infor_train_data_path_cp[u])
            u_value_len = len(u_value)
            i = random.randint(0,u_value_len-1)
            try:
                infor_train_data_path_cp[u].remove(u_value[i])
            except:
                import pdb;pdb.set_trace()


    for _ in range(1000):
        u = random.randint(0, 10181)
        i = random.randint(0, 8626)
        if i not in infor_train_data_path_cp[u]:
            infor_train_data_path_cp[u].add(i)

    ########add edges in social domain
    for _ in range(300):
        u = random.randint(8486, 14174)
        if len(social_links_cp[u]) > 0:
            u_value = list(social_links_cp[u])
            u_value_len = len(u_value)
            i = random.randint(0,u_value_len-1)
            try:
                social_links_cp[u].remove(u_value[i])
            except:
                pdb.set_trace()

    for _ in range(300):
        u1 = random.randint(8486, 14174)
        u2 = random.randint(8486, 14174)
        if u2 not in social_links_cp[u1]:
            social_links_cp[u1].add(u2)

    infor_fake_item_dict_cp = defaultdict(set)
    for user,items in infor_train_data_path_cp.items():
        for i in items:
            infor_fake_item_dict_cp[i].add(user)
    
    return infor_train_data_path_cp, social_links_cp, infor_fake_item_dict_cp



# construct original local graph#
def construct_infor_mat(soc_dict, user_dict, item_dict, is_user):
    if is_user == True:
        infor_index, infor_value = [], []
        #common user
        #'''
        for user in soc_dict.keys():
            friends_list = soc_dict[user]
            if user not in user_dict:
                for f in friends_list:
                    fri_friend = soc_dict[f]
                    infor_index.append([user, f])
                    #infor_value.append(1.0/(np.sqrt(len(friends_list)*len(fri_friend))))
                    infor_value.append(1.0/len(friends_list))

        #'''
        for user in user_dict.keys():
            item_list = user_dict[user]
            if user not in soc_dict:
                for i in item_list:
                    user_list = item_dict[i]
                    infor_index.append([user, i+conf.num_all_user_id])
                    #infor_value.append(1.0/(np.sqrt(len(item_list)*len(user_list))))
                    infor_value.append(1.0/len(item_list))
        
        for user in user_dict.keys():
            if user in soc_dict.keys():
                friends_list = soc_dict[user]
                item_list = user_dict[user]
                #'''
                for f in friends_list:
                    fri_friend = soc_dict[f]
                    infor_index.append([user, f])
                    #infor_value.append(1.0/(np.sqrt(len(friends_list)*len(fri_friend))))
                    infor_value.append(1.0/(len(friends_list)+len(item_list)))
                
                for i in item_list:
                    user_list = item_dict[i]
                    infor_index.append([user, i+conf.num_all_user_id])
                    #infor_value.append(1.0/(np.sqrt(len(item_list)*len(user_list))))
                    infor_value.append(1.0/(len(item_list)+len(friends_list)))

        #'''
        for item in item_dict.keys():
            user_list = item_dict[item]
            for u in user_list:
                item_list = user_dict[u]
                infor_index.append([item + conf.num_all_user_id, u])
                #infor_value.append(1.0/(np.sqrt(len(user_list)*len(item_list))))
                infor_value.append(1.0/len(user_list))
        
        length = conf.num_all_user_id + conf.num_items
        user_agg_mat = torch.sparse.FloatTensor(torch.LongTensor(infor_index).t().cuda(), \
            torch.FloatTensor(infor_value).cuda(), torch.Size([length, length]))#.to_dense()  
        #import pdb;pdb.set_trace()
        return user_agg_mat


############construct corrupted local graph###############
def construct_corrupted_graph(infor_fake_dict, infor_fake_item_dict, social_fake_dict):
    infor_index, infor_value = [], []
    #common user
    #'''
    for user in infor_fake_dict.keys():
        item_list = infor_fake_dict[user]
        for i in item_list:
            infor_index.append([user, i+conf.num_all_user_id])
            #infor_value.append(1.0/(np.sqrt(len(item_list)*len(user_list))))
            infor_value.append(1.0/len(item_list))
    
    for item in infor_fake_item_dict.keys():
        user_list = infor_fake_item_dict[item]
        for u in user_list:
            infor_index.append([item+conf.num_all_user_id, u])
            infor_value.append(1.0/len(user_list))
    
    for user in social_fake_dict.keys():
        friend_list = social_fake_dict[user]
        for f in friend_list:
            infor_index.append([user, f])
            infor_value.append(1.0/len(friend_list))

    length = conf.num_all_user_id + conf.num_items
    fake_agg_mat = torch.sparse.FloatTensor(torch.LongTensor(infor_index).t().cuda(), \
        torch.FloatTensor(infor_value).cuda(), torch.Size([length, length]))#.to_dense()  
    #import pdb;pdb.set_trace()
    return fake_agg_mat


##############3construct global in social domian################
def construct_global_social(soc_dict):
    infor_index, infor_value = [], []
    for user in soc_dict.keys():
        friends_list = soc_dict[user]
        for f in friends_list:
            fri_friend = soc_dict[f]
            infor_index.append([user, f])
            #infor_value.append(1.0/(np.sqrt(len(friends_list)*len(fri_friend))))
            infor_value.append(1.0/len(friends_list))
    length = conf.num_all_user_id + conf.num_items
    user_agg_mat = torch.sparse.FloatTensor(torch.LongTensor(infor_index).t().cuda(), \
        torch.FloatTensor(infor_value).cuda(), torch.Size([length, length]))#.to_dense()  
    #import pdb;pdb.set_trace()
    return user_agg_mat

#construct global in information domain#
def construct_global_infor(user_dict, item_dict):
    infor_index, infor_value = [], []
    for user in user_dict.keys():
        item_list = user_dict[user]
        for i in item_list:
            user_list = item_dict[i]
            infor_index.append([user, i+conf.num_all_user_id])
            #infor_value.append(1.0/(np.sqrt(len(item_list)*len(user_list))))
            infor_value.append(1.0/len(item_list))

    for item in item_dict.keys():
        user_list = item_dict[item]
        for u in user_list:
            item_list = user_dict[u]
            infor_index.append([item + conf.num_all_user_id, u])
            #infor_value.append(1.0/(np.sqrt(len(user_list)*len(item_list))))
            infor_value.append(1.0/len(user_list))
        
    length = conf.num_all_user_id + conf.num_items
    user_agg_mat = torch.sparse.FloatTensor(torch.LongTensor(infor_index).t().cuda(), \
        torch.FloatTensor(infor_value).cuda(), torch.Size([length, length]))#.to_dense()  
    #import pdb;pdb.set_trace()
    return user_agg_mat




#
def construct_infor_fake_graph(infor_fake_dict, infor_fake_item_dict, soc_dict, is_true):
    infor_index, infor_value = [], []
    #common user
    #'''
    for user in infor_fake_dict.keys():
        item_list = infor_fake_dict[user]
        for i in item_list:
            infor_index.append([user, i+conf.num_all_user_id])
            #infor_value.append(1.0/(np.sqrt(len(item_list)*len(user_list))))
            infor_value.append(1.0/len(item_list))
    
    for item in infor_fake_item_dict.keys():
        user_list = infor_fake_item_dict[item]
        for u in user_list:
            infor_index.append([item+conf.num_all_user_id, u])
            infor_value.append(1.0/len(user_list))

    if is_true == True:
        for user in soc_dict.keys():
            friend_list = soc_dict[user]
            for f in friend_list:
                infor_index.append([user, f])
                infor_value.append(1.0/len(friend_list))
    
    length = conf.num_all_user_id + conf.num_items
    fake_infor_agg_mat = torch.sparse.FloatTensor(torch.LongTensor(infor_index).t().cuda(), \
        torch.FloatTensor(infor_value).cuda(), torch.Size([length, length]))#.to_dense()  
    #import pdb;pdb.set_trace()
    return fake_infor_agg_mat
        

##########construct fake graph in social domain#############
def construct_social_fake_graph(social_fake_dict, user_dict, item_dict, is_true):
    social_index,social_value = [],[]
    for user in social_fake_dict.keys():
        friend_list = social_fake_dict[user]
        for f in friend_list:
            social_index.append([user, f])
            social_value.append(1.0/len(friend_list))
    
    if is_true == True:
        for user in user_dict.keys():
            item_list = user_dict[user]
            for i in item_list:
                social_index.append([user, i+conf.num_all_user_id])
                #infor_value.append(1.0/(np.sqrt(len(item_list)*len(user_list))))
                social_value.append(1.0/len(item_list))


        for item in item_dict.keys():
            user_list = item_dict[item]
            for u in user_list:
                item_list = user_dict[u]
                social_index.append([item + conf.num_all_user_id, u])
                #infor_value.append(1.0/(np.sqrt(len(user_list)*len(item_list))))
                social_value.append(1.0/len(user_list))



    length = conf.num_all_user_id + conf.num_items
    fake_social_agg_mat = torch.sparse.FloatTensor(torch.LongTensor(social_index).t().cuda(), \
        torch.FloatTensor(social_value).cuda(), torch.Size([length, length]))#.to_dense()  
    #import pdb;pdb.set_trace()
    return fake_social_agg_mat


###########construct infor h-g link ##############
def construct_infor_link(soc_dict, user_dict, item_dict):
    infor_index, infor_value = [], []

    for user in user_dict.keys():
        if user in soc_dict.keys():
            items = user_dict[user]
            for i in items:
                infor_index.append([user, i])
                infor_value.append(1.0)
    length = conf.num_all_user_id + conf.num_items
    agg_mat = torch.sparse.FloatTensor(torch.LongTensor(infor_index).t().cuda(), \
        torch.FloatTensor(infor_value).cuda(), torch.Size([length, length]))#.to_dense()  
    #import pdb;pdb.set_trace()
    return agg_mat



# TrainData is used to train the model
class TrainData():
    def __init__(self, infor_rating_train_data, infor_hash_data, social_link_dict, infor_user_dict, link_hash_data):
        self.features_ps = infor_rating_train_data
        self.train_mat = infor_hash_data
        self.social_link = social_link_dict
        self.infor_user = infor_user_dict
        self.social_hash = link_hash_data


    def ng_sample(self):
        features_fill = []
        for x in self.features_ps:
            u, i = x[0], x[1]

            for t in range(conf.num_train_neg):
                j = np.random.randint(conf.num_items)
                while (u, j) in self.train_mat:
                    j = np.random.randint(conf.num_items)
                    
                features_fill.append([u, i, j])

        self.features_fill = features_fill
        self.link_ng_sample()
        self.infor_bridge_sample()

    #'''
    def link_ng_sample(self):
        link_features_fill = []
        
        for user,friends in self.social_link.items():
            if user in self.infor_user:
                for f in friends:
                    j = np.random.randint(conf.num_bri_user_start, conf.num_all_user_id)
                    while (user,j) in self.social_hash:
                        j = np.random.randint(conf.num_bri_user_start, conf.num_all_user_id)
                    link_features_fill.append([user, f, j])
        self.link_features_fill = link_features_fill        
    #'''

    def infor_bridge_sample(self):
        infor_bridge_fill = []

        for user,items in self.infor_user.items():
            if user in self.social_link:
                for i in items:
                    j = np.random.randint(conf.num_items)
                    while (user, j) in self.train_mat:
                        j = np.random.randint(conf.num_items)
                    infor_bridge_fill.append([user, i, j])
        self.infor_bridge_fill = infor_bridge_fill


    def __len__(self):
        return len(self.features_ps) * (conf.num_train_neg)

    def __getitem__(self, idx):
        features = self.features_fill

        user = features[idx][0]
        pos = features[idx][1]
        neg = features[idx][2]

        link_features_fill = self.link_features_fill
        idx = np.random.randint(len(link_features_fill))

        s_bri = link_features_fill[idx][0]
        s_bri_pos = link_features_fill[idx][1]
        s_bri_neg = link_features_fill[idx][2]

        infor_bridge_fill = self.infor_bridge_fill
        idx_2 = np.random.randint(len(infor_bridge_fill))

        i_bri = infor_bridge_fill[idx_2][0]
        i_bri_pos = infor_bridge_fill[idx_2][1]
        i_bri_neg = infor_bridge_fill[idx_2][2]

        return user, pos, neg, s_bri, s_bri_pos, s_bri_neg, i_bri, i_bri_pos, i_bri_neg
        
class EvaData():
    def __init__(self, eva_data):
        self.eva_data = eva_data
        self.length = len(eva_data.keys())

    def get_batch(self, batch_idx_list):
        user_list, item_list = [], []
        for idx in batch_idx_list:
            user_list.extend([self.eva_data[idx][0]]*(len(self.eva_data[idx])-1))
            item_list.extend(self.eva_data[idx][1:])

        return torch.LongTensor(user_list).cuda(), \
            torch.LongTensor(item_list).cuda()
