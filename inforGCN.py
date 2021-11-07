import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter
import numpy as np
from time import time
import config_data as conf
import random


class inforGCN(nn.Module):
    def __init__(self):
        super(inforGCN, self).__init__()
        self.infor_user_embedding = \
            nn.Parameter(torch.randn(conf.num_all_user_id, conf.mf_dim)) # num user * 32
        self.infor_item_embedding = \
            nn.Parameter(torch.randn(conf.num_items, conf.mf_dim)) # num user *32
        self.infor_fake_user_embedding = \
            nn.Parameter(torch.randn(conf.num_all_user_id, conf.mf_dim)) # num user * 32       
        self.infor_fake_item_embedding = \
            nn.Parameter(torch.randn(conf.num_items, conf.mf_dim)) # num user *32   

        self.social_global_user_embedding = \
            nn.Parameter(torch.randn(conf.num_all_user_id, conf.mf_dim)) # num user * 32
        self.social_global_item_embedding = \
            nn.Parameter(torch.randn(conf.num_items, conf.mf_dim)) # num user *32  


        self.infor_global_user_embedding = \
            nn.Parameter(torch.randn(conf.num_all_user_id, conf.mf_dim)) # num user * 32
        self.infor_global_item_embedding = \
            nn.Parameter(torch.randn(conf.num_items, conf.mf_dim)) # num user *32   
        self.BCELoss = torch.nn.BCELoss()  

        self.bilinear_w = nn.Parameter(torch.randn(64,64))
        self.bilinear_b = nn.Parameter(torch.zeros([1]))
        self.reinit()


    # random seed, randomly initialize
    def reinit(self):
        torch.manual_seed(0); \
            nn.init.normal_(self.infor_user_embedding, std=0.01)
        torch.manual_seed(0); \
            nn.init.normal_(self.infor_item_embedding, std=0.01)
        torch.manual_seed(0); \
            nn.init.normal_(self.infor_fake_user_embedding, std=0.01)
        torch.manual_seed(0); \
            nn.init.normal_(self.infor_fake_item_embedding, std=0.01)
        torch.manual_seed(0); \
            nn.init.normal_(self.social_global_user_embedding, std=0.01)
        torch.manual_seed(0); \
            nn.init.normal_(self.social_global_item_embedding, std=0.01)
        torch.manual_seed(0); \
            nn.init.normal_(self.infor_global_user_embedding, std=0.01)
        torch.manual_seed(0); \
            nn.init.normal_(self.infor_global_item_embedding, std=0.01)
        
        torch.manual_seed(0); \
            nn.init.normal_(self.bilinear_w, std=0.01)

    # 

    def gcn_layer(self, A_matrix, user_embed, item_embed):
        node_embed = torch.cat((user_embed, item_embed), dim = 0)
        final_emb = [node_embed]
        for k in range(3):
            tmp_user_emb = torch.sparse.mm(A_matrix, final_emb[-1]) + final_emb[-1]
            final_emb.append(tmp_user_emb)
        #final_user_emb = torch.stack(final_user_emb, dim=1)
        #final_item_emb = torch.stack(final_item_emb, dim=1)
        final_emb = torch.stack(final_emb, dim=1)
        #final_user_emb = torch.mean(final_user_emb, dim=1)
        #final_item_emb = torch.mean(final_item_emb, dim=1)
        final_emb = torch.mean(final_emb, dim = 1)
        
        user_embedding, item_embedding = torch.split(final_emb, [conf.num_all_user_id, conf.num_items], dim = 0)
        #user_embedding, item_embedding = final_emb[:conf.num_all_user_id], final_emb[conf.num_all_user_id:]
        #import pdb;pdb.set_trace()
        return user_embedding, item_embedding





    def forward(self, infor_user_mat, corrupted_local_mat, global_social_user_mat, global_infor_user_mat, user, pos, neg, s_bri, s_bri_pos, s_bri_neg, i_bri, i_bri_pos, i_bri_neg):  
        #'''

        user_embedding, item_embedding = self.gcn_layer(infor_user_mat, self.infor_user_embedding, self.infor_item_embedding)

        corrupted_user_embedding, corrupted_item_embedding = self.gcn_layer(corrupted_local_mat, self.infor_fake_user_embedding, self.infor_fake_item_embedding)


        global_social_user_embedding, _ = self.gcn_layer(global_social_user_mat, self.social_global_user_embedding, self.social_global_item_embedding)

        global_infor_user_embedding, global_infor_item_embedding = self.gcn_layer(global_infor_user_mat, self.infor_global_user_embedding, self.infor_global_item_embedding)


        #fake_infor_user_embedding, fake_infor_item_embedding = self.gcn_layer(infor_fake_user_mat, self.infor_fake_user_embedding, self.infor_fake_item_embedding)
        #fake_social_user_embedding, _ = self.gcn_layer(social_fake_mat, self.social_fake_user_embedding, self.social_fake_item_embedding)

        #########################bpr loss#########################
        infor_user_embed = user_embedding[user] # (batch, embed_dim)
        infor_item_pos_embed = item_embedding[pos] # (batch, embed_dim)
        infor_item_neg_embed = item_embedding[neg]

        pos_pred = torch.sum(infor_user_embed * infor_item_pos_embed, dim=1)
        neg_pred = torch.sum(infor_user_embed * infor_item_neg_embed, dim=1)
        l2_reg_infor = 0.01*(infor_user_embed**2 + infor_item_pos_embed**2 + infor_item_neg_embed**2).sum(dim=-1)
        #import pdb; pdb.set_trace()
        rating_loss = - ((pos_pred - neg_pred).sigmoid().log().sum()) + l2_reg_infor.sum()


        ####################social discriminator#################
        #global social user representation
        global_social_user = torch.mean(global_social_user_embedding[conf.num_bri_user_start:],dim=0,keepdim=True)

        #fake_global_social_user = torch.mean(fake_social_user_embedding[conf.num_bri_user_start:],dim=0,keepdim=True)

        #positiva sample [local, global]
        soc_real_predict = torch.sigmoid(torch.cat([user_embedding[s_bri], global_social_user.repeat(len(s_bri),1)], 1))

        #negative sample
        soc_fake_predict = torch.sigmoid(torch.cat([corrupted_user_embedding[s_bri], global_social_user.repeat(len(s_bri),1)], 1))

        #discriminator
        #import pdb;pdb.set_trace()
        social_loss_1 = self.BCELoss(soc_real_predict, torch.ones(soc_real_predict.size()).cuda()) 
        social_loss_2 = self.BCELoss(soc_fake_predict, torch.zeros(soc_fake_predict.size()).cuda())
        social_loss = social_loss_1 + social_loss_2
        #import pdb;pdb.set_trace()


        ####################infor discriminator###################
        
        #local infor user$item representation
        #link representation [ua,vi]
        infor_pos = torch.cat([torch.sigmoid(user_embedding[i_bri]), torch.sigmoid(item_embedding[i_bri_pos])], 1)
        infor_neg = torch.cat([torch.sigmoid(corrupted_user_embedding[i_bri]), torch.sigmoid(corrupted_item_embedding[i_bri_neg])], 1)
        #infor_pos = torch.sigmoid(user_embedding[i_bri])
        #infor_neg = torch.sigmoid(fake_infor_user_embedding[i_bri])


        #global infor user&item representation
        infor_global_user = torch.mean(torch.sigmoid(global_infor_user_embedding[i_bri]), dim=0, keepdim=True)
        infor_global_item = torch.mean(torch.sigmoid(global_infor_item_embedding[i_bri_pos]), dim=0, keepdim=True)
        infor_global_emb = torch.cat([infor_global_user, infor_global_item], 1).repeat(len(i_bri), 1)

        #
        '''
        fake_infor_global_user = torch.mean(torch.sigmoid(fake_infor_user_embedding[i_bri]), dim=0, keepdim=True)
        fake_infor_global_item = torch.mean(torch.sigmoid(fake_infor_item_embedding[i_bri_neg]), dim=0, keepdim=True)
        fake_infor_global_emb = torch.cat([fake_infor_global_user, infor_global_item], 1).repeat(len(i_bri), 1)
        #'''
        
        #positive sample
        #infor_real_predict = torch.cat([infor_pos, infor_global_emb], 1)
        infor_real_predict = torch.cat([torch.sigmoid(user_embedding[i_bri]), infor_global_emb], 1)

        #negative sample
        #infor_fake_predict = torch.cat([infor_neg, infor_global_emb], 1)
        infor_fake_predict = torch.cat([torch.sigmoid(corrupted_user_embedding[i_bri]), infor_global_emb], 1)
        
        #discriminator
        infor_loss_1 = self.BCELoss(infor_real_predict, torch.ones(infor_real_predict.size()).cuda())
        infor_loss_2 = self.BCELoss(infor_fake_predict, torch.zeros(infor_fake_predict.size()).cuda())
        infor_loss = infor_loss_1 + infor_loss_2

        #########################LOSS##########################
        obj = rating_loss + 100*social_loss + 1000*infor_loss
        return obj, rating_loss , social_loss, infor_loss




######################test##############################
    def predict_infor_rating(self,infor_user_mat):       
        #user_embedding, item_embedding = self.gcn_layer(infor_user_mat, self.infor_user_embedding, self.infor_item_embedding)
        
        #import pdb;pdb.set_trace()
        pred = torch.matmul(all_embed, torch.transpose(all_embed, 0, 1))
        return pred
    

    def predict_link(self,soc_mat):  
        social_user_embedding = self.soc_gcn_layer(soc_mat, self.soc_user_embedding)
        pred = torch.matmul(social_user_embedding, torch.transpose(social_user_embedding, 0, 1))        
        return pred

    def predict_rating(self, infor_user_mat):
        #all_user = conf.num_users + conf.num_soc_user - conf.num_common_user
        #social_user_embedding = self.soc_gcn_layer(soc_mat, self.soc_user_embedding, com_agg_mat)
        #infor_user, infor_item = self.infor_gcn_layer(infor_user_mat, infor_item_mat,\
                         #self.infor_user_embedding, self.infor_item_embedding, infor_com_user_mat)

        #user_embedding, item_embedding = self.infor_gcn_layer(infor_user_mat, infor_item_mat, self.infor_user_embedding, self.infor_item_embedding)
        user_embedding, item_embedding = self.gcn_layer(infor_user_mat, self.infor_user_embedding, self.infor_item_embedding)
        user_embedding = user_embedding.cpu().detach().numpy()
        item_embedding = item_embedding.cpu().detach().numpy()
        #import pdb;pdb.set_trace()
        pred = np.matmul(user_embedding, item_embedding.T)
        
        #pred = torch.matmul(user_embedding, torch.transpose(item_embedding, 0, 1))
        return pred

    def predict_soc_rating(self, infor_user_mat):
        #all_user = conf.num_users + conf.num_soc_user - conf.num_common_user
        #social_user_embedding = self.soc_gcn_layer(soc_mat, self.soc_user_embedding, com_agg_mat)
        #infor_user, infor_item = self.infor_gcn_layer(infor_user_mat, infor_item_mat,\
                         #self.infor_user_embedding, self.infor_item_embedding, infor_com_user_mat)

        #user_embedding, item_embedding = self.infor_gcn_layer(infor_user_mat, infor_item_mat, self.infor_user_embedding, self.infor_item_embedding)
        user_embedding, item_embedding = self.gcn_layer(infor_user_mat, self.infor_user_embedding, self.infor_item_embedding)
        #user_embedding = user_embedding.cpu().detach().numpy()
        #item_embedding = item_embedding.cpu().detach().numpy()
        #import pdb;pdb.set_trace()
        #pred = np.matmul(user_embedding, item_embedding.T)
        
        pred = torch.matmul(user_embedding, torch.transpose(item_embedding, 0, 1))
        return pred


