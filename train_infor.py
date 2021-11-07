import DataModule_domain_infor as data_utils
import config_data as conf
import os,sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pdb
from time import time, strftime
from copy import deepcopy
from evaluate import  ful_evaluate_rating
from Logging import Logging
model_name = 'inforGCN'



if __name__ == '__main__':
    log = Logging('%s/train_%s_%s_pretrain_5.log'\
         % (conf.out_path, conf.data_name, model_name))

    log.record('--'*20)
    #########note###############################
    log.record('-----------------')

    exec('from %s import %s' % (model_name, model_name))
    exec('model=%s()' % model_name)


    #################
    #'''
    model_param = model.state_dict()
    pretrain_params = torch.load(conf.save_path)
    #import pdb;pdb.set_trace()

    model_param['infor_user_embedding'] = pretrain_params['infor_user_embedding']
    model_param['infor_item_embedding'] = pretrain_params['infor_item_embedding']
    model_param['infor_fake_user_embedding'] = pretrain_params['infor_fake_user_embedding']
    model_param['infor_fake_item_embedding'] = pretrain_params['infor_fake_item_embedding']
    model_param['social_global_user_embedding'] = pretrain_params['social_global_user_embedding']
    model_param['social_global_item_embedding'] = pretrain_params['social_global_item_embedding']
    model_param['infor_global_user_embedding'] = pretrain_params['infor_global_user_embedding']
    model_param['infor_global_item_embedding'] = pretrain_params['infor_global_item_embedding']
    model_param['bilinear_w'] = pretrain_params['bilinear_w']

    model.load_state_dict(model_param)
    #'''
    #####################
    model.cuda()


    
    optimizer = torch.optim.SGD(\
        model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
  
    print('System start to load data..')
    t0 = time()
    
    infor_hash_data, infor_rating_train_data, eva_infor_rating_test_data,\
     infor_user_dict, infor_item_dict, eva_common_rating_test_data, \
     social_link_dict, eva_social_rating_test_data, link_train_data, \
     link_hash_data, eva_soc_test, eva_soc_test_ground_truth, common_rating_dict = data_utils.load_all()

    #import sys; sys.exit(0)
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1-t0))


    train_data = data_utils.TrainData(infor_rating_train_data, infor_hash_data, social_link_dict, infor_user_dict, link_hash_data)
    train_loader = data.DataLoader(train_data, batch_size = 10240, shuffle=True, num_workers=4)

    #social_agg_mat = data_utils.construct_agg_mat(social_dict)
    infor_user_mat = data_utils.construct_infor_mat(social_link_dict, infor_user_dict, infor_item_dict, True)
   

    global_social_user_mat = data_utils.construct_global_social(social_link_dict)
    global_infor_user_mat = data_utils.construct_global_infor(infor_user_dict, infor_item_dict)


    max_infor_rating_test_hr, max_infor_rating_test_hr_1, count, count_1 = 0,0,0,0
    for epoch in range(1, conf.train_epoch):
        t0 = time()
        train_loader.dataset.ng_sample()
        t1 = time()
        print('train loader cost:{:.4f}s'.format(t1-t0))
        model.train()

        infor_fake_dict, social_fake_dict, infor_fake_item_dict = data_utils.load_corrupt_edge()
        corrupted_local_mat = data_utils.construct_corrupted_graph(infor_fake_dict, infor_fake_item_dict,social_fake_dict)

        train_loss_list, rating_loss_list, soc_loss_list, infor_loss_list = [] ,[] ,[], []

        
        for user, pos, neg, s_bri, s_bri_pos, s_bri_neg, i_bri, i_bri_pos, i_bri_neg in train_loader:
            user = user.cuda()
            pos = pos.cuda()
            neg = neg.cuda()
            s_bri = s_bri.cuda()
            s_bri_pos = s_bri_pos.cuda()
            s_bri_neg = s_bri_neg.cuda()
            i_bri = i_bri.cuda()
            i_bri_pos = i_bri_pos.cuda()
            i_bri_neg = i_bri_neg.cuda()

            loss, rating_loss, soc_loss, infor_loss = \
                    model(infor_user_mat, corrupted_local_mat, global_social_user_mat, global_infor_user_mat,\
                       user, pos, neg, s_bri, s_bri_pos, s_bri_neg, i_bri, i_bri_pos, i_bri_neg)

            train_loss_list.append(loss.item())
            rating_loss_list.append(rating_loss.item())
            soc_loss_list.append(soc_loss.item())
            infor_loss_list.append(infor_loss.item())
          

            model.zero_grad(); loss.backward(); optimizer.step()


        t1 = time()
        log.record('epoch:{}, training cost:{:.4f}s, train loss:{:.4f}, rating loss:{:.4f}, soc_loss:{:.4f}, infor_loss:{:.4f}'.\
                format(epoch, t1-t0, np.mean(train_loss_list), np.mean(rating_loss_list), np.mean(soc_loss_list), np.mean(infor_loss_list)))

        '''
        t4 = time()
        soc_rating_test_hr, soc_rating_test_ndcg = ful_evaluate_bpr_rating(\
                eva_social_rating_test_data, model.predict_rating, infor_user_mat)
        
        t5 = time()
        log.record('epoch:{}, time cost:{:.4f}s, soc rating hr: {:.6f}, ndcg:{:.6f}'.format(epoch, (t5-t4), soc_rating_test_hr, soc_rating_test_ndcg))
        '''
        t6 = time()
        com_rating_test_hr, com_rating_test_ndcg = ful_evaluate_rating(infor_user_dict, common_rating_dict, model.predict_rating, infor_user_mat, 10)
        
        t7 = time()
        log.record('epoch:{}, time cost:{:.4f}s, com rating hr: {:.6f}, ndcg:{:.6f}'.format(epoch, (t7-t6), com_rating_test_hr, com_rating_test_ndcg))
        #'''

        t4 = time()
        test_hr, test_ndcg = ful_evaluate_rating(infor_user_dict, eva_soc_test_ground_truth, model.predict_rating, infor_user_mat, 10)
        t5 = time()
        log.record('epoch:{}, time cost:{:.4f}s, soc rating hr: {:.6f}, ndcg:{:.6f}'.format(epoch, (t5-t4), test_hr, test_ndcg))
        


        #'''
        if test_hr > max_infor_rating_test_hr:
            torch.save(model.state_dict(), conf.save_path)
            print('save non epoch:{}'.format(epoch))
            count = 0
            
        if test_hr < max_infor_rating_test_hr:
            count += 1
            if count > 10 and count_1 > 10:
                import sys;sys.exit(0) 

        max_infor_rating_test_hr = max(max_infor_rating_test_hr, test_hr)

        #'''
        if com_rating_test_hr > max_infor_rating_test_hr_1:
            torch.save(model.state_dict(), conf.save_path_bri)
            print('save bri epoch:{}'.format(epoch))
            count_1 = 0
            
        if com_rating_test_hr < max_infor_rating_test_hr_1:
            count_1 += 1
            if count >  10 and count_1 > 10:
                import sys;sys.exit(0)  

        max_infor_rating_test_hr_1 = max(max_infor_rating_test_hr_1, com_rating_test_hr)














