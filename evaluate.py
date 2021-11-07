import numpy as np
import torch
import config_data as conf
from collections import defaultdict
import math
import heapq
 

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def hr_ndcg(indices_sort_top,index_end_i,top_k): 
    hr_topK=0
    ndcg_topK=0

    ndcg_max=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        temp_max_ndcg+=1.0/math.log(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg

    max_hr=top_k
    max_ndcg=ndcg_max[top_k-1]
    if index_end_i<top_k:
        max_hr=(index_end_i)*1.0
        max_ndcg=ndcg_max[index_end_i-1] 
    count=0
    for item_id in indices_sort_top:
        if item_id < index_end_i:
            hr_topK+=1.0
            ndcg_topK+=1.0/math.log(count+2) 
        count+=1
        if count==top_k:
            break

    hr_t=hr_topK/max_hr
    ndcg_t=ndcg_topK/max_ndcg  
    # hr_t,ndcg_t,index_end_i,indices_sort_top
    # pdb.set_trace() 
    return hr_t,ndcg_t



def ful_evaluate_rating(infor_train_data_path, test_ground_truth, func, infor_user_mat, top_k):
    HR, NDCG = [], []
    all_pre = func(infor_user_mat)
    
    for u, _ in test_ground_truth.items():
        item_i_list = list(test_ground_truth[u])
        item_j_list = list(set(range(conf.num_items)) - test_ground_truth[u] - infor_train_data_path[u])
        item_i_list.extend(item_j_list)
        index_end_i = len(test_ground_truth[u])
        #import pdb;pdb.set_trace()
        pre_one = all_pre[u][item_i_list]
        indices = largest_indices(pre_one, top_k)
        indices = list(indices[0])

        hr_t, ndcg_t = hr_ndcg(indices, index_end_i, top_k)
        HR.append(hr_t)
        NDCG.append(ndcg_t)
    hr_test=round(np.mean(HR),4)
    ndcg_test=round(np.mean(NDCG),4)
    return hr_test, ndcg_test



# def ful_rating(infor_train_data_path, test_ground_truth, func, infor_user_mat):
    
#     all_pre = func(infor_user_mat)
    
#     for top_k in range(10,55,10):
#         HR, NDCG = [], []
#         for u, _ in test_ground_truth.items():
#             item_i_list = list(test_ground_truth[u])
#             item_j_list = list(set(range(conf.num_items)) - test_ground_truth[u] - infor_train_data_path[u])
#             item_i_list.extend(item_j_list)
#             index_end_i = len(test_ground_truth[u])
#             #import pdb;pdb.set_trace()
#             pre_one = all_pre[u][item_i_list]
#             indices = largest_indices(pre_one, top_k)
#             indices = list(indices[0])

#             hr_t, ndcg_t = hr_ndcg(indices, index_end_i, top_k)
#             HR.append(hr_t)
#             NDCG.append(ndcg_t)
#         hr_test=round(np.mean(HR),4)
#         ndcg_test=round(np.mean(NDCG),4)
#         print('topk: {}, HR: {:.6f}, NDCG: {:.6f}'.format(top_k, hr_test, ndcg_test))
     
def ful_rating(infor_train_data_path, test_ground_truth, func, infor_user_mat):
    
    all_pre = func(infor_user_mat)
    
    for top_k in range(10,15,10):
        HR, NDCG = [], []
        for u, _ in test_ground_truth.items():
            item_i_list = list(test_ground_truth[u])
            item_j_list = list(set(range(conf.num_items)) - test_ground_truth[u] - infor_train_data_path[u])
            item_i_list.extend(item_j_list)
            index_end_i = len(test_ground_truth[u])
            #import pdb;pdb.set_trace()
            pre_one = all_pre[u][item_i_list]
            indices = largest_indices(pre_one, top_k)
            indices = list(indices[0])

            hr_t, ndcg_t = hr_ndcg(indices, index_end_i, top_k)
            
            HR.append(hr_t)
            NDCG.append(ndcg_t)
        #hr_test=round(np.mean(HR),4)
        #ndcg_test=round(np.mean(NDCG),4)
        import pdb;pdb.set_trace()
        print(sort(HR))

        #print('topk: {}, HR: {:.6f}, NDCG: {:.6f}'.format(top_k, hr_test, ndcg_test))