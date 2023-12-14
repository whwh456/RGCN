from /data import dataset
from config import opts
from ../util import F_Info
from ../util import F_Perturbation as F_per
from ../util import F_Normalize as F_nor
from ../util import F_Test
from ../util import find_neighbor_idx as F_find_neighbor_idx
from ../util import construct_sub_graph as F_construct_sub_graph
import warnings


from tqdm.auto import tqdm


import numpy as np
import torch as t
import argparse
import os
import time
import sys

sys.path.append(os.path.abspath('..'))


device=t.device("cuda"if t.cuda.is_available()else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=bool, default=True, help="use CUDA? [True, False]")
parser.add_argument('--seed', type=int, default=30, help="random seed ")
parser.add_argument('--dataset', type=str, default="cora",
                    help="the name of dataset {\"cora\",\"pubmed\",\"citeseer\"}")
parser.add_argument('--attack_node_num', type=int, default=3, help="the chosen attack node number")
parser.add_argument('--fake_node_num', type=int, default=2, help="the fake node num on each attack node")
parser.add_argument('--anci_node_num', type=int, default=20, help="the nodes num for modifying features")
parser.add_argument('--test_time', type=int, default=1, help="multiple test number")
parser.add_argument('--target_class_id', type=int, default=0,
                    help="the target class, cora(0~6) citeseer(0~5) pubmed(0~2)")
parser.add_argument('--feature_budget', type=int, default=25, help="the budget for computing features of fake nodes")

args = parser.parse_args()


def Att_group_node_multi_attack_nodes(opt: opts, adj, feat, label, root):
    fake_base_adj = np.copy(adj)
    fake_base_feat = np.copy(feat)

    fake_attack_node_np = np.copy(opt.attack_node_np)
    fake_ancillary_node_np = np.copy(opt.ancillary_node_np)

    label_attack = np.where(label[fake_attack_node_np[0]])[0]

    for lp_attack_id in range(fake_attack_node_np.shape[0]):
        fake_base_adj, fake_base_feat = F_per.Per_add_fake_node(fake_base_adj, fake_base_feat,
                                                                fake_attack_node_np[lp_attack_id],
                                                                node_num=opt.fake_node_num_each_attack_node)

    fake_node_idx_np = np.arange(adj.shape[0], adj.shape[0] + opt.fake_node_num_each_attack_node * opt.attack_node_num)


    if opt.model == 'GCN':
        opt.model_path = "../checkpoint/{}/GCN.t7".format(opt.dataset)
    else:
        raise NotImplementedError

    model = t.load(opt.model_path)['model']

    record_output = np.zeros([opt.limit_fake_feat, fake_ancillary_node_np.shape[0], label[0].shape[0]])

    record_success_rate = np.zeros([opt.limit_fake_feat])

    record_best_feat = np.zeros([fake_base_feat.shape[0], fake_base_adj.shape[1]])

    fake_feat_add_np = np.copy(fake_base_feat)

    print("STEP: Constructing Fake nodes...\n")

    for lp_limit_feat_id in range(opt.limit_fake_feat):

        lp_anci_id = 0

        feat_grad_sum = np.zeros([opt.fake_node_num_each_attack_node * opt.attack_node_num, feat.shape[1]])


        for lp_anci_idx in fake_ancillary_node_np:

            temp_fake_adj = fake_base_adj
            for lp_attack_idx in fake_attack_node_np:
                temp_fake_adj[lp_attack_idx][lp_anci_idx] = 1
                temp_fake_adj[lp_anci_idx][lp_attack_idx] = 1


            temp_neighbor_set = F_find_neighbor_idx(temp_fake_adj, 2, lp_anci_idx)

   
            proj_o_to_s = {} 
            proj_s_to_o = {}
            for lp_set_id in range(temp_neighbor_set.shape[0]):
                proj_s_to_o[lp_set_id] = temp_neighbor_set[lp_set_id]
                proj_o_to_s[temp_neighbor_set[lp_set_id]] = lp_set_id

          
            lp_anci_idx_proj = proj_o_to_s[lp_anci_idx]


            fake_idx_proj = np.zeros(fake_node_idx_np.shape[0], dtype=np.int16)
            for lp_fake_node_id in np.arange(fake_node_idx_np.shape[0]):
                fake_idx_proj[lp_fake_node_id] = proj_o_to_s[fake_node_idx_np[lp_fake_node_id]]


            sub_adj, sub_d, sub_feat = F_construct_sub_graph(temp_fake_adj, fake_feat_add_np, temp_neighbor_set)


            sub_adj_nor = F_nor.nor_sub_adj_eye(sub_adj, sub_d)
            sub_feat_nor = F_nor.normalize_feat(sub_feat)

  
            sub_adj_nor_T = t.from_numpy(sub_adj_nor).float()
            sub_feat_nor_T = t.from_numpy(sub_feat_nor).float()
            label_attack_T = t.from_numpy(label_attack).int()


            if opt.use_cuda:

                sub_adj_nor_T = sub_adj_nor_T.to(device)

                sub_feat_nor_T = sub_feat_nor_T.to(device)
  
                label_attack_T = label_attack_T.to(device)

            sub_feat_nor_T.requires_grad = True

            model.eval()
            if opt.use_cuda:
             
                model.to(device)


            if opt.model == "GCN":

                output = model(sub_feat_nor_T, sub_adj_nor_T)
            else:
                raise NotImplementedError

  
            label_anci_T = output[[lp_anci_idx_proj]].squeeze().argmax().unsqueeze(dim=0)


            f_minus = -output[lp_anci_idx_proj][label_attack_T.item()] + output[lp_anci_idx_proj][label_anci_T.item()]

            f_minus.backward()

     
            temp_feat_grad = sub_feat_nor_T.grad.cpu().detach().numpy()
            output_anci = output[lp_anci_idx_proj].cpu().detach().numpy()
            

            feat_grad_sum = feat_grad_sum + temp_feat_grad[fake_idx_proj]
            record_output[lp_limit_feat_id, lp_anci_id] = output_anci

            lp_anci_id = lp_anci_id + 1


            for lp_attack_idx in fake_attack_node_np:
                if adj[lp_attack_idx][lp_anci_idx] == 1:
                    continue
                else:
                    temp_fake_adj[lp_attack_idx][lp_anci_idx] = 0
                    temp_fake_adj[lp_anci_idx][lp_attack_idx] = 0
        

        attack_success_rate = F_Test.Test_attack_success_rate_for_Class_Node(label, record_output[lp_limit_feat_id],
                                                                             fake_attack_node_np[0])

        record_success_rate[lp_limit_feat_id] = attack_success_rate


        if lp_limit_feat_id == 0:
            record_best_feat = np.copy(fake_feat_add_np)
            record_best_iter = lp_limit_feat_id
        elif record_success_rate.max() <= attack_success_rate:
            record_best_feat = np.copy(fake_feat_add_np)
            record_best_iter = lp_limit_feat_id


        fake_feat_add_np = F_per.Per_add_fake_feat_based_on_grad_multi_attack_nodes(feat_grad_sum, fake_feat_add_np)




    victim_node_idx = np.arange(adj.shape[0])


    label_attack = label_not_one_hot[fake_attack_node_np[0]]

    victim_node_idx = np.setdiff1d(victim_node_idx, np.where(label_not_one_hot == label_attack))

    victim_node_idx = np.setdiff1d(victim_node_idx, fake_ancillary_node_np)


    if victim_node_idx.shape[0] > 2000:
        victim_node_idx = np.random.choice(victim_node_idx, 2000)

    temp_record = 0
    print("STEP: Testing on victim node set \n")

    for lp_victim_node_id in tqdm(range(victim_node_idx.shape[0]), position=0, leave=True, ncols=80):
        lp_victim_node_idx = victim_node_idx[lp_victim_node_id]


        temp_test_adj = fake_base_adj
        for lp_attack_idx in fake_attack_node_np:
            temp_test_adj[lp_attack_idx, lp_victim_node_idx] = 1
            temp_test_adj[lp_victim_node_idx, lp_attack_idx] = 1


        test_neighbor_set = F_find_neighbor_idx(temp_test_adj, 2, lp_victim_node_idx)


        test_proj_o_to_s = {}
        test_proj_s_to_o = {}
        for lp_set_id in range(test_neighbor_set.shape[0]):
            test_proj_s_to_o[lp_set_id] = test_neighbor_set[lp_set_id]
            test_proj_o_to_s[test_neighbor_set[lp_set_id]] = lp_set_id

        lp_test_idx_proj = test_proj_o_to_s[lp_victim_node_idx]

    
        sub_adj, sub_d, sub_feat = F_construct_sub_graph(temp_test_adj, fake_feat_add_np, test_neighbor_set)

   
        test_sub_adj_nor = F_nor.nor_sub_adj_eye(sub_adj, sub_d)
        test_sub_feat_nor = F_nor.normalize_feat(sub_feat)


        sub_adj_nor_T = t.from_numpy(test_sub_adj_nor).float()
        sub_feat_nor_T = t.from_numpy(test_sub_feat_nor).float()


        if opt.use_cuda:

            sub_adj_nor_T = sub_adj_nor_T.to(device)

            sub_feat_nor_T = sub_feat_nor_T.to(device)

        model.eval()
        if opt.use_cuda:

            model.to(device)

        if opt.model == "GCN":
            output = model(sub_feat_nor_T, sub_adj_nor_T)
        else:
            raise NotImplementedError


        label_anci_T = output[lp_test_idx_proj].argmax().item()

        if label_anci_T == label_attack:
            temp_record = temp_record + 1

 
        for lp_attack_idx in fake_attack_node_np:
            if adj[lp_attack_idx][lp_victim_node_idx] == 1:
                continue
            else:
                temp_test_adj[lp_attack_idx, lp_victim_node_idx] = 0
                temp_test_adj[lp_victim_node_idx, lp_attack_idx] = 0

    temp_success_rate = temp_record / victim_node_idx.shape[0]

    temp_log = "\nThe attack success rate on victim node set：{}\n".format(temp_success_rate)
    print(temp_log)
    with open("./logs/{}Test{}.txt".format(opt.dataset, time.strftime("%Y%m%d")), 'a+') as f:
        f.write(temp_log)

    save_base_root = r'{}/class_id_{}/{}_fake_each_attack/{}_attack/{}_anci/test_{}'.format(root, label_attack,
                                                                                            opt.fake_node_num_each_attack_node,
                                                                                            opt.attack_node_num,
                                                                                            opt.ancillary_node_num,
                                                                                            opt.temp_test_time)

    if not os.path.isdir(save_base_root):
        os.makedirs(save_base_root)

    np.save("{}/best_feat".format(save_base_root), record_best_feat)
    np.save("{}/best_iter".format(save_base_root), record_best_iter)
    np.save("{}/success_rate_for_train".format(save_base_root), record_success_rate)
    np.save("{}/output_train".format(save_base_root), record_output)
    np.save("{}/attack_np".format(save_base_root), fake_attack_node_np)
    np.save("{}/ancillary_np".format(save_base_root), fake_ancillary_node_np)
    np.save("{}/success_rate_for_test".format(save_base_root), temp_success_rate)


if __name__ == '__main__':

    model_path = "../checkpoint"

    opt = opts()
    opt.data_path = r"/data/Data/Planetoid"
    opt.model_path = r"../checkpoint"
    opt.dataset = args.dataset 
    opt.model = 'GCN'

    opt.use_cuda = args.to(device)
    opt.limit_fake_feat = args.feature_budget 
    opt.fake_node_num_each_attack_node = args.fake_node_num  
    opt.attack_node_num = args.attack_node_num  
    # opt.ancillary_node_num = args.anci_node_num  
    opt.total_test_time = args.test_time  
    opt.np_random_seed = args.seed
    opt.target_class = args.target_class_id
    opt.fake_node_num = opt.fake_node_num_each_attack_node * opt.attack_node_num


    data_load = dataset.c_dataset_loader(opt.dataset, opt.data_path)
    base_adj, base_feat, label, idx_train, idx_val, idx_test = data_load.process_data()
    label_not_one_hot = F_Info.F_one_hot_to_label(label)

    data_info = F_Info.C_per_info(base_adj, base_feat, label, idx_train, idx_val, idx_test, opt)


    if opt.target_class < 0 or opt.target_class > label_not_one_hot.max():
        print("invalid class id")
        raise ValueError


    anci = [opt.ancillary_node_num]


    class_num = {}
    class_num['cora'] = 7
    class_num['citeseer'] = 6
    class_num['pubmed'] = 3


    attack_node_np = np.zeros([opt.total_test_time, opt.attack_node_num]).astype(np.int16)
    ancillary_node_np = np.zeros([opt.total_test_time, opt.ancillary_node_num]).astype(np.int16)


    for lp_random_seed in np.arange(opt.np_random_seed, opt.np_random_seed + opt.total_test_time):
        data_info.random_seed = lp_random_seed
     
        attack_node_np[lp_random_seed - opt.np_random_seed] = data_info.F_get_K_random_idx_of_single_class(
            opt.target_class,
            opt.attack_node_num)

        ancillary_node_np[lp_random_seed - opt.np_random_seed] = data_info.F_get_K_random_idx_except_one_class(
            opt.target_class, opt.ancillary_node_num)

    warnings.filterwarnings("ignore")
    opt.temp_test_time = 0
    for lp_test_time in range(opt.total_test_time):
        opt.attack_node_np = attack_node_np[lp_test_time]
        opt.ancillary_node_np = ancillary_node_np[lp_test_time]

        temp_log = "Runing on {}，Class {}， Test {}.... {} attack nodes...{} fake nodes each...{} anci nodes... \n".format(
            opt.dataset, opt.target_class,
            lp_test_time,
            opt.attack_node_num,
            opt.fake_node_num_each_attack_node, opt.ancillary_node_num)
        print(temp_log)

        if not os.path.isdir("./logs"):
            os.makedirs("./logs")
        with open("./logs/{}Test{}.txt".format(opt.dataset, time.strftime("%Y%m%d")), 'a+') as f:
            f.write(temp_log)

        Att_group_node_multi_attack_nodes(opt, adj=base_adj, feat=base_feat, label=label,
                                          root="./fold_result/{}/{}".format(opt.model, opt.dataset))
        opt.temp_test_time = opt.temp_test_time + 1
