import numpy as np


class opts():

    lr = 0.0045
    num_hiden_layer = 8
    init_type = 'xavier'  
    drop_out = 0.6
    optim = "adam"  
    weight_decay = 5e-4
    dataset = 'cora' 
    epoch = 800  
    lr_decay_epoch = 5000
    feature_Nor = True 
    model = 'GCN'

    data_path = "./fold_data/Data/Planetoid"
    model_path = "./checkpoint"
    np_random_seed = 100  
    use_cuda = True
    fake_node_num_each_attack_node = 2  
    limit_fake_feat = 25 
    attack_node_num = 3
    flag_att_group_nodes = 1


    adj_num = {"cora":2708,"citeseer":3327, "pubmed":19717}
