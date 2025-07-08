import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler

class def_complete_model(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int,input_dim3: int, input_dim4: int,input_dim5: int,input_dim6: int,int,input_dim7: int,
                 hidden_dim: int,output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc4 = nn.Linear(input_dim4, hidden_dim)
        self.fc5 = nn.Linear(input_dim5, hidden_dim)
        # self.fc6 = nn.Linear(input_dim6, hidden_dim)
        # self.fc7 = nn.Linear(input_dim7, hidden_dim)
        self.fc1 = nn.Linear(input_dim1 + hidden_dim + hidden_dim, output_dim)
        # self.fc2 = nn.Linear(input_dim2 + hidden_dim + hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(input_dim3 + hidden_dim + hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor,input_3: torch.Tensor):#, input_4: torch.Tensor,input_5: torch.Tensor,
                #input_6: torch.Tensor, input_7: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """

        h2=self.act(self.fc4(input_2))
        h3 = self.act(self.fc5(input_3))
        x = torch.cat([input_1, h2, h3], dim=1)
        h = self.fc1(x)
        return h

        # Tensor, shape (*, input_dim1 + input_dim2)
        # h4 = self.act(self.fc4(input_4))
        # h5 = self.act(self.fc5(input_5))
        # h6 = self.act(self.fc6(input_6))
        # h7 = self.act(self.fc7(input_7))
        #
        # x2 = torch.cat([input_2, h4, h5], dim=1)
        # x3 = torch.cat([input_3, h6, h7], dim=1)
        # h2 = self.act(self.fc2(x2))
        # h3 = self.act(self.fc3(x3))
        #
        # x1 = torch.cat([input_1, h2, h3], dim=1)
        # # Tensor, shape (*, output_dim)
        # h = self.fc1(x1)
        # return h

        # for i in range(bayes_num):
        #     if i == 1:
        #         input_3 = 1
        #         self.h1 = self.singe_prdict(input_1, input_2, input_3)
        #     elif 1 << i & i << 3:
        #         self.h2 = self.singe_prdict(input_1, input_2, self.h1)
        #     elif i == 3:
        #         self.h = self.singe_prdict(input_1, input_2, self.h2)
        # return self.h

    # def singe_prdict(self, input_1: torch.Tensor, input_2: torch.Tensor, input_3: torch.Tensor):
    #     x = torch.cat([input_1, input_2], dim=1)
    #     # Tensor, shape (*, output_dim)
    #     h = self.fc2(self.act(self.fc1(x)))
    #     return h
    #
    # def built_DAG(self, input_1: torch.Tensor, input_2: torch.Tensor, bayes_num):
    #     x = torch.cat([input_1, input_2], dim=1)
    #     # Tensor, shape (*, output_dim)
    #     h = self.fc2(self.act(self.fc1(x)))
    #     return h

    ############################################################################################################################################################################
    #########################################################    最重要的运行部分   #############################################################################################
    ############################################################################################################################################################################
    # 抛开它，自己写，这样还快
def compute_src_dst_node_neighbor(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
    """
    compute source and destination node temporal embeddings
    :param src_node_ids: ndarray, shape (batch_size, )
    :param dst_node_ids: ndarray, shape (batch_size, )
    :param node_interact_times: ndarray, shape (batch_size, )
    :return:
    """
    # get the first-hop neighbors of source and destination nodes
    #########################          获取源节点和目的节点的第一跳邻居         #####################################
    # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
    # 三个列表用于存储源节点的第一跳邻居id、边缘id 和交互时间戳信息，列表长度为batch_size
    src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
        self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

    # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
    #三个列表用于存储目标节点的第一跳邻居id、边缘id 和交互时间戳信息，其中batch_size为列表长度
    dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
        self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)



    ###################    共现特征的提取，重点就是把两者共同出现的频率计算一下，但这点加入的特征，感觉完全没有啥信息量        #####################################
    # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
    # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
    src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
        self.count_nodes_appearances(src_padded_nodes_neighbor_ids=src_nodes_neighbor_ids_list,
                                            dst_padded_nodes_neighbor_ids=dst_nodes_neighbor_ids_list)


    return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features

def neighbor_max(src_mapping_dict):
    for key, value in src_mapping_dict.items():
        if (value == max(src_mapping_dict.values())):
            return key, value

def count_singe_node(src_padded_node_neighbor_ids,dst_padded_node_neighbor_ids):
    src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids, return_inverse=True,
                                                                 return_counts=True)
    src_mapping_dict = dict(zip(src_unique_keys, src_counts))
    dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids, return_inverse=True,
                                                                 return_counts=True)
    dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

    for key, value in dst_mapping_dict.items():
        if key in src_mapping_dict:
            src_mapping_dict["key"] += value
        else:
            del src_mapping_dict["key"]
    max_key, max_value = neighbor_max(src_mapping_dict)
    return torch.stack([max_key, max_value], dim=1)

def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
    src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
    for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):
        src_padded_nodes_appearances.append(count_singe_node(src_padded_node_neighbor_ids,dst_padded_node_neighbor_ids))
    src_dst_nodes_all_max = torch.stack(src_padded_nodes_appearances, dim=0)
    return src_dst_nodes_all_max

class Mybayes(nn.Module): ## 将左右两支路的输出作为特征输入的注意力，或者三种拼接方式

    def __init__(self, input_dim1: int, input_dim2: int,hidden_dim: int, output_dim: int,bayes_num:int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor, bayes_num):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        for i in range(bayes_num):
            if i==1:
                input_3=1
                self.h1=self.singe_prdict(input_1, input_2, input_3)
            elif 1<<i&i<<3:
                self.h2=self.singe_prdict(input_1, input_2, self.h1)
            elif i==3:
                self.h = self.singe_prdict(input_1, input_2, self.h2)
        return self.h

    def singe_prdict(self, input_1: torch.Tensor, input_2: torch.Tensor,input_3: torch.Tensor):
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h

    def built_DAG(self, input_1: torch.Tensor, input_2: torch.Tensor, bayes_num):
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h

