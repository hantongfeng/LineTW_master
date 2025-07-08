from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import torch
import os
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler,Data
from utils.load_configs import get_link_prediction_args
import pickle
class SetMax:
    def __init__(self,data: Data):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,用时间间隔控制采样偏好
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times 每个节点的邻居id、边id和交互时间列表，按交互时间排序
        max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
        adj_list = [[] for _ in range(max_node_id + 1)]
        for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
            adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
            adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological

            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

    def count_all_max(self, full_data, src_node_ids, dst_node_ids, node_interact_times,k=12, 
                  temporal_bias=0.1, spatial_bias=0.5, ee_bias=0.3):
        args = get_link_prediction_args(is_evaluation=False)

        full_neighbor_sampler = get_neighbor_sampler(
            data=full_data, 
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor, 
            seed=1
        )

        src_nodes_neighbors_info = full_neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=src_node_ids, node_interact_times=node_interact_times)
        dst_nodes_neighbors_info = full_neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=dst_node_ids, node_interact_times=node_interact_times)

        src_neighbors_times = src_nodes_neighbors_info[1]  
        dst_neighbors_times = dst_nodes_neighbors_info[1]  
        all_walk_nodes = []  
        for i, (src_ids, dst_ids, src_neighbors, dst_neighbors, src_times, dst_times, interact_time) in \
                enumerate(zip(src_node_ids, dst_node_ids, src_nodes_neighbors_info[0], 
                            dst_nodes_neighbors_info[0], src_neighbors_times, 
                            dst_neighbors_times, node_interact_times)):
            src_neighbors_set = set(src_neighbors)
            dst_neighbors_set = set(dst_neighbors)
            common_neighbors = list(src_neighbors_set.intersection(dst_neighbors_set))
            
            if len(common_neighbors) == 0:
                combined_neighbors = np.concatenate((src_neighbors, dst_neighbors))
                if len(combined_neighbors) != 0:
                    unique_neighbors, counts = np.unique(combined_neighbors, return_counts=True)
                    max_id = unique_neighbors[np.argmax(counts)]
                else:
                    max_id=-1
                
                all_walk_nodes.append([max_id] * args.walk_length)
                continue
            
            sampled_path = self.spatiotemporal_biased_walk(
                common_neighbors=common_neighbors,
                src_neighbors=src_neighbors,
                dst_neighbors=dst_neighbors,
                src_times=src_times, 
                dst_times=dst_times,
                interact_time=interact_time,
                full_data=full_data,
                full_neighbor_sampler=full_neighbor_sampler,
                k=args.walk_length,
                temporal_bias=temporal_bias,
                spatial_bias=spatial_bias,
                ee_bias=ee_bias
            )
            
            all_walk_nodes.append(sampled_path)
        return np.array(all_walk_nodes)

    def spatiotemporal_biased_walk(self, common_neighbors, src_neighbors, dst_neighbors, 
                               src_times, dst_times, interact_time, full_data, 
                               full_neighbor_sampler, k=4, temporal_bias=0.1, 
                               spatial_bias=0.5, ee_bias=0.3):
        
        visit_counts = {node: 0 for node in common_neighbors}
        
        
        node_to_time = {}
        for node, time in zip(src_neighbors, src_times):
            if node in common_neighbors:
                node_to_time[node] = time
        
        for node, time in zip(dst_neighbors, dst_times):
            if node in common_neighbors:
                node_to_time[node] = max(node_to_time.get(node, 0), time)
        
        
        node_degrees = {}
        for node in common_neighbors:
            temp_neighbors_info = full_neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=np.array([node]), 
                node_interact_times=np.array([interact_time])
            )
            node_degrees[node] = max(len(temp_neighbors_info[0][0]), 1)
        
        
        if len(common_neighbors) > 0:
            
            combined_neighbors = np.concatenate((src_neighbors, dst_neighbors))
            
            unique_common, counts = np.unique(
                [n for n in combined_neighbors if n in common_neighbors], 
                return_counts=True
            )
            if len(unique_common) > 0:
                current_node = unique_common[np.argmax(counts)]
            else:
                
                current_node = np.random.choice(list(common_neighbors))
        else:
            
            return []
        
        path = [current_node]
        visit_counts[current_node] += 1
        
        for step in range(k-1):
            
            temp_neighbors_info = full_neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=np.array([current_node]), 
                node_interact_times=np.array([interact_time])
            )
            current_neighbors = temp_neighbors_info[0][0]
            next_candidates = [n for n in current_neighbors if n in common_neighbors]
            
            if len(next_candidates) == 0:
                
                min_visits = min(visit_counts.values())
                restart_candidates = [n for n in common_neighbors if visit_counts[n] == min_visits]
                next_node = np.random.choice(restart_candidates)
            else:
                
                time_diffs = np.abs([node_to_time[n] - interact_time for n in next_candidates])
                
                temporal_scores = np.exp(-temporal_bias * time_diffs)
                
                degrees = np.array([node_degrees[n] for n in next_candidates])
                
                spatial_scores = degrees ** spatial_bias
                
                visited_times = np.array([visit_counts[n] for n in next_candidates])
                
                ee_scores = np.exp(-ee_bias * visited_times) 
                
                
                combined_scores = temporal_scores * spatial_scores * ee_scores
                
               
                if combined_scores.sum() == 0:
                    combined_scores = np.ones(len(next_candidates))
                
                combined_scores = combined_scores / combined_scores.sum()
                
                next_node = np.random.choice(next_candidates, p=combined_scores)
            
            path.append(next_node)
            visit_counts[next_node] += 1
            current_node = next_node
        
        return path

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class SmallData:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)   
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids) 
        self.num_unique_nodes = len(self.unique_node_ids) 

def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'

    # padding the features of edges and nodes to the same dimension ( 172 for all the datasets )
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))# np.quantile 

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask], max_ids_list=None,
                      max_src_label_list=None,max_dst_label_list=None)
    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                                node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, max_ids_list=None,
                      max_src_label_list=None,max_dst_label_list=None)
    setmax=SetMax(full_data)
    max_ids_list=setmax.count_all_max(full_data,src_node_ids, dst_node_ids, node_interact_times)

    
    max_ids_list = np.array(max_ids_list)

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                                node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, max_ids_list=max_ids_list,
                      max_src_label_list=None,max_dst_label_list=None)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask],max_ids_list=max_ids_list[train_mask],
                      max_src_label_list=None,max_dst_label_list=None)

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask], max_ids_list=max_ids_list[val_mask],
                      max_src_label_list=None,max_dst_label_list=None)

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask],max_ids_list=max_ids_list[test_mask],
                      max_src_label_list=None,max_dst_label_list=None)

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask],max_ids_list=max_ids_list[new_node_val_mask],
                      max_src_label_list=None,max_dst_label_list=None)

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask],max_ids_list=max_ids_list[new_node_test_mask],
                      max_src_label_list=None,max_dst_label_list=None)
    
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_or_create_data(folder_path, dataset_name, val_ratio, test_ratio):
    """
    Check if the data exists in the folder. If not, create it and save it to the folder.
    """
    files_exist = all(os.path.exists(os.path.join(folder_path, f"{name}.pkl")) for name in [
        'node_raw_features', 'edge_raw_features', 'full_data', 'train_data', 'val_data', 'test_data', 'new_node_val_data', 'new_node_test_data'])

    if files_exist:
        print(f"Loading data from {folder_path}...")
        node_raw_features = load_data(os.path.join(folder_path, 'node_raw_features.pkl'))
        edge_raw_features = load_data(os.path.join(folder_path, 'edge_raw_features.pkl'))
        full_data = load_data(os.path.join(folder_path, 'full_data.pkl'))
        train_data = load_data(os.path.join(folder_path, 'train_data.pkl'))
        val_data = load_data(os.path.join(folder_path, 'val_data.pkl'))
        test_data = load_data(os.path.join(folder_path, 'test_data.pkl'))
        new_node_val_data = load_data(os.path.join(folder_path, 'new_node_val_data.pkl'))
        new_node_test_data = load_data(os.path.join(folder_path, 'new_node_test_data.pkl'))
    else:
        print(f"Data not found in {folder_path}. Creating data...")
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
            get_link_prediction_data(dataset_name=dataset_name, val_ratio=val_ratio, test_ratio=test_ratio)

        # Ensure the directory exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save data to .pkl files
        save_data(node_raw_features, os.path.join(folder_path, 'node_raw_features.pkl'))
        save_data(edge_raw_features, os.path.join(folder_path, 'edge_raw_features.pkl'))
        save_data(full_data, os.path.join(folder_path, 'full_data.pkl'))
        save_data(train_data, os.path.join(folder_path, 'train_data.pkl'))
        save_data(val_data, os.path.join(folder_path, 'val_data.pkl'))
        save_data(test_data, os.path.join(folder_path, 'test_data.pkl'))
        save_data(new_node_val_data, os.path.join(folder_path, 'new_node_val_data.pkl'))
        save_data(new_node_test_data, os.path.join(folder_path, 'new_node_test_data.pkl'))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    ### 试着加入一个   max_ids,  max_label
    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data
