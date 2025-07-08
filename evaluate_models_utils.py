import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from utils.DataLoader import SetMax

def evaluate_model_link_prediction(model_name: str, model: nn.Module,bayes_model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, full_data:Data, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            evaluate_data_indices = evaluate_data_indices[evaluate_data_indices < len(evaluate_data.src_node_ids)]
            assert len(evaluate_data.src_node_ids) == len(evaluate_data.dst_node_ids) == len(evaluate_data.node_interact_times) == len(evaluate_data.edge_ids), "Data lengths are not consistent!"
            try:
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_max_ids= \
                        evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                        evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices],evaluate_data.max_ids_list[evaluate_data_indices]
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"evaluate_data_indices: {evaluate_data_indices}")
                continue
            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            sset=SetMax(full_data)
            max_neg_ids=sset.count_all_max(full_data,batch_neg_src_node_ids,batch_neg_dst_node_ids,batch_node_interact_times)

            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times)

            # get temporal embedding of negative source and negative destination nodes
            # two Tensors, with shape (batch_size, node_feat_dim)
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                    dst_node_ids=batch_neg_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times)
            pos_src_paths = []
            pos_dst_paths = []
            neg_src_paths = []
            neg_dst_paths = []
           
            if np.all(batch_max_ids != None):
                if len(batch_max_ids.shape) == 1:
                    batch_max_ids = np.expand_dims(batch_max_ids, axis=-1)  # shape (N,) -> (N, 1)
                batch_max_ids_T=batch_max_ids.T
                max_neg_ids_T=max_neg_ids.T
                for step_id in range(1):
                    max_ids_p = batch_max_ids_T[step_id]
                    max_neg_P=max_neg_ids_T[step_id]
                    batch_src_node_embeddings2, batch_dst_node_embeddings2 = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=max_ids_p,
                                                                        node_interact_times=batch_node_interact_times)

                    batch_src_node_embeddings3, batch_dst_node_embeddings3 = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=max_ids_p,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times)
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings2, batch_neg_dst_node_embeddings2 = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=max_neg_P,
                                                                        node_interact_times=batch_node_interact_times)
                    batch_neg_src_node_embeddings3, batch_neg_dst_node_embeddings3 = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=max_neg_P,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times)
                    positive_probabilities2 = model[1](input_1=batch_src_node_embeddings2, input_2=batch_dst_node_embeddings2).squeeze(dim=-1).sigmoid()
                    positive_probabilities3 = model[1](input_1=batch_src_node_embeddings3, input_2=batch_dst_node_embeddings3).squeeze(dim=-1).sigmoid()
                    negative_probabilities2 = model[1](input_1=batch_neg_src_node_embeddings2, input_2=batch_neg_dst_node_embeddings2).squeeze(dim=-1).sigmoid()
                    negative_probabilities3 = model[1](input_1=batch_neg_src_node_embeddings3, input_2=batch_neg_dst_node_embeddings3).squeeze(dim=-1).sigmoid()
                    pos_src_paths.append(positive_probabilities2)
                    pos_dst_paths.append(positive_probabilities3)
                    neg_src_paths.append(negative_probabilities2)
                    neg_dst_paths.append(negative_probabilities3)
                    
                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                pos_src_paths = torch.stack(pos_src_paths)
                pos_dst_paths = torch.stack(pos_dst_paths)
                neg_src_paths = torch.stack(neg_src_paths)
                neg_dst_paths = torch.stack(neg_dst_paths)
                final_positive_probabilities=bayes_model(positive_probabilities.detach().unsqueeze(dim=-1), 
                                                        pos_src_paths.detach(),
                                                        pos_dst_paths.detach()).squeeze(dim=-1).sigmoid()
                final_negative_probabilities = bayes_model(negative_probabilities.detach().unsqueeze(dim=-1),
                                                        neg_src_paths.detach(),
                                                        neg_dst_paths.detach()).squeeze(dim=-1).sigmoid()
            else:
                final_positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                final_negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([final_positive_probabilities, final_negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics