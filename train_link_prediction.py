import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.LineTW import LineTW
from models.modules import MergeLayer,def_complete_model
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler,get_adj_list 
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data,SetMax,load_or_create_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda") 
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")  
        print("Using CPU")

    # model.to(device)
    # inputs, labels = inputs.to(device), labels.to(device)


    warnings.filterwarnings('ignore')

    # get arguments 
    args = get_link_prediction_args(is_evaluation=False)

    o=time.time()
    folder_path = './processed_data/my_dataset_folder/{}'.format(args.dataset_name)
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
    load_or_create_data(folder_path, dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    p=time.time()
    print("load_or_create_data 准备时间:",p-o)

    

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0) #9179
    
    

    # initialize validation and test neighbor sampler to retrieve temporal graph 
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1) #9228
    
    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    _,train_data_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(train_data.src_node_ids))
    train_data_neg_src_node_ids = train_data.src_node_ids
    
    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'configuration is {args}')

        dynamic_backbone = LineTW(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                        num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                        max_input_sequence_length=args.max_input_sequence_length, device=args.device).to(device)
    
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                     hidden_dim=node_raw_features.shape[1], output_dim=1).to(device) 
        bayes_model = def_complete_model(input_dim=1,hidden_dim=node_raw_features.shape[1]*2,output_dim=1).to(device)
        model = nn.Sequential(dynamic_backbone, link_predictor).to(device) 



        #logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay) #
        final_optimizer = create_optimizer(model=bayes_model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)
        bayes_model = convert_to_gpu(bayes_model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()

        for epoch in range(args.num_epochs): 
            model.train()
            bayes_model.train()
            model[0].set_neighbor_sampler(train_neighbor_sampler) 

            # store train losses and metrics
            train_losses, train_metrics = [], []
            final_train_losses, final_train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_max_ids,batch_neg_dst_node_ids= \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices],train_data.max_ids_list[train_data_indices],\
                    train_data_neg_dst_node_ids[train_data_indices]

                batch_neg_src_node_ids = batch_src_node_ids  
                
                sset=SetMax(full_data)
                max_neg_ids=sset.count_all_max(full_data,batch_neg_src_node_ids,batch_neg_dst_node_ids,batch_node_interact_times)
                
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times)
            
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
                    model.eval()
                    with torch.no_grad():
                        batch_max_ids_T = batch_max_ids.T
                        max_neg_ids_T=max_neg_ids.T
                        for step_id in range(1):
                            max_ids_p = batch_max_ids_T[step_id]
                            max_neg_P=max_neg_ids_T[step_id]
                            # get temporal embedding of source and destination nodes
                            # two Tensors, with shape (batch_size, node_feat_dim)
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
                    model.train()
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
                time_e=time.time()
                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                loss = loss_func(input=predicts, target=labels)
                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()    
                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, fianl train loss: {loss.item()}')          
                final_predicts = torch.cat([final_positive_probabilities, final_negative_probabilities], dim=0)
                final_loss = loss_func(input=final_predicts, target=labels)
                final_train_losses.append(final_loss.item()) 
                final_train_metrics.append(get_link_prediction_metrics(predicts=final_predicts, labels=labels))
                final_optimizer.zero_grad()
                final_loss.backward()
                final_optimizer.step()
                

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,bayes_model=bayes_model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     full_data=full_data,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)


            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,bayes_model=bayes_model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       full_data=full_data,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)


            logger.info(f'Epoch: {epoch + 1}, learning rate: {final_optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,bayes_model=bayes_model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           full_data=full_data,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)


                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,bayes_model=bayes_model,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             full_data=full_data,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap)


                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            models=[model, bayes_model]
            early_stop = early_stopping.step(val_metric_indicator, models)

            if early_stop:
                break

        early_stopping.load_checkpoint(models)
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,bayes_model=bayes_model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   full_data=full_data,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)


        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,bayes_model=bayes_model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     full_data=full_data,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap)
        
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}


        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')


    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
