from .base_data_provider import DataProvider
from .custom_data_loader import GlobalGroupDataset_gpo, collate_fn_gpo_global_padding
from .constants import COUNTRIES, TRAIN_COUNTRIES, EVAL_COUNTRIES
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
from utils import get_logger

"""
Module: global_qa_provider

This module defines a data provider class for the GlobalQA dataset, used in alignment experiments
with group-based question embeddings. It supports loading group-specific datasets, constructing
DataLoaders for centralized and federated learning, and preparing inputs for training and evaluation.

Classes:
    - GlobalQAProvider: Inherits from `DataProvider` to handle dataset loading and preparation.
    - CollateFunction: Custom collate function to sample and organize context/target questions per batch.
"""
class GlobalQAProvider(DataProvider):
     
    
    def __init__(self, args=None):
        """
    Data provider for the GlobalQA dataset. Supports centralized and per-client DataLoader
    creation for federated learning experiments.

    Attributes:
        embedding_folder (str): Path to folder containing precomputed embeddings.
        dataset_groups (List[str]): All groups in the dataset.
        max/min_ctx/tar_num_qs (int): Bounds on number of context/target questions.
        emb_model (str): Name of the embedding model used (e.g., 'alpaca').
        dataset (str): Dataset name (e.g., 'globalqa').
        groups (List[str]): List of group names.
        group_split (float): Proportion of groups used for training.
        data_split (float): Proportion of each group’s questions used for training.
        train_batch_size (int): Batch size for training dataloaders. #this number don't matter, as each batch represent a group.
        eval_batch_size (int): Batch size for evaluation dataloaders. #this number don't matter, as each batch represent a group.
    """
    
        self.embedding_folder = args.embedding_folder
        self.dataset_groups = COUNTRIES
        self.max_ctx_num_qs = args.max_ctx_num_qs
        self.min_ctx_num_qs = args.min_ctx_num_qs
        self.max_tar_num_qs = args.max_tar_num_qs
        self.min_tar_num_qs = args.min_tar_num_qs
        self.emb_model = args.emb_model
        self.dataset = args.dataset
        self.groups = COUNTRIES
        self.group_split = args.group_split
        self.data_split = args.data_split
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        
        self.train_dataloader : DataLoader = None
        self.eval_dataloader : DataLoader = None
        
        self.cleint_train_dataloader : dict[int,DataLoader] = {}
        self.cleint_test_dataloader : dict[int,DataLoader] = {}
        
        self.num_samples_train_loader = 0
        self.num_samples_test_loader = 0
        
        self.logger = get_logger(__name__)
    
    def load_data_set(self):
        """
        Loads dataset from precomputed embeddings and prepares DataLoaders for both centralized
        and per-client training and evaluation.
        
        @Todo: This function can be enehanced to run faster, by preparing all the data in one go.

        Returns:
            Tuple containing:
                - total_train_questions (int): Total number of training questions across all groups.
                - total_test_questions (int): Total number of test questions across all groups.
                - train_dataloader (DataLoader): DataLoader for training data.
                - eval_dataloader (DataLoader) : DataLoader for evaluation data.
                - cleint_train_dataloader (dict[int, DataLoader]): DataLoader for each client group.
                - cleint_test_dataloader (dict[int, DataLoader]) : DataLoader for each client group.
                - train_groups (List[str]): List of training group names.
                - eval_groups (List[str]): List of evaluation group names.
        """
        self.load_embedding()
        #self.train_groups = np.random.choice(self.groups, size=int(len(self.groups)*self.group_split), replace=False)
        #self.eval_groups = [group for group in self.groups if group not in self.train_groups]
        self.train_groups = TRAIN_COUNTRIES
        self.eval_groups = EVAL_COUNTRIES
        train_dataset = GlobalGroupDataset_gpo(self.embedding, self.train_groups,1,mode='train')
        eval_dataset = GlobalGroupDataset_gpo(self.embedding, self.eval_groups, 1,mode='eval')
        
        collate_function = CollateFunction(self.max_ctx_num_qs, self.min_ctx_num_qs, self.max_tar_num_qs, self.min_tar_num_qs,self.data_split)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, collate_fn=collate_function, num_workers=0)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, collate_fn=collate_function, num_workers=0)
        
        client_train_collate_function = CollateFunction(self.max_ctx_num_qs, self.min_ctx_num_qs, self.max_tar_num_qs, self.min_tar_num_qs,self.data_split)
        client_test_collate_function = CollateFunction(self.max_ctx_num_qs, self.min_ctx_num_qs, self.max_tar_num_qs, self.min_tar_num_qs,1-self.data_split)
        for group_id, group_name in enumerate(self.train_groups):
            client_train_dataset = GlobalGroupDataset_gpo(self.embedding, [group_name],mode='train',split=self.data_split)
            client_test_dataset = GlobalGroupDataset_gpo(self.embedding, [group_name], mode='test',split=1-self.data_split)
            self.cleint_train_dataloader[group_id] =  DataLoader(client_train_dataset, batch_size=self.train_batch_size, collate_fn=client_train_collate_function, num_workers=0)
            self.cleint_test_dataloader[group_id] = DataLoader(client_test_dataset, batch_size=self.eval_batch_size, collate_fn=client_test_collate_function, num_workers=0)
            
        self.logger.info(f"Train dataloader size: {len(self.train_dataloader)}")
        self.logger.info(f"Eval dataloader size: {len(self.eval_dataloader)}")
        
        self.calculate_total_questions()
        
        return self.total_train_questions, self.total_test_questions, self.train_dataloader, self.eval_dataloader, self.cleint_train_dataloader, self.cleint_test_dataloader,\
            self.train_groups,self.eval_groups
        

    def calculate_total_questions(self):
        """
        Calculates total number of questions used in train and test datasets.
        """
        self.total_train_questions = sum(len(group_data['qkeys']) for group_data in self.train_dataloader.dataset.data)
        self.total_test_questions = sum(len(group_data['qkeys']) for group_data in self.eval_dataloader.dataset.data)
        
        print(f"Total questions in train dataset: {self.total_train_questions}")
        print(f"Total questions in test dataset: {self.total_test_questions}")

    
    def load_embedding(self):
        """
        Loads the precomputed question embeddings from the embedding pickle file.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        embedding_folder = os.path.join(current_dir, "../llm_embedding")
        self.embedding = pd.read_pickle(embedding_folder + f'/embeddings_{self.emb_model}_{self.dataset}.pkl')
        # self.embedding = pd.read_pickle(self.embedding_folder + f'/embeddings_{self.emb_model}_{self.dataset}.pkl')
    
    
class CollateFunction:
    """
    Collate function to control the number of context and target questions sampled from each batch.

    Args:
        max_ctx_num_points (int): Max number of context questions.
        min_ctx_num_points (int): Min number of context questions.
        max_tar_num_points (int): Max number of target questions.
        min_tar_num_points (int): Min number of target questions.
        split (float): Proportion of questions to sample.

    Used in:
        torch.utils.data.DataLoader
    """
    def __init__(self, max_ctx_num_points, min_ctx_num_points, max_tar_num_points, min_tar_num_points, split):
        self.max_ctx_num_points = int(max_ctx_num_points * split)
        self.min_ctx_num_points = int(min_ctx_num_points * split)
        self.max_tar_num_points = int(max_tar_num_points * split)
        self.min_tar_num_points = int(min_tar_num_points * split)
    def __call__(self, batch):
        return collate_fn_gpo_global_padding(batch, self.max_ctx_num_points, self.min_ctx_num_points, self.max_tar_num_points, self.min_tar_num_points)