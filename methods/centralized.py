import torch
from utils import get_logger,Tracker
import json
from performance_metric import calculate_JD, calculate_WD
from utils import DatasetKeys, RunningAverage,running_average_to_dict
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from attrdict import AttrDict
import time
import tqdm
import os
class Centralized_Server():
    def __init__(self,server_dict, args):
        self.logger = get_logger(__name__)
        self.logger.info('Centralized Server Init with {}'.format(server_dict))
        
        
        self.train_data : DataLoader = server_dict['train_data']
        self.test_data : DataLoader = server_dict['test_data']
        self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        self.model_type = server_dict['model_type']
        self.model_config = server_dict['model_config']
        self.model = self.model_type(**self.model_config).to(self.device)
        self.mean_alignment_score = 0.0
        self.round = 0
        self.args = args
        self.dataset = server_dict['dataset']
        self.train_groups = server_dict['train_groups']
        self.test_groups = server_dict['test_groups']
        self.eval_num_qs = args.eval_num_qs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_steps)
        self.ravg = RunningAverage()
        self.save_path = server_dict['save_path']
        self.tracker: Tracker = server_dict['tracker']

        # wandb.define_metric("centralized - tar_ll", step_metric="epoch")
        # wandb.define_metric("centralized - loss", step_metric="epoch")
        
        # self.eval_df = server_dict['eval_df']

    def train(self):
        self.logger.info(f"starting training for centralized server with {self.args.central_epochs} epochs")
        best_alignscore = 0
        # train the local model
        self.model.to(self.device)
        self.model.train()
        self.ravg.reset()

        eval_alignment_score = 0
        eval_train_alignment_score = 0
        
        #how many local epochs 
        for step in tqdm.tqdm(range(0, self.args.central_epochs)):
            self.round = step
            server_start_time = time.time()  # Record the start time
            self.optimizer.zero_grad()
            for batch in self.train_data:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                outs = self.model(batch)
                outs.loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                for key, val in outs.items():
                    self.ravg.update(key, val)
                tolog = running_average_to_dict(self.ravg)
                tolog.update({'train_step':self.round})
                wandb.log(tolog)
                if step % self.args.eval_freq == 0:
                    ckpt = AttrDict()
                    ckpt.model = self.model.state_dict()
                    ckpt.optimizer = self.optimizer.state_dict()
                    ckpt.scheduler = self.scheduler.state_dict()
                    ckpt.step = step + 1
                    torch.save(ckpt, '{}/{}'.format(self.save_path, f'centralized_model_{step}.tar'))
                    eval_alignment_score = self.test()
                    if eval_alignment_score > best_alignscore:
                        best_alignscore = eval_alignment_score
                        # torch.save(ckpt, os.path.join("centralized/", f'ckpt_{step}.tar'))
                        torch.save(ckpt, '{}/{}'.format(self.save_path, f'centralized_best_model_{step}.tar'))
                    eval_train_alignment_score = self.test(is_train=True)
                    self.tracker.wandb_alignment.update({"eval_step": step})
                    wandb.log(self.tracker.wandb_alignment)
                    self.tracker.wandb_alignment = {}
                self.ravg.reset()
                    
            server_end_time = time.time()  # Record the end time
            server_elapsed_time = server_end_time - server_start_time  # Calculate elapsed time
            self.tracker.add_centralized_server_tracker_time(step, server_elapsed_time)
            self.tracker.add_centralized_model_accuracy_test(step,eval_alignment_score)
            self.tracker.add_centralized_model_accuracy_train(step,eval_train_alignment_score)
            

        weights = self.model.cpu().state_dict()
        #self.logger.info(f"[Round {self.round}] Client {self.client_id}:{self.group_name} finished training. Returning updated weights.")

        return weights
    
    def test(self,is_train=False):
        self.model.eval()
        #self.logger.info(f"[Round {self.round}] Evaluating model on dataset `{self.dataset}`")
        self.model.to(self.device)
        if self.dataset == DatasetKeys.OQA.value:
            score = calculate_WD(self.eval_num_qs, self.model, self.eval_df, mode='eval')
            self.logger.info(f"[Round {self.round}] Calculated Average Wasserstein Distance Score: {score:.4f}")
            self.model.train()
            return score
        elif self.dataset == DatasetKeys.GLOBALQA.value:
            score = calculate_JD(self.eval_num_qs, self.model, self.test_data.dataset if not is_train else self.train_data.dataset, mode='eval' if not is_train else 'train',type='centralized',step_size=self.round+1,tracker=self.tracker)
            self.logger.info(f"[Round {self.round}] Calculated Average Jensen Distance Score: {score:.4f}")
            self.model.train()
            return score
        else:
            self.logger.error(f"[Round {self.round}] Dataset `{self.dataset}` is not supported.")
            raise ValueError(f"Dataset `{self.dataset}` not supported")