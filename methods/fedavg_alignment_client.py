import torch
import time
from utils import DatasetKeys,RunningAverage,running_average_to_dict
from utils import get_logger,Tracker
from utils import DatasetKeys
from performance_metric import calculate_JD, calculate_WD

"""
Module: fedavg_client
This module defines a federated learning client using the FedAvg algorithm. Each client represents a group.
Each client trains locally on their own data and sends model updates to the server.
"""
class FedAvg_Client():
    
    """
    Represents a single federated learning client using the FedAvg algorithm.

    Attributes:
        client_id (int): Unique ID for the client. Used for logging and tracking.
        group_name (str): Name for the group (e.g., a country).
        device (str): CUDA device assigned to the client.
        model_type (nn.Module): Model class to be instantiated.
        model_config (dict): Parameters for initializing the model.
        train_dataloader (DataLoader): DataLoader for the client training data.
        test_dataloader (DataLoader): DataLoader for the client evaluation data.
        args (Namespace): Parsed training arguments.
        round (int): Current communication round.
        ravg (RunningAverage): Tracker for averaging loss/metrics accross local training, logged in wandb.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler): Cosine annealing scheduler for LR decay.
        tracker (Tracker): Tracks logs and performance metrics.
    """
    def __init__(self, client_dict, args):
        #train_data is a data loader. 
        self.logger = get_logger(__name__)
        self.client_id = client_dict['client_id']
        self.group_name = client_dict['group_name']
        self.device = 'cuda:{}'.format(client_dict['device'])

        #self.logger.info(f"Client {self.client_id}:{self.group_name} initialized in process `{current_process().name}` on {self.device}")

        self.model_type = client_dict['model_type']
        self.model_config = client_dict['model_config']
        self.model = self.model_type(**self.model_config).to(self.device)
        self.args = args
        self.round = 0
        self.train_dataloader =  client_dict['train_data']
        self.test_dataloader = client_dict['test_data']
        self.ravg = RunningAverage()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,betas=(0.9, 0.999))  # Default betas used in Adam)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_steps)
        self.eval_num_qs = args.eval_num_qs
        self.dataset = client_dict['dataset']
        self.tracker:Tracker = client_dict['tracker']
        self.train_sample = 0
        
    
    def load_client_state_dict(self, server_state_dict):
        self.logger.debug(f"[Round {self.round}] Client {self.client_id}:{self.group_name} loading model weights from server")
        self.model.load_state_dict(server_state_dict)
    
    def run(self, received_info):
        """
        Main entry point for the client during each FL round.
        Loads server weights, performs training, and returns results.

        Args:
            received_info (dict): Server-sent model state dict.

        Returns:
            List[dict]: Contains training results such as weights, metrics, and sample (questions) count.
        """
        
        self.logger.debug(f"[Round {self.round}] Client {self.client_id}:{self.group_name} started training")
        client_results = []
        
        self.load_client_state_dict(received_info)
        weights, loss = self.train()

        #alignment_score = self.test()
        client_results.append({'group_name':self.group_name,'weights':weights, 'num_samples':self.train_sample,'ravg':self.ravg,
                               'client_index':self.client_id, 'alignment_score':0, "loss":loss})
        
        self.logger.debug(f"[Round {self.round}] Client {self.client_id} completed training with {self.train_sample} samples")
        
        self.round += 1
        return client_results
        
    def train(self):
        """
        Trains the client model locally for a fixed number of epochs.

        Returns:
            Tuple[dict, dict]: 
                - Updated model weights (as state_dict)
                - Dictionary of average losses/metrics
        """
        self.logger.debug(f"[Round {self.round}] Client {self.client_id}:{self.group_name} started local training for {self.args.epochs} epochs")
        # train the local model
        self.model.to(self.device)
        self.model.train()
        self.ravg.reset()        
        client_start_time = time.time()
        for epoch in range(self.args.epochs):
            self.optimizer.zero_grad()
            # The training loop processes only a single batch per epoch. 
            # For more details on how the batch is constructed, refer to the implementation of the custom DataLoader.
            for batch in self.train_dataloader:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                if(epoch == 0):
                    self.train_sample = batch['y'].shape[1]
                outs = self.model(batch)
                outs.loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            for key, val in outs.items():
                self.ravg.update(key, val)
        #this is the average loss for all the epochs. logged in wandb
        self.tracker.wandb_loss.update(running_average_to_dict(self.ravg,source=self.group_name))
        weights = self.model.cpu().state_dict()
        self.logger.debug(f"[Round {self.round}] Client {self.client_id}:{self.group_name} finished training. Returning updated weights.")
        client_end_time = time.time()  # Record the end time
        client_elapsed_time = client_end_time - client_start_time  # Calculate elapsed time
        self.tracker.add_client_time_tracker(self.round, self.group_name, client_elapsed_time)
        return weights,running_average_to_dict(self.ravg,source=self.group_name)
    
    def test(self):
        """
        Evaluates the client model using the appropriate metric for the dataset.
        the GLOBALQA dataset uses Jensen distance, while the OQA dataset uses Wasserstein distance.
        GlobalQA is a dataset that is supported now.

        Returns:
            float: Alignment score (Jensen or Wasserstein)
        """
        self.logger.info(f"[Round {self.round}] Evaluating model on dataset `{self.dataset}`")
        self.model.to(self.device)
        if self.dataset == DatasetKeys.OQA.value:
            score = calculate_WD(self.eval_num_qs, self.model, self.eval_df, mode='eval')
            self.logger.info(f"[Round {self.round}] Group: {self.group_name} Calculated Average Wasserstein Distance Score: {score:.4f}")
            return score
        elif self.dataset == DatasetKeys.GLOBALQA.value:
            score = calculate_JD(self.eval_num_qs, self.model, self.train_dataloader.dataset, mode='eval')
            self.logger.info(f"[Round {self.round}] Group: {self.group_name} Calculated Jensen Distance Score: {score:.4f}")
            return score
        else:
            self.logger.error(f"[Round {self.round}] Dataset `{self.dataset}` is not supported.")
            raise ValueError(f"Dataset `{self.dataset}` not supported")