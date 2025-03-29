import torch
import time
from utils import get_logger
import json
from performance_metric import calculate_JD, calculate_WD
from utils import DatasetKeys, Tracker
from torch.utils.data import DataLoader
import wandb
from attrdict import AttrDict
import time

"""
Module: fedavg_server

This module defines a federated learning server using the FedAvg algorithm.
The server coordinates client updates, aggregates model parameters, evaluates global performance,
and tracks experiment results using WandB.

Classes:
    - FedAvg_Server: Simulates the central aggreagtion server in a federated learning setup.
"""
class FedAvg_Server():
    """
    Federated Learning Server using the FedAvg aggregation strategy.

    Responsibilities:
    - Aggregates client model updates using weighted averaging.
    - Periodically evaluates global model performance.
    - Logs loss and alignment metrics via WandB.
    - Saves checkpoints and tracks the best-performing global model.

    Attributes:
        model (torch.nn.Module): Global model shared among clients.
        train_data/test_data (DataLoader): DataLoaders for centralized evaluation. train/test dtaa represent groups
        args (Namespace): Parsed arguments for configuration.
        tracker (Tracker): Logging and performance tracking utility.
        train_groups/test_groups (List[str]): Group identifiers used in training/evaluation.
        device (str): CUDA device to use.
        save_path (str): Directory for saving checkpoints.
        eval_num_qs (int): Number of questions used for evaluation.
    """
    def __init__(self,server_dict, args):
        self.logger = get_logger(__name__)
        self.logger.info('Server Init with {}'.format(server_dict))
        self.train_data : DataLoader = server_dict['train_data']
        self.test_data : DataLoader = server_dict['test_data']
        self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        self.model_type = server_dict['model_type']
        self.model_config = server_dict['model_config']
        self.model = self.model_type(**self.model_config).to(self.device)
        self.highest_mean_alignment_score = 0.0
        self.mean_alignment_score = 0.0 #for the train groups
        self.mean_train_alignment_score = 0.0 #for the test groups
        self.round = 0
        self.args = args
        self.save_path = server_dict['save_path']
        self.dataset = server_dict['dataset']
        self.train_groups = server_dict['train_groups']
        self.test_groups = server_dict['test_groups']
        self.eval_num_qs = args.eval_num_qs
        self.tracker: Tracker = server_dict['tracker']
        
    def run(self, received_info):
        """
        Main entry point for the FL server during each communication round.
        Aggregates client updates, evaluates performance, logs metrics,
        and prepares updated model for next round.

        Args:
            received_info (List[dict]): Client training results.

        Returns:
            Tuple[float, float, List[dict]]: 
                - Mean test alignment score,
                - Mean train alignment score,
                - Updated global model weights (one copy per client).
        """
        acumloss = 0
        for x in received_info:
            for loss_key, loss_value in x['loss'].items():
                if "loss" in loss_key:
                    acumloss+=loss_value
        self.tracker.wandb_loss.update({'federated - loss': acumloss /len(received_info), 'train_step':self.round})
        wandb.log(self.tracker.wandb_loss)
        self.tracker.wandb_loss = {}
        
        server_start_time = time.time()  # Record the start time
        server_outputs = self.operations(received_info)
        server_end_time = time.time()  # Record the end time
        server_elapsed_time = server_end_time - server_start_time  # Calculate elapsed time
        self.tracker.add_centralized_server_tracker_time(self.round, server_elapsed_time)
            
        if self.round % self.args.federated_eval ==0:
            ckpt = AttrDict()
            ckpt.model = self.model.state_dict()
            ckpt.step = self.round + 1
            torch.save(ckpt, '{}/{}'.format(self.save_path, f'federated_model_{self.round}.tar'))
            self.mean_alignment_score = self.test()
            if self.highest_mean_alignment_score < self.mean_alignment_score:
                torch.save(ckpt, '{}/{}'.format(self.save_path, f'federated_best_model_{self.round}.tar'))
                self.logger.debug("saving model in federated aggregation")
                self.highest_mean_alignment_score = self.mean_alignment_score
            self.mean_train_alignment_score = self.test(is_train=True)
            #this is the alignment score for all groups. logged in wandb
            self.tracker.wandb_alignment.update({"eval_step": self.round})
            wandb.log(self.tracker.wandb_alignment)
            self.tracker.wandb_alignment = {}
            
        self.round += 1
        return self.mean_alignment_score,self.mean_train_alignment_score, server_outputs
    
    def start(self):
        with open('{}/fl_config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [self.model.cpu().state_dict() for x in range(len(self.train_groups))]

    def operations(self, client_info):
        """
        Performs FedAvg aggregation on client weights.

        Args:
            client_info (List[dict]): List of client outputs containing weights and sample counts.

        Returns:
            List[dict]: Aggregated model weights replicated per client.
        """
        self.logger.debug(f"[Round {self.round}] Processing {len(client_info)} client updates for aggregation.")
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]
        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(len(self.train_groups))]
    
    def test(self,is_train=False):
        """
        Evaluates the global model on the test or train dataset using appropriate metrics.
        the GLOBALQA dataset uses Jensen distance, while the OQA dataset uses Wasserstein distance.
        GlobalQA is a dataset that is supported now.
        Args:
            is_train (bool): If True, evaluates on training data; otherwise on test data.

        Returns:
            float: Calculated alignment score (Wasserstein or Jensen-Shannon distance).
        """
        self.model.eval()
        #self.logger.info(f"[Round {self.round}] Evaluating model on dataset `{self.dataset}`")
        self.model.to(self.device)
        if self.dataset == DatasetKeys.OQA.value:
            score = calculate_WD(self.eval_num_qs, self.model, self.eval_df, mode='eval')
            self.logger.info(f"[Round {self.round}] Calculated Average Wasserstein Distance Score: {score:.4f}")
            return score
        elif self.dataset == DatasetKeys.GLOBALQA.value:
            score = calculate_JD(self.eval_num_qs, self.model, self.test_data.dataset if not is_train else self.train_data.dataset, mode='eval' if not is_train else 'train',step_size=self.round,tracker=self.tracker)
            self.logger.info(f"[Round {self.round}] Calculated Average Jensen Distance Score: {score:.4f}")
            return score
        else:
            self.logger.error(f"[Round {self.round}] Dataset `{self.dataset}` is not supported.")
            raise ValueError(f"Dataset `{self.dataset}` not supported")