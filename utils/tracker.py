import os
import pickle

class Tracker:
    """
    Tracker class for managing metrics, timing, and experiment logging 
    during Federated Learning and Centralized training.

    Responsibilities:
    - Log accuracy, loss, and runtime per client and per round.
    - Interface with Weights & Biases (wandb) for online logging.
    - Persist results to disk using Pickle.
    """
    def __init__(self, args,train_group,eval_group,wandb):
        """
        Initializes trackers and wandb logging structure.

        Args:
            args: Command-line arguments with experiment configs.
            train_group (List[str]): Names of groups used for training.
            eval_group (List[str]): Names of groups used for evaluation.
            wandb: The wandb instance for online logging.
        """
        self.args = args
        # Time spent by each client per communication round
        self.client_time_tracker = {i: [0] * self.args.comm_round for i in train_group}
        
        # Time spent by the federated server per round
        self.fed_Server_tracker_time_tracker = [-1 for _ in range(self.args.comm_round)]
        
        # Time spent by the centralized server per round
        self.centralized_Server_time_tracker = [-1 for _ in range(self.args.comm_round)]

        # Accuracy values per round (federated) - it's alignment score in this case.
        self.model_accuracy_train = [-1 for _ in range(self.args.comm_round)]
        self.model_accuracy_test = [-1 for _ in range(self.args.comm_round)]

        # Accuracy values per round (centralized) - it's alignment score in this case.
        self.centralized_model_accuracy_train = [-1 for _ in range(self.args.comm_round)]
        self.centralized_model_accuracy_test = [-1 for _ in range(self.args.comm_round)]

        self.wandb = wandb
        
        # Buffers for storing metrics before wandb.log()
        self.wandb_alignment = {}
        self.wandb_loss = {}
        
        self.train_group = train_group
        self.eval_group = eval_group
        
        # Initialize wandb metric structure
        self.init_wandb()
        
    def init_wandb(self):
        """
        Initialize wandb metrics for logging.
        This includes defining metrics for loss, accuracy, and alignment scores.
        """
        self.wandb.define_metric("centralized - loss", step_metric="train_step")
        self.wandb.define_metric("federated - loss", step_metric="train_step")
        for i in self.train_group:
            self.wandb.define_metric(f"i - loss", step_metric="train_step")
            
        for type in ['Train','Eval']:
            self.wandb.define_metric(f"Fed_{type}_alignment_score_mean", step_metric="eval_step")
            self.wandb.define_metric(f"Centralized_{type}_alignment_score_mean", step_metric="eval_step")
            
            for group in self.train_group:
                self.wandb.define_metric(f"Centralized-{type}_alignment_score_{group}", step_metric="centralized_eval_step")
                self.wandb.define_metric(f"Fed-{type}_alignment_score_{group}", step_metric="eval_step")
                
            for group in self.eval_group:
                self.wandb.define_metric(f"Centralized-{type}_alignment_score_{group}", step_metric="eval_step")
                self.wandb.define_metric(f"Fed-{type}_alignment_score_{group}", step_metric="eval_step")
            

    def add_model_accuracy_test(self, iteration, accuracy):
        #self.wandb.log({'model_global_accuracy_test': accuracy})
        self.model_accuracy_test[iteration] = accuracy

    def add_model_accuracy_train(self, iteration, accuracy):
        #self.wandb.log({'model_global_accuracy_train': accuracy})
        self.model_accuracy_train[iteration] = accuracy

    def add_centralized_model_accuracy_test(self, iteration, accuracy):
        #self.wandb.log({'model_global_accuracy_test': accuracy})
        self.centralized_model_accuracy_test[iteration] = accuracy

    def add_centralized_model_accuracy_train(self, iteration, accuracy):
        #self.wandb.log({'model_global_accuracy_train': accuracy})
        self.centralized_model_accuracy_train[iteration] = accuracy

    def add_sia_accuracy(self, iteration, client_index, accuracy):
        self.sia_accuracy[client_index][iteration] = accuracy

    def add_sia_accuracy_avg(self, iteration, accuracy):
        self.sia_accuracy_avg[iteration] = accuracy
        
    def add_client_time_tracker(self, iteration, client_index, time_value):
        if client_index in self.client_time_tracker:
            self.client_time_tracker[client_index][iteration] = time_value
        else:
            print(f"Client index {client_index} not found in tracker.")

    def add_fed_server_tracker_time(self, iteration, time_value):
        self.fed_Server_tracker_time_tracker[iteration] = time_value
        
    def add_centralized_server_tracker_time(self, iteration, time_value):
        self.centralized_Server_time_tracker[iteration] = time_value


    def save_results(self, folder, filename='saved_tracker.pkl'):
        """
        Save all tracked results into a single pickle file for analysis.

        Args:
            folder (str): Directory to save results in.
            filename (str): Name of the pickle file.
        """
        file_path = os.path.join(folder, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb+') as f:
            pickle.dump({"fed_Server_tracker_time_tracker":self.fed_Server_tracker_time_tracker,
                         "centralized_Server_time_tracker": self.centralized_Server_time_tracker,
                         "client_time_tracker":self.client_time_tracker,
                          "model_accuracy_train": self.model_accuracy_train,
                        "model_accuracy_test": self.model_accuracy_test,
                        "centralized_model_accuracy_train": self.centralized_model_accuracy_train,
                        "centralized_model_accuracy_test":self.centralized_model_accuracy_test}, f)
        
        print(f'Results saved at {file_path}')