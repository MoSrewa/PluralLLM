# ---------------------- Imports ----------------------
import os                         # For file and directory operations
import time                       # For timestamps and measuring durations
import yaml                       # To load YAML config files
import torch                      # PyTorch framework
import random                     # For random seed control
import argparse                   # For command-line argument parsing
import numpy as np               # Numerical computations
import wandb                      # Weights & Biases for experiment tracking
from tqdm import tqdm             # For progress bar

# ------------------ Project-specific imports ------------------
from utils import (
    get_logger,                   # Logger setup function
    add_args,                     # Function to define and parse CLI arguments
    Tracker,                      # Custom class to track performance, logs, etc.
    DatasetKeys,                  # Enum for dataset names (OQA, GLOBALQA, etc.)
    EmbeddidngModelType           # Enum for embedding model types (e.g., alpaca, llama)
)

from data_loaders.data_providers import GlobalQAProvider, DataProvider  # Dataset-specific loaders
from methods import FedAvg_Server, FedAvg_Client, Centralized_Server    # FL + Centralized strategies
from models.gpo import GPO     

# Setup Functions
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    
    # Set up logging
    logging = get_logger(__name__)
    
    # Set up argument parser
    args = add_args()

    set_random_seed(args.seed)

    data_provider: DataProvider = None
    
    # ------------------- Dataset Setup -------------------
    if args.dataset ==DatasetKeys.GLOBALQA.value:
        wandb.init(project='group-alignment-gpo-anthropic', name=args.wndb_project_name, config=args)
        data_provider = GlobalQAProvider(args)
    # @todo include the OQAProvider in extenstion 
    elif args.dataset == DatasetKeys.OQA.value:
        wandb.init(project='group-alignment-gpo-oqa', name=args.wndb_project_name, config=args)
        pass
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')
    
    # ------------------- Load Data -------------------
    # Load global and local dataloaders and group splits
    train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_dict,\
    test_data_local_dict,train_groups,eval_groups = data_provider.load_data_set()
    
    # ------------------- Tracking -------------------
    # Create a Tracker instance to log losses, times, scores, etc.
    tracker = Tracker(args,train_groups,eval_groups,wandb)

    # ------------------- Model Config -------------------
    # Load model hyperparameters from a YAML config file
    with open(f'models/gpo/configs/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    wandb.config.update(config)

   # Set embedding dimension according to model type
    if args.emb_model == EmbeddidngModelType.ALPACA.value:
        config['dim_x'] = 4096
    elif args.emb_model == EmbeddidngModelType.LLAMA.value:
        config['dim_x'] = 5120
    else:
        raise ValueError(f'Embedding model {args.emb_model} not supported')
    
    # ------------------- Server & Client Setup -------------------
    # Common dict for server initialization
    server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': GPO,'model_config':config,
                    "train_groups": train_groups, "test_groups": eval_groups, "dataset": args.dataset, "tracker":tracker}
    
    # Build individual client dictionaries (1 per train group)
    client_dict = [{'train_data': train_data_local_dict[i], 'test_data': test_data_local_dict[i], 'device': i % torch.cuda.device_count(),
                     'model_type': GPO,'model_config':config, "tracker":tracker, "dataset": args.dataset,'client_id':i,'group_name':str(group_name)} for i , group_name in enumerate(train_groups)]
    
    centralized_server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': GPO,'model_config':config,
                    "train_groups": train_groups, "test_groups": eval_groups, "dataset": args.dataset, "tracker":tracker}
    
 
    # Instantiate client objects
    client_info = []
    for i in range(len(client_dict)):
        client_info.append(FedAvg_Client(client_dict[i], args))

    # ------------------- Output & Save Path -------------------
    # Define path where logs and checkpoints will be saved
    server_dict['save_path'] = '{}/logs/{}__{}_e{}_c{}_s{}'.format(os.getcwd(),
        time.strftime("%Y%m%d_%H%M%S"), args.method, args.epochs, args.client_number,args.seed)
    if not os.path.exists(server_dict['save_path']):
        os.makedirs(server_dict['save_path']) 
    
    # ------------------- Server Instantiation -------------------
    centraliezed_server = Centralized_Server(server_dict, args)
    fl_server = FedAvg_Server(server_dict, args)
    server_outputs = fl_server.start()
    
    # ------------------- Federated Training Loop -------------------
    for r in tqdm(range(0, args.comm_round)):
        logging.debug(f'************** Round: {r} ***************')
        round_start = time.time()

        # Each client processes data and sends results back to the server
        client_outputs = []
        for i, client in enumerate(client_info):
            client_output = client.run(server_outputs[i])
            client_outputs.extend([x for x in client_output])
            logging.debug(f'Client {i} finished processing data')
        mean_alignment_score, mean_train_alignment_score, server_outputs = fl_server.run(client_outputs)
        
        # Log metrics to tracker
        tracker.add_centralized_model_accuracy_test(r, mean_alignment_score)
        tracker.add_model_accuracy_train(r, mean_train_alignment_score)
        round_end = time.time()
        logging.info(f'Round {r} Time: {round_end - round_start:.2f}s')
        tracker.add_fed_server_tracker_time(r, round_end - round_start)
    logging.info('Training Finished')
    
    # ------------------- Centralized Training (Optional) -------------------
    centraliezed_server.train()  # Train a model on the combined global dataset

    # ------------------- Save Final Logs -------------------
    tracker.save_results(server_dict['save_path'])  # Write metrics to file
