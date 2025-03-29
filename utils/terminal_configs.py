# import argparse
# #@todo add all these configurations to a yml file
# def add_args(parser):
#     # Training settings
#     parser.add_argument('--method', type=str, default='fedavg', metavar='N',
#                         help='Options are: fedavg')


#     parser.add_argument('--client_number', type=int, default=13, metavar='NN',
#                         help='number of clients in the FL system (including the Eval/Train Group)')
    
#     parser.add_argument('--epochs', type=int, default=20, metavar='EP',
#                         help='how many epochs will be trained locally per round')

#     parser.add_argument('--comm_round', type=int, default=25,
#                         help='how many rounds of communications are conducted')

#     parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
#                         help='learning rate (default: 3e-4)')


#     parser.add_argument('--save_client', action='store_true', default=False,
#                         help='Save client checkpoints each round')

#     #under the fact that all clients will be able to follow up each iteration.
#     #@todo: implement that
#     parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
#                         help='Fraction of clients to sample')
#     #--------------------------------------------------------------------------------------------
    
#     parser.add_argument('--max_ctx_num_qs', type=int, default=100)
#     parser.add_argument('--min_ctx_num_qs', type=int, default=10)
#     parser.add_argument('--max_tar_num_qs', type=int, default=100)
#     parser.add_argument('--min_tar_num_qs', type=int, default=10)
#     parser.add_argument('--emb_model', type=str, default='alpaca')

#     parser.add_argument('--dataset', type=str, default='globalqa', help='oqa or globalqa')
    
#     parser.add_argument('--eval_num_qs', type=int, default=20)
#     parser.add_argument('--eval_seed', type=int, default=0)

#     args = parser.parse_args()
#     #@todo externalize all this configuration to yml file.
#     args.method = 'fedavg'
#     args.comm_round = 100
#     args.epochs = 6
#     args.device = 'cuda:0'
#     args.gpu = 1    
#     args.group_split = 0.6
#     args.eval_num_qs = 20
#     args.num_steps = 100
#     args.model = 'gpo'
    
#     args.central_epochs = 100 
#     args.eval_freq = 10
#     args.federated_eval = 10
#     args.seed = 5000
    
    
#     args.embedding_folder = "llm_embedding"
#     args.max_ctx_num_qs = 100
#     args.min_ctx_num_qs = 10
#     args.max_tar_num_qs = 100
#     args.min_tar_num_qs = 10
#     args.emb_model = 'alpaca'
#     args.dataset = 'globalqa'
#     args.group_split = 0.6
#     args.data_split = 0.9
#     args.train_batch_size = 32
#     args.eval_batch_size = 16
    
#     args.wndb_project_name = 'ayla'
#     return args

import argparse

# @todo: Externalize all these configurations to a YAML file
def add_args():
    parser = argparse.ArgumentParser(description='Federated Learning Configurations')

    # Federated learning parameters
    parser.add_argument('--method', type=str, default='fedavg', help='Options: fedavg support only')
    parser.add_argument('--client_number', type=int, default=13, help='Number of clients (including Train/Eval group): Dont change this')
    parser.add_argument('--comm_round', type=int, default=100, help='Number of communication rounds')
    parser.add_argument('--epochs', type=int, default=6, help='Number of local epochs per round')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--save_client', action='store_true', help='Whether to save client checkpoints')

    # Model & training settings
    parser.add_argument('--model', type=str, default='gpo', help='Model name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--gpu', type=int, default=1, help='GPU id')
    parser.add_argument('--seed', type=int, default=5000, help='Random seed')

    # Centralized training (optional)
    parser.add_argument('--central_epochs', type=int, default=100, help='Number of centralized training epochs')

    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=10, help='Central evaluation frequency')
    parser.add_argument('--federated_eval', type=int, default=10, help='Frequency of federated evaluation')
    parser.add_argument('--eval_num_qs', type=int, default=20, help='Number of questions used for evaluation')
    parser.add_argument('--eval_seed', type=int, default=0, help='Random seed for evaluation')

    # Dataset and batching
    parser.add_argument('--dataset', type=str, default='globalqa', help='Dataset name: oqa or globalqa')
    parser.add_argument('--embedding_folder', type=str, default='llm_embedding', help='Folder containing embeddings')
    parser.add_argument('--group_split', type=float, default=0.6, help='Proportion of groups for training')
    parser.add_argument('--data_split', type=float, default=0.9, help='Train/test split inside each client/group')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of training steps (used as T_max in cosine learning rate decay)')

    # Context/Target configuration
    parser.add_argument('--max_ctx_num_qs', type=int, default=100)
    parser.add_argument('--min_ctx_num_qs', type=int, default=10)
    parser.add_argument('--max_tar_num_qs', type=int, default=100)
    parser.add_argument('--min_tar_num_qs', type=int, default=10)
    parser.add_argument('--emb_model', type=str, default='alpaca', help='Embedding model type')

    # WandB or other tracking
    parser.add_argument('--wndb_project_name', type=str, default='ayla', help='WandB project name')

    args = parser.parse_args()
    return args
