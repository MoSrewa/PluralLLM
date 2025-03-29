import os.path as osp
import ast
import torch
import numpy as np
from scipy.stats import wasserstein_distance
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from utils import Tracker
from scipy.spatial import distance
"""
Module: alignment_score

This module evaluates the alignment performance of a prediction model using Jensen-Shannon Divergence (JSD).
The key function, `calculate_JD`, computes alignment scores by comparing predicted distributions to ground truth
distributions over a group of questions. It supports optional logging using Weights & Biases (wandb).

# This code is originally written in https://github.com/jamqd/Group-Preference-Optimization

"""

def calculate_JD(eval_num_qs, model, dataset, mode='eval', logging=True, type='fed', tracker: Tracker = None, step_size = 0):
    model.eval()
    #batch size = 1, because each batch is a group. 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    distances_all = []
    for i, batch in enumerate(dataloader):
        distances_group = []
        this_group = batch['groups']
        group_questions = batch['questions']
        num_questions = len(group_questions)
        context_questions = np.random.choice(np.arange(num_questions), size=eval_num_qs, replace=False)
        target_questions = np.setdiff1d(np.arange(num_questions), context_questions)
        # Now, let's collect the context embeddings and probabilities.
        ctx_embeddings = []
        ctx_prob_ys = []
        tar_embeddings = []
        tar_prob_ys = []
        for context_q_idx in context_questions:
            ctx_embeddings.append(group_questions[context_q_idx]['q_emb'])
            ctx_prob_ys.append(group_questions[context_q_idx]['prob_ys'][0])
        ctx_embeddings = torch.cat(ctx_embeddings, dim=1).to('cuda')
        ctx_prob_ys = torch.cat(ctx_prob_ys, dim=1).unsqueeze(-1).to('cuda', dtype=torch.float)
        for target_q_idx in target_questions:
            tar_embeddings = group_questions[target_q_idx]['q_emb'].to('cuda')
            tar_prob_ys = group_questions[target_q_idx]['prob_ys'].to('cuda')
            with torch.no_grad():
                predicted_distribution = model.predict(ctx_embeddings, ctx_prob_ys, tar_embeddings).loc
                predicted_distribution = softmax_normalize(predicted_distribution.reshape(-1))
                    

                D_H = tar_prob_ys
                D_H_np = np.array(D_H.cpu())
                D_H_np = D_H_np.squeeze()
                predicted_distribution_np = predicted_distribution.cpu().detach().numpy().squeeze()
                normalized_jd = distance.jensenshannon(predicted_distribution_np, D_H_np)
                if torch.isnan(torch.tensor(normalized_jd)).any():
                    normalized_jd = 0.0
                distances_all.append(normalized_jd)
                distances_group.append(normalized_jd)
        mean_distance_group = np.mean(distances_group)
        if logging:
            #wandb.log({f"{type.capitalize()}-{mode.capitalize()}_alignment_score_{this_group}": 1 - mean_distance_group})
            tracker.wandb_alignment.update({f"{type.capitalize()}-{mode.capitalize()}_alignment_score_{this_group}": 1 - mean_distance_group})
            # print({f"{mode.capitalize()}_alignment_score_{this_group}": 1 - mean_distance_group})
        # print(f"{mode.capitalize()}_alignment_score_{this_group}:  {1 - mean_distance_group}")
    mean_distance = np.mean(distances_all)
    #print(f"{mode.capitalize()} Mean Jensen Divergence: {mean_distance} Mean alignment score:{1-mean_distance}")
    if logging:
        #wandb.log({f"{type.capitalize()}_{mode.capitalize()}_alignment_score_mean": 1-mean_distance})
        tracker.wandb_alignment.update({f"{type.capitalize()}_{mode.capitalize()}_alignment_score_mean": 1-mean_distance})
        # print({f"{mode.capitalize()}_alignment_score_mean_testgroup": 1-mean_distance})
    return 1-mean_distance

def softmax_normalize(tensor):
    """Applies softmax normalization along the last dimension of the tensor"""
    return F.softmax(tensor, dim=-1)

