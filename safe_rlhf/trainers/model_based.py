import numpy as np
import torch
from scipy.special import softmax
import matplotlib.pyplot as plt
import pandas as pd

class DualOptimizer():
    def __init__(self,
                 helpfulness_scores,
                 safety_scores,
                 thresholds, # E_{pi}[safety] >= thresholds
                 kl_coeff,
                 **kwargs):
        self.helpfulness_scores = helpfulness_scores
        self.safety_scores = safety_scores
        self.thresholds = thresholds
        self.kl_coeff = kl_coeff
        self.kwargs = kwargs
    
    def log_mean_Z_values(self, logits):
        return np.log(np.mean(np.exp(logits-logits.max(axis=1).reshape(-1, 1)), axis=1))+logits.max(axis=1)
    
    def solve(self, optimizer='GD', set_optimum=False, verbose=False, **kwargs):
        lam_init = 1 if 'lam_init' not in kwargs.keys() else kwargs['lam_init']
        lr = 1 if 'lr' not in kwargs.keys() else 2*kwargs['lr']
        max_iters = 200 if 'num_iters' not in kwargs.keys() else kwargs['num_iters']
        err = 1e-5 if 'err' not in kwargs.keys() else kwargs['err']
        if optimizer == 'GD':
            is_converge = False
            num_loops = 0
            while not is_converge:
                lam = lam_init
                lr = lr/2**num_loops # learning rate decay
                lam_trajectory = []
                objective_trajectory = []
                constraint_trajectory = []
                helpfulness_trajectory = []
                safety_trajectory = []
                for idx_iter in range(max_iters):
                    logits = (self.helpfulness_scores + (self.safety_scores - self.thresholds) * lam) / self.kl_coeff
                    sm_probs = softmax(logits, axis=1)
                    gradient = np.sum(sm_probs * (self.safety_scores - self.thresholds), axis=1).mean()
                    lam = np.maximum(lam - lr*gradient, 0)
                    lam_trajectory.append(lam)
                    objective_trajectory.append(self.kl_coeff*np.mean(self.log_mean_Z_values(logits))-lam*gradient)
                    constraint_trajectory.append(gradient)
                    helpfulness_trajectory.append(np.sum(sm_probs * self.helpfulness_scores, axis=1).mean())
                    safety_trajectory.append(np.sum(sm_probs * self.safety_scores, axis=1).mean())
                    if idx_iter >= 10 and np.abs(lam-np.array(lam_trajectory[-1:-6:-1])).max()<err: # the maximal difference, compared to the last 5 iterations, are smaller than 1e-5
                        is_converge = True
                        if verbose:
                            print(f'The optimization converges to a finite maximizer!')
                        break
                if is_converge:
                    if set_optimum:
                        self.lam_star = lam
                    break
                num_loops += 1
                if num_loops >= 5: # fail the optimization after 5 loops
                    print(f'The dual problem (threshold={self.thresholds}, sample_shape={self.helpfulness_scores.shape}) may not have a finite maximizer!')
                    if set_optimum:
                        self.lam_star = None
                    break
            return lam_trajectory, objective_trajectory, constraint_trajectory, helpfulness_trajectory, safety_trajectory if is_converge else [None] * 5

full_scores_v1 = torch.load('/home/xinmeng/safe-rlhf/output/score/alpaca-7b-reproduced_cost_reward_scores_v1.pt')[:,:,-2:].cpu().numpy()
full_helpfulness_scores_v1 = full_scores_v1[:,:,0]
full_safety_scores_v1 = -full_scores_v1[:,:,1]
print(f'The expected (helpfulness, safety) score of the reference model is {full_helpfulness_scores_v1.mean(), full_safety_scores_v1.mean()}')

full_scores_new2_v1 = torch.load('/home/xinmeng/safe-rlhf/output/score/alpaca-7b-reproduced_cost_reward_scores_new2_v1.pt')[:,:,-2:].cpu().numpy()
full_helpfulness_scores_new2_v1 = full_scores_new2_v1[:,:,0]
full_safety_scores_new2_v1 = -full_scores_new2_v1[:,:,1]
print(f'The expected (helpfulness, safety) score of the reference model is {full_helpfulness_scores_new2_v1.mean(), full_safety_scores_new2_v1.mean()}')

length = min(full_scores_v1.shape[0], full_scores_new2_v1.shape[0])

full_helpfulness_scores = np.concatenate([full_helpfulness_scores_v1[:length], full_helpfulness_scores_new2_v1[:length]], axis=1)
full_safety_scores = np.concatenate([full_safety_scores_v1[:length], full_safety_scores_new2_v1[:length]], axis=1)

threshold_grid = np.linspace(-0.08463882, 2, 20)
num_prompt_grid = list(np.arange(100, 1401, 100))+[full_helpfulness_scores.shape[0]]

lam_tp_list = np.full((len(threshold_grid), len(num_prompt_grid)), None, dtype=object)
for idx_lb, threshold in enumerate(threshold_grid):
    for idx_num, num_prompt in enumerate(num_prompt_grid):
        do = DualOptimizer(full_helpfulness_scores[:num_prompt], full_safety_scores[:num_prompt], thresholds=threshold, kl_coeff=0.1)
        do.solve(lr=2, max_iters=200, set_optimum=True, errr=1-6)
        lam_tp_list[idx_lb][idx_num] = do.lam_star