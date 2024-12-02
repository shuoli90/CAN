import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap
from scipy.special import softmax
from scipy.optimize import minimize_scalar
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
        if optimizer == 'scipy':
            def dual_loss(lam):
                logits = (self.helpfulness_scores + (self.safety_scores - self.thresholds) * lam) /self. kl_coeff
                logits_max = logits.max(axis=1).reshape(-1, 1)
                return self.kl_coeff*np.mean(np.log(np.mean(np.exp(logits-logits_max)))+logits_max)

            result = minimize_scalar(dual_loss, bounds = (-1, 10))
            if set_optimum:
                self.lam_star = result.x

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
                if num_loops >= 10: # fail the optimization after 5 loops
                    print(f'The dual problem (threshold={self.thresholds}, sample_shape={self.helpfulness_scores.shape}) may not have a finite maximizer!')
                    if set_optimum:
                        self.lam_star = None
                    break
            return lam_trajectory, objective_trajectory, constraint_trajectory, helpfulness_trajectory, safety_trajectory if is_converge else [None] * 5


if __name__ == '__main__':
    
    full_scores_v1 = torch.load('/home/xinmeng/safe-rlhf/output/score/alpaca-7b-reproduced_cost_reward_scores_v1.pt')[:,:,-2:].cpu().numpy()
    full_helpfulness_scores_v1 = full_scores_v1[:,:,0]
    full_safety_scores_v1 = -full_scores_v1[:,:,1]

    full_scores_new2_v1 = torch.load('/home/xinmeng/safe-rlhf/output/score/alpaca-7b-reproduced_cost_reward_scores_new2_v1.pt')[:,:,-2:].cpu().numpy()
    full_helpfulness_scores_new2_v1 = full_scores_new2_v1[:,:,0]
    full_safety_scores_new2_v1 = -full_scores_new2_v1[:,:,1]

    length = min(full_scores_v1.shape[0], full_scores_new2_v1.shape[0])
    full_helpfulness_scores = np.concatenate([full_helpfulness_scores_v1[:length], full_helpfulness_scores_new2_v1[:length]], axis=1)
    full_safety_scores = np.concatenate([full_safety_scores_v1[:length], full_safety_scores_new2_v1[:length]], axis=1)


    num_prompt = 1000
    num_response = 128

    np.random.seed(42)
    idx_sample = np.random.choice(length, 1000, replace=False)

    plt.scatter(full_safety_scores[idx_sample, :128], full_helpfulness_scores[idx_sample, :128], s=0.1)
    plt.xlabel(r'$r(x,y)$', fontsize=15)
    plt.ylabel(r'g(x,y)', fontsize=15)
    plt.xlim(-20, 20)
    plt.ylim(-25,10)
    plt.show()
    plt.savefig('dual_boundedness.pdf')