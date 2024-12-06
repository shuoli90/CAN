This is the repo for the submission [*One-Shot Safety Alignment for Large Language Models via Optimal Dualization*](https://arxiv.org/abs/2405.19544).

# The folders are organized as follows:
- *safe-rlhf*: contain our optimal and dual optimization algorithms; the dual optimization implementations are under *safe-rlhf/trainer*; and primal optimization algorithms are under *algorithm/cdpo* fodler; evaluation 
- *output*: folder to save our generated results
- *script*: bash files to generate response and collect model-based safety and helpfulness scores

# Collected data
We have also uploaded our collected scores to run our primal/dual optimization algorithms. Please find the collected results from this link: [Collected data](https://drive.google.com/file/d/142yNqzgb4iS60lnnCkTSonsWyDZDvcf6/view?usp=sharing)

# Steps:
- Download the collected results; put data into corresponding folders. Detailed instructions can be found from the readme.md in the collected data folder.
- Setup the virual environment:
Setup a conda environment using [`conda`](https://github.com/conda/conda) / [`mamba`](https://github.com/mamba-org/mamba):

```bash
conda env create --file conda-recipe.yaml  # or `mamba env create --file conda-recipe.yaml`
conda activate safe-rlhf
```
- To run dual optimization, please enter into *safe-rlhf/trainer/ folder:
-- run *model_based_dual_trainer.ipynb* for model-based dual optimization
-- run *preference_based_dual_trainer.ipynk* for preference-based dual optimization.
- To run primal optimization, please enter into safe-rlhf/algorithms/cdpo*
-- run ```python dpo.py --lamb [LAMB] --output_dir [OUTPUT_DIR]``` for MoCAN
-- run ```python dpo_alg2.py --lamb [LAMB] --output_dir [OUTPUT_DIR]`` for PeCAN

# Citation
Please feel free to email us at [Xinmeng Huamg](mailto:xinmengh@sas.upenn.edu), [Shuo Li](mailto:lishuo1@seas.upenn.edu), or [Dongsheng Ding](mailto:dongshed@seas.upenn.edu). If you find this work useful in your own research, please consider citing our work:
```
@misc{huang2024oneshotsafetyalignmentlarge,
      title={One-Shot Safety Alignment for Large Language Models via Optimal Dualization}, 
      author={Xinmeng Huang and Shuo Li and Edgar Dobriban and Osbert Bastani and Hamed Hassani and Dongsheng Ding},
      year={2024},
      eprint={2405.19544},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.19544}, 
}
```
