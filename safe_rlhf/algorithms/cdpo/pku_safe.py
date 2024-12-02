# from torch.utils.data import Dataset
from datasets import Dataset
from datasets import load_dataset
from typing import Dict
import json

def PKU():

    def __init__(self, prefer_safe: bool = True, tokenizer=None) -> None:
        
        self.dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
        self.prefer_safe = prefer_safe
        self.tokenizer = tokenizer

        def return_prompt_and_responses(samples) -> Dict[str, str, str]:
            response_0 = samples["response_0"]
            response_1 = samples["response_1"] 
            responses_list = zip(response_0, response_1)
            if self.prefer_safe:
                indices = samples["safer_response_id"]
            else:
                indices = samples['better_response_id']
            return {
                "prompt": [
                    tokenizer.encode("Question: " + question + "\n\nAnswer: ", return_tensors='pt') for question in samples["prompt"]
                ],
                "chosen": [tokenizer.encode(responses[safer_index], return_tensors='pt') for responses, safer_index in zip(responses_list, indices)],
                "rejected": [tokenizer.encode(responses[1 - safer_index], return_tensors='pt') for responses, safer_index in zip(responses_list, indices)]
            }
        
        self.dataset = self.dataset.map(return_prompt_and_responses)

class PKU_C(Dataset):

    def __init__(self, file_path: str = None, lamb: float = 0.0) -> None:
        with open(file_path, 'r') as f:
            self.dataset = json.load(f)

        self.dataset = Dataset.from_dict(self.dataset)
        self.lamb = lamb
    
        def return_prompt_and_responses(samples):
            weight_0s = samples['reward_scores_0'] + self.lamb * samples['cost_scores_0']
            weight_1s = samples['reward_scores_1'] + self.lamb * samples['cost_scores_1']
            indices = [0 if weight_0 > weight_1 else 1 for weight_0, weight_1 in zip(weight_0s, weight_1s)]
            breakpoint()
            return {
                    "prompt": [sample['prompt_ids'] for sample in samples],
                    "chosen": [samples["response_0_ids"][i] if index == 0 else samples["response_1_ids"][i] for i, index in enumerate(indices)],
                    "rejected": [samples["response_1_ids"][i] if index == 0 else samples["response_0_ids"][i] for i, index in enumerate(indices)]
                }

        self.dataset = self.dataset.map(return_prompt_and_responses)
    