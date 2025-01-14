# for MODEL in "dpo_safety_b0.025" "dpo_safety_b0.1" "dpo_help_b0.025" "dpo_help_b0.1" "lamb_0.05_new" "lamb_0.1_new" "lamb_0.15_new" "lamb_0.2_new" "lamb_0.25_new" "lamb_0.32_new" "lamb_1.0_new" "lamb_3.2_new" "MoCAN_b0.1_l0.1" "MoCAN_b0.1_l0.3511/checkpoint-1000" "MoCAN_b0.1_l0.5" "MoCAN_b0.1_l0.7547/checkpoint-800" "MoCAN_b0.1_l1.25" "MoCAN_b0.01_l1.1263" "MoCAN_b0.01_l2.0"
for MODEL in "MoCAN_b0.1_l1.0_v3" "MoCAN_b0.1_l3.0_v3" "MoCAN_b0.1_l5.0_v3"
do
    # python3 -m safe_rlhf.evaluate.gpt4.gpt_eval_rebuttal --red_model_name alpaca-7b-reproduced \
    #     --blue_model_name $MODEL --dataset Truthful \
    #     --mode safety --output_dir output/gpt_eval_rebuttal
    # python3 -m safe_rlhf.evaluate.gpt4.gpt_eval_rebuttal --red_model_name alpaca-7b-reproduced \
    #     --blue_model_name $MODEL --dataset Advbench \
    #     --mode safety --output_dir output/gpt_eval_rebuttal
    python -m safe_rlhf.evaluate.gpt4.gpt_eval --red_model_name alpaca-7b-reproduced \
        --blue_model_name $MODEL \
        --mode safety --output_dir output/gpt_eval_rebuttal
    python -m safe_rlhf.evaluate.gpt4.gpt_eval --red_model_name alpaca-7b-reproduced \
        --blue_model_name $MODEL \
        --mode help --output_dir output/gpt_eval_rebuttal
done

# for MODEL in "dpo_safety_b0.025" "dpo_safety_b0.01" "dpo_help_b0.025" "dpo_help_b0.1"
# do
#     python3 -m safe_rlhf.evaluate.gpt4.gpt_eval_rebuttal --red_model_name alpaca-7b-reproduced \
#         --blue_model_name $MODEL --dataset Truthful \
#         --mode safety --output_dir output/gpt_eval_rebuttal
#     python3 -m safe_rlhf.evaluate.gpt4.gpt_eval_rebuttal --red_model_name alpaca-7b-reproduced \
#         --blue_model_name $MODEL --dataset Advbench \
#         --mode safety --output_dir output/gpt_eval_rebuttal
# done