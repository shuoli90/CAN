for MODEL in "MoCAN_b0.1_l1.0_v3" "MoCAN_b0.1_l3.0_v3" "MoCAN_b0.1_l5.0_v3"
do
    python -m safe_rlhf.evaluate.gpt4.gpt_generate --model_name_or_path PKU-Alignment/alpaca-7b-reproduced --peft_dir $MODEL \
    --mode safety --output_dir output/gpt_generate_saferlhf
    python -m safe_rlhf.evaluate.gpt4.gpt_generate --model_name_or_path PKU-Alignment/alpaca-7b-reproduced --peft_dir $MODEL \
        --mode help --output_dir output/gpt_generate_saferlhf

    python -m safe_rlhf.evaluate.gpt4.gpt_eval --red_model_name alpaca-7b-reproduced \
        --blue_model_name $MODEL \
        --mode safety --output_dir output/gpt_eval
    python -m safe_rlhf.evaluate.gpt4.gpt_eval --red_model_name alpaca-7b-reproduced \
        --blue_model_name $MODEL \
        --mode help --output_dir output/gpt_eval
done