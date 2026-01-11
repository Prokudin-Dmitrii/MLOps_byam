import os

import argparse

from transformers import PreTrainedTokenizerFast
from transformers import (
    GPT2LMHeadModel
)

import torch

import time

from src.utils import manual_seed, load_config_params

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_params_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    args = parser.parse_args()

    yaml_config_path = args.config_params_path
    input_path = args.input_path
    output_path = args.output_path

    config_params = load_config_params(yaml_config_path)

    set_seed = config_params['set_fixed_seed']
    if set_seed:
        manual_seed(config_params)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config_params['inference']['tokenizer_path'])
    model = GPT2LMHeadModel.from_pretrained(config_params['inference']['model_path'])

    device = config_params['inference'].get('device', 'cpu')

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    max_length = config_params['inference']['max_length']
    do_sample = config_params['inference']['sample']
    top_p = config_params['inference']['top_p']
    temperature = config_params['inference']['temperature']
    repetition_penalty = config_params['inference']['repetition_penalty']

    model.to(device)
    model.eval()

    language = config_params['inference']['language']

    with open(input_path, 'r') as file:
        input_sequences = [input_sequence.rstrip() for input_sequence in file]

    output_sequences = []

    for input_sequence in input_sequences:
        ids = tokenizer(input_sequence, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
    
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(ids, max_length=max_length, do_sample=do_sample, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty)
        result_time = time.time() - start_time
    
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_sequences.append(outputs)
        
    with open(output_path, 'w') as file:
        for output_sequence in output_sequences:
            file.write(output_sequence + '\n')

if __name__ == "__main__":
    main()