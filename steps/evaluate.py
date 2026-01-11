import os

from deep_translator import GoogleTranslator
import fasttext
import argparse
import json

from transformers import PreTrainedTokenizerFast
from transformers import (
    GPT2LMHeadModel
)

import time

from src.utils import logger_configure, manual_seed, load_config_params

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_params_path', type=str, required=True)
    parser.add_argument('--input_sequence', type=str, required=True)

    args = parser.parse_args()

    yaml_config_path = args.config_params_path
    input_sequence = args.input_sequence

    config_params = load_config_params(yaml_config_path)
    logger = logger_configure(config_params, 1)

    set_seed = config_params['set_fixed_seed']
    if set_seed:
        logger.info('Фиксация random_seed')
        manual_seed(config_params)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config_params['inference']['tokenizer_path'])
    model = GPT2LMHeadModel.from_pretrained(config_params['inference']['model_path'])

    device = config_params['inference']['device']
    max_length = config_params['inference']['max_length']
    do_sample = config_params['inference']['sample']
    top_p = config_params['inference']['top_p']
    temperature = config_params['inference']['temperature']
    repetition_penalty = config_params['inference']['repetition_penalty']

    model.to(device)
    model.eval()

    language = config_params['inference']['language']

    logger.info(f'Запуск инференса на устройстве {device}.')
    logger.info(f'Последовательность на вход: {input_sequence}')
    
    input_sequence_ru = GoogleTranslator(source=language, target='ru').translate(input_sequence)
    logger.info(f'Перевод входной последовательности на русский: {input_sequence_ru}')

    ids = tokenizer(input_sequence, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
    
    start_time = time.time()
    outputs = model.generate(ids, max_length=max_length, do_sample=do_sample, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty)
    result_time = time.time() - start_time
    
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logger.info(f'Ответ модели: {outputs}.')
    outputs_ru = GoogleTranslator(source=language, target='ru').translate(outputs)
    logger.info(f'Перевод ответа модели на русский: {outputs_ru}')
    logger.info(f'Время получения ответа в секундах: {result_time:.4f}')

    language_checker_path = config_params['inference']['language_checker_path']
    language_checker = fasttext.load_model(language_checker_path)

    result_language = language_checker.predict(outputs.replace('\n', ' '))
    result_language_prob = result_language[1].item()
    result_language = result_language[0][0].split('__label__')[1]

    logger.info(f'Язык ответа по fasttext: {result_language}, вероятность языка: {result_language_prob}')

    final_metrics = dict()
    final_metrics['Result_language'] = result_language
    final_metrics['Result_language_prob'] = result_language_prob
    final_metrics['Result_cs'] = outputs
    final_metrics['Result_ru'] = outputs_ru
    final_metrics['Inference_time'] = result_time

    with open('evaluate_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_metrics , f) 


if __name__ == "__main__":
    main()