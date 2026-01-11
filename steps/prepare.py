import os

import argparse

from transformers import PreTrainedTokenizerFast

from src.data_processing import load_initial_data, process_data, create_tokenizer, create_dataloader
from src.utils import logger_configure, manual_seed, load_config_params

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_params_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, required=False, default=0, help='Включает логгирование в поток вывода (командную строку)')

    args = parser.parse_args()

    yaml_config_path = args.config_params_path
    verbose = args.verbose

    config_params = load_config_params(yaml_config_path)
    logger = logger_configure(config_params, (verbose > 0))

    set_seed = config_params['set_fixed_seed']
    if set_seed:
        logger.info('Фиксация random_seed')
        manual_seed(config_params)

    logger.info('Предобработка данных')
    process_data(config_params)

    logger.info('Обучение токенизатора')
    tokenizer = create_tokenizer(config_params)

if __name__ == "__main__":
    main()