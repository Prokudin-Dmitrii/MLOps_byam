import os

import argparse

from transformers import PreTrainedTokenizerFast

from src.data_processing import load_initial_data, process_data, create_tokenizer, create_dataloader
from src.model import create_model
from src.utils import logger_configure, manual_seed, load_config_params
from src.training import model_training

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_params_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, required=False, default=0, help='Включает логгирование в поток вывода (командную строку)')

    args = parser.parse_args()

    yaml_config_path = args.config_params_path
    verbose = args.verbose

    config_params = load_config_params(yaml_config_path)

    #print(args)

    logger = logger_configure(config_params, (verbose > 0))

    set_seed = config_params['set_fixed_seed']
    if set_seed:
        logger.info('Фиксация random_seed')
        manual_seed(config_params)

    create_data_from_zero = config_params['create_data_from_zero']
    create_tokenizer_from_zero = config_params['create_tokenizer_from_zero']
    if create_data_from_zero:
        logger.info('Загрузка изначальных данных')
        load_initial_data(config_params)

        logger.info('Предобработка данных')
        process_data(config_params)
    
    if create_tokenizer_from_zero:
        logger.info('Обучение токенизатора')
        tokenizer = create_tokenizer(config_params)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config_params['data_processing']['wrapped_tokenozer_savefile_path'])
    
    logger.info('Создание dataloader\'a')
    train_dataloader = create_dataloader(config_params, tokenizer)

    logger.info('Создание модели')
    model = create_model(config_params, tokenizer)

    model, epoch_losses = model_training(config_params, logger, model, tokenizer, train_dataloader)

    logger.info('Сохранение финального чекпоинта модели после обучения')
    model.save_pretrained(config_params['model']['final_checkpoint_save_path'])


if __name__ == "__main__":
    main()