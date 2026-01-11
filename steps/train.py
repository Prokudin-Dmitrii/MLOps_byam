import os

import argparse

from transformers import PreTrainedTokenizerFast
import torch
import mlflow
import mlflow.transformers

import yaml

from src.data_processing import create_dataloader
from src.model import create_model
from src.utils import logger_configure, manual_seed, load_config_params
from src.training import model_training

def get_dvc_hash_from_lock(path: str) -> str:
    with open('dvc.lock', 'r') as f:
        lock = yaml.safe_load(f)

    for stage in lock['stages'].values():
        for out in stage.get('outs', []):
            if out['path'] == path:
                return out[out['hash']]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_params_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, required=False, default=0, help='Включает логгирование в поток вывода (командную строку)')

    args = parser.parse_args()

    yaml_config_path = args.config_params_path
    verbose = args.verbose

    config_params = load_config_params(yaml_config_path)
    logger = logger_configure(config_params, (verbose > 0))

    mlflow.set_experiment('BYAM_training')
    mlflow.transformers.autolog()

    with mlflow.start_run():
        mlflow.log_params(config_params['model'])

        mlflow.set_tag(
            'dvc_data_hash',
            get_dvc_hash_from_lock('data/clean_czech_data.jsonl')
        )

        set_seed = config_params['set_fixed_seed']
        if set_seed:
            logger.info('Фиксация random_seed')
            manual_seed(config_params)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(config_params['data_processing']['wrapped_tokenozer_savefile_path'])
        
        logger.info('Создание dataloader\'a')
        train_dataloader = create_dataloader(config_params, tokenizer)

        logger.info('Создание модели')
        model = create_model(config_params, tokenizer)

        model, epoch_losses = model_training(config_params, logger, model, tokenizer, train_dataloader)

        for epoch, loss in enumerate(epoch_losses):
            mlflow.log_metric('train_loss', loss, step=epoch)

        logger.info('Сохранение финального чекпоинта модели после обучения')
        model.save_pretrained(config_params['model']['final_checkpoint_save_path'])

        torch.save(model.state_dict(), './model/model.pt')

        mlflow.log_artifacts(config_params['model']['final_checkpoint_save_path'], artifact_path='model_hf')
        mlflow.log_artifact('./model/model.pt')

        if os.path.exists('dvc.lock'):
            mlflow.log_artifact('dvc.lock')

        if os.path.exists('dvc.yaml'):
            mlflow.log_artifact('dvc.yaml')

        logger.info("MLflow run завершён")


if __name__ == "__main__":
    main()