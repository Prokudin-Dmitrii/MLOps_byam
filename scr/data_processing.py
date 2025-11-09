import json
from datatrove.pipeline.readers import ParquetReader


import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import fasttext
import re

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def load_initial_data(config_params):
    data_source = config_params['data_processing']['source']
    data_size = config_params['data_processing']['size']
    data_savefile_path = config_params['data_processing']['savefile_path']

    data_reader = ParquetReader(data_source)

    total_size = 0
    
    with open(data_savefile_path, 'w', encoding='utf-8') as data_file:
        for document in data_reader():
            
            text = document.text.strip()
            
            if text is None:
                continue
            
            data_sample = json.dumps({'text': text}, ensure_ascii=False) + '\n'
            sample_size = len(data_sample.encode('utf-8'))
            
            if total_size + sample_size > data_size * 1024 * 1024:
                break
            
            data_file.write(data_sample)
            total_size += sample_size


def clean_sample(sample: str):
    sample = re.sub(r'[\U00010000-\U0010ffff]', '', sample)
    sample = re.sub('<[^>]+>', ' ', sample)
    sample = re.sub(r'http\S+|www\.\S+', ' ', sample)
    sample = re.sub(r'\s+', ' ', sample).strip()

    return sample


def process_data(config_params):
    language_checker_path = config_params['data_processing']['language_checker_path']
    data_savefile_path = config_params['data_processing']['savefile_path']
    data_clean_savefile_path = config_params['data_processing']['clean_savefile_path']
    data_language = config_params['data_processing']['language']
    data_language_confidence = config_params['data_processing']['language_confidence']
    
    language_checker = fasttext.load_model(language_checker_path)

    with open(data_savefile_path, 'r', encoding='utf-8') as data_file, open(data_clean_savefile_path, 'w', encoding='utf-8') as clean_data_file:
        for data_sample in data_file:
            sample = json.loads(data_sample)['text']

            sample_language = language_checker.predict(sample.replace('\n', ' '))
            sample_language_prob = sample_language[1].item()
            sample_language = sample_language[0][0].split('__label__')[1]
            
            sample = clean_sample(sample)

            if sample_language == data_language:
                if sample_language_prob < data_language_confidence:
                    continue
                else:
                    clean_data_sample = json.dumps({'text': sample}, ensure_ascii=False) + '\n'
                    clean_data_file.write(clean_data_sample)
            else:
                continue


def create_tokenizer(config_params):
    data_clean_savefile_path = config_params['data_processing']['clean_savefile_path']
    tokenizer_vocabulary_size = config_params['data_processing']['vocabulary_size']
    tokenizer_savefile_path = config_params['data_processing']['tokenizer_savefile_path']
    
    tokenizer_data_samples = []

    with open(data_clean_savefile_path, 'r', encoding='utf-8') as clean_data_file:
        for data_sample in clean_data_file:
            sample = json.loads(data_sample)['text']
            tokenizer_data_samples.append(sample)
    

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(vocab_size=tokenizer_vocabulary_size, special_tokens=["[PAD]", "[UNK]", "[EOS]"], show_progress=True)

    tokenizer.train_from_iterator(tokenizer_data_samples, trainer=trainer, length=len(tokenizer_data_samples))

    tokenizer.post_processor = TemplateProcessing(
        single="$A [EOS]",
        special_tokens=[
            ("[EOS]", tokenizer.token_to_id("[EOS]"))
        ]
    )

    tokenizer.save(tokenizer_savefile_path)
    
    return tokenizer


def samples_blocks_division(data_sample, tokenizer, block_size=256):
    blocks = []
    for sample in data_sample['text']:
        ids = tokenizer.encode(sample).ids

        for i in range(0, len(ids), block_size // 2):
            blocks.append(ids[i: i + block_size])
    
    return {'input_ids': blocks}


def data_collator(batch, tokenizer):
    input_ids = []
    for sample in batch:
        input_ids.append(torch.tensor(sample['input_ids']))
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return input_ids, attention_mask


def create_dataloader(config_params):
    data_clean_savefile_path = config_params['data_processing']['clean_savefile_path']
    context_size = config_params['data_processing']['context_size']
    tokenizer_savefile_path = config_params['data_processing']['tokenizer_savefile_path']
    data_batch_size = config_params['data_processing']['batch_size']

    tokenizer = create_tokenizer(config_params)

    dataset = load_dataset('json', data_files=data_clean_savefile_path)['train']
    blocked_tokenized_dataset = dataset.map(lambda sample: samples_blocks_division(sample, tokenizer, block_size=context_size), batched=True, remove_columns=['text'])

    wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_savefile_path, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    dataloader = DataLoader(
        blocked_tokenized_dataset,
        batch_size=data_batch_size,
        shuffle=True,
        collate_fn=lambda batch: data_collator(batch, wrapped_tokenizer),
    )

    return dataloader