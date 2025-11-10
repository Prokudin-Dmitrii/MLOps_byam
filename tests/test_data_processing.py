import pytest
import numpy as np
from unittest.mock import MagicMock
import json

from transformers import PreTrainedTokenizerFast

from src.data_processing import load_initial_data, process_data, samples_blocks_division, data_collator

@pytest.fixture
def mock_fasttext(monkeypatch):
    mock_model = MagicMock()
    mock_model.predict.return_value = (['__label__cs'], np.array([0.94]))
    monkeypatch.setattr('fasttext.load_model', lambda path: mock_model)
    
    return mock_model

class FakeDoc:
    def __init__(self, text): 
        self.text = text

@pytest.fixture
def mock_parquet_reader(monkeypatch):
    def fake_reader():
        yield FakeDoc('První falešný text v Češtině')
        yield FakeDoc('Druhý falešný text v Češtině')
        yield FakeDoc('Třetí fake text v Češtině')

    monkeypatch.setattr('src.data_processing.ParquetReader', lambda path: fake_reader)
    return fake_reader


def test_load_initial_data(tmp_path, mock_parquet_reader): # Фикстуру подаём, чтобы она отработала и заменила паркет ридер
    fake_config_params = {
        'data_processing': {
            'source': 'whocareswemockedtheparquetreaderanywaaaay',
            'size': 1,
            'savefile_path': tmp_path / 'test_out.jsonl'
        }
    }

    load_initial_data(fake_config_params)
    text = (tmp_path / 'test_out.jsonl').read_text(encoding='utf-8')
    
    assert 'falešný text v Češtině' in text


def test_process_data(tmp_path, mock_fasttext): # Аналогично 
    input_file = tmp_path / 'initial_data.jsonl'
    output_file = tmp_path / 'clean_data.jsonl'
    
    input_file.write_text(json.dumps({'text': 'Nejlepší na světě není zatím fakulta VUT'}) + '\n', encoding='utf-8')

    fake_config_params = {
        'data_processing': {
            'language_checker_path': 'whocareswemockedthelanguagecheckeranywaaaay',
            'savefile_path': input_file,
            'clean_savefile_path': output_file,
            'language': 'cs',
            'language_confidence': 0.9
        }
    }

    process_data(fake_config_params)
    
    assert output_file.exists()
    
    output_content = output_file.read_text(encoding='utf-8')
    assert 'na světě není zatím' in output_content


def test_samples_blocks_division():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('data/tokenizer/')
    
    fake_data = {
        'text': ['Lenin jde do baru!']
        }
    
    result = samples_blocks_division(fake_data, tokenizer, block_size=4)

    assert isinstance(result['input_ids'], list)
    assert len(result['input_ids']) > 0


def test_data_collator():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('data/tokenizer/')
    
    fake_batch = [
        {'input_ids': [124, 2151, 3421]}, 
        {'input_ids': [532, 214]},
    ]
    
    input_ids, attention_mask = data_collator(fake_batch, tokenizer)
    assert input_ids.shape == attention_mask.shape
    assert input_ids.ndim == 2
    assert input_ids.shape[0] == 2
    assert input_ids.shape[1] == 3