import torch
import logging
from src.training import model_training
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel

class FakeLoader:
    def __len__(self): 
        return 4
    
    def __iter__(self):
        yield torch.ones((1, 5), dtype=torch.long), torch.ones((1, 5), dtype=torch.long)
        yield torch.ones((1, 5), dtype=torch.long), torch.ones((1, 5), dtype=torch.long)
        yield torch.ones((1, 5), dtype=torch.long), torch.ones((1, 5), dtype=torch.long)
        yield torch.ones((1, 3), dtype=torch.long), torch.zeros((1, 3), dtype=torch.long)

def test_model_training_runs(tmp_path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained('data/tokenizer/')
    model = GPT2LMHeadModel.from_pretrained('model/byam_step_final/')

    fake_config_params = {
        'training': {
            'n_epochs': 1,
            'device': 'cpu',
            'learning_rate': 1e-5,
            'weight_decay': 1e-3,
            'warmup_fraction': 0.1,
            'few_shot_steps': 1000,
            'save_steps': 1000,
            'logger_save_steps': 1000,
            'few_shots_examples': ['Studenti '],
            'few_shots_max_length': 10,
            'few_shots_sample': False,
            'few_shots_top_p': 0.9,
            'few_shot_temperature': 0.9,
            'few_shot_repetition_penalty': 1.0,
            'save_folder': str(tmp_path)
        }
    }

    logger = logging.getLogger('test')
    
    train_dataloader = FakeLoader()
    
    model, losses = model_training(fake_config_params, logger, model, tokenizer, train_dataloader)
    
    assert isinstance(losses, list)
    assert len(losses) == fake_config_params['training']['n_epochs']