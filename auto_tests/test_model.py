from src.model import create_model
from transformers import PreTrainedTokenizerFast

def test_create_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('./data/tokenizer/')
    
    fake_config = {
        'model': {
            'n_positions': 64,
            'n_embedding': 64,
            'n_layer': 2,
            'n_head': 2
        }
    }

    model = create_model(fake_config, tokenizer)
    
    assert model.config.n_positions == 64
    assert model.config.n_embedding == 64
    assert model.config.n_layer == 2
    assert model.config.n_head == 2
    assert model.config.vocab_size == tokenizer.vocab_size