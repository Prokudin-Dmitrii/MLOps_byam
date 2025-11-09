from transformers import (
    GPT2Config,
    GPT2LMHeadModel
)


def create_model(config_params, tokenizer):
    n_positions = config_params['model']['n_positions']
    n_embedding = config_params['model']['n_embedding']
    n_layer = config_params['model']['n_layer']
    n_head = config_params['model']['n_head']

    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions,
        n_embd=n_embedding,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        loss_type='ForCausalLMLoss'
    )

    model = GPT2LMHeadModel(model_config)

    return model