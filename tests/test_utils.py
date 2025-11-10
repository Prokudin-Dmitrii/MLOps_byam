import torch
import numpy as np

from src.utils import manual_seed

def test_manual_seed():
    fake_config_params = {
        'utils': {
            'seed': 451
            }
        }
    
    manual_seed(fake_config_params)
    torch_before = torch.rand(10)
    numpy_before = np.random.rand(10)
    
    manual_seed(fake_config_params)
    torch_after = torch.rand(10)
    numpy_after = np.random.rand(10)
    
    assert torch.allclose(torch_before, torch_after)
    assert np.allclose(numpy_before, numpy_after)
