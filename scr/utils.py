import os

import logging
import numpy as np
import torch
import random

def logger_configure(config_params, verbose=False):
    logger_file = config_params['utils']['logger_output_file']

    logger_handlers = [logging.FileHandler(logger_file, mode='w', encoding='utf-8')]
    if verbose:
        logger_handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=logger_handlers
    )

    logger = logging.getLogger(__name__)
    
    return logger


def manual_seed(config_params):
    fix_seed = config_params['utils']['seed']
    
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    
    os.environ['PYTHONHASHSEED'] = str(fix_seed)

    # А можно ещё шаманить с детерминированностью куды, но это уже страшные материи