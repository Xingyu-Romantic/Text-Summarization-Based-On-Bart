import os


import sys
sys.path.append(os.getcwd())
from util.test_utils import generate
from util.common_utils import function_with_args_and_default_kwargs

def run():
    config = {
        'model_name_or_path': './output/Final_bart.pdparams',
        'dataset_name': 'csl_title_public',
        'output_path': './generate.txt',
        'max_source_length': 1024,
        'max_target_length': 142,
        'min_target_length': 0,
        'decode_strategy': 'greedy_search',
        'top_k': 2,
        'top_p': 1.0,
        'num_beams': 1,
        'length_penalty': 0.6,
        'early_stopping': False,
        'diversity_rate': 0.0,
        'faster': True,
        'use_fp16_decoding': False,
        'batch_size': 12,
        'seed': 2022,
        'device': 'GPU',
        'ignore_pad_token_for_loss': True,
        'logging_steps': 100
    }

    args  = function_with_args_and_default_kwargs(config)

    generate(args)

if __name__ == '__main__':
    run()
