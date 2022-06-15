
import wandb

import os
import sys
sys.path.append(os.getcwd())
from util.train_utils import getDataset, do_train
from util.common_utils import set_seed, function_with_args_and_default_kwargs



def run():
    wandb.init(project="Transformer-Summary", entity="involute")

    
    config = {
        'seed': 2022,
        'pretrained_model': './output/Final_bart.pdparams',
        'dataset_name': 'lcsts_new',  # lcsts_new, csl_title_public
        'train_batch_size': 18,
        'eval_batch_size': 16,
        'save_steps':3000,
        'logging_steps':200,
        'max_source_length': 512,
        'max_target_length': 128,
        'min_target_length': 0,
        'max_steps': -1,
        'warmup_proportion': 0.1,
        'warmup_steps': 0,
        'num_train_epochs': 30,
        'learning_rate': 1e-5,
        'adam_epsilon': 1e-6,
        'weight_decay': 0.01,
        'use_amp': False,
        'scale_loss': 2**15,
        'output_dir': './output',
        'device': 'GPU',
        'ignore_pad_token_for_loss': True
    }

    wandb.config = config
    args  = function_with_args_and_default_kwargs(config)

    do_train(args, wandb)



    

if __name__ == '__main__':

    run()
    
    


    