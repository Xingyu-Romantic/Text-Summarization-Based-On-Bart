import os
import json

from functools import partial
from tqdm import tqdm

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Tuple, Stack, Pad

from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader

ROOT_DIR = './dataset/csl_title_public/'

Dataset_Key_Value = {'csl_title_public': ['abst', 'title'],
                     'lcsts_new': ['source', 'target']}

def read_json(input_file: str):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    return list(map(json.loads, tqdm(lines, desc='Reading...')))

def read(data_path):
    data = read_json(data_path)
    for i in data:
        abst = i['abst']
        title = i['title']
        yield {'abst': abst, 'title': title}


def convert_example(example,
                    text_column,
                    summary_column,
                    tokenizer,
                    decoder_start_token_id,
                    max_source_length,
                    max_target_length,
                    ignore_pad_token_for_loss=True,
                    is_train=True):
    """
    Convert a example into necessary features.
    """
    inputs = example[text_column]
    targets = example[summary_column]
    labels = tokenizer(
        targets,
        max_seq_len=max_target_length,
        pad_to_max_seq_len=True)
    decoder_input_ids = [decoder_start_token_id] + labels["input_ids"][:-1]
    if ignore_pad_token_for_loss:
        labels["input_ids"] = [(l if l != tokenizer.pad_token_id else -100)
                               for l in labels["input_ids"]]
    if is_train:
        model_inputs = tokenizer(
            inputs,
            max_seq_len=max_source_length,
            pad_to_max_seq_len = True,
            return_attention_mask=True,
            return_length=False)
        return model_inputs["input_ids"], model_inputs[
            "attention_mask"], decoder_input_ids, labels["input_ids"]
    else:
        model_inputs = tokenizer(
            inputs,
            max_seq_len=max_source_length,
            pad_to_max_seq_len = True,
            return_attention_mask=True,
            return_length=True)
        return model_inputs["input_ids"], model_inputs["attention_mask"], \
        model_inputs["seq_len"], decoder_input_ids, labels["input_ids"]


def getDataset(args, model, tokenizer, is_test = False):

    

    if args.dataset_name == 'csl_title_public':
        train = load_dataset(read, data_path=os.path.join(ROOT_DIR, 'csl_title_train.json'), lazy=False)
        val = load_dataset(read, data_path=os.path.join(ROOT_DIR, 'csl_title_dev.json'), lazy=False)
        # test = load_dataset(read, data_path=os.path.join(ROOT_DIR, 'csl_title_test.json'), lazy=True)
    elif args.dataset_name == 'lcsts_new':
        train, val = load_dataset('lcsts_new', splits=["train", "dev"])

    # decoder_start_token_id = model.gpt.decoder_start_token_id,
    model.save_pretrained('./model')
    trans_func = partial(convert_example, 
                            text_column = Dataset_Key_Value[args.dataset_name][0], 
                            summary_column = Dataset_Key_Value[args.dataset_name][1], 
                            tokenizer=tokenizer,
                            decoder_start_token_id = model.eos_token_id,
                            max_source_length = args.max_source_length,
                            max_target_length = args.max_target_length,
                            is_train=True)

    trans_func_test = partial(convert_example, 
                            text_column = Dataset_Key_Value[args.dataset_name][0], 
                            summary_column = Dataset_Key_Value[args.dataset_name][1], 
                            tokenizer=tokenizer, 
                            decoder_start_token_id = model.eos_token_id,
                            max_source_length = args.max_source_length,
                            max_target_length = args.max_target_length,
                            is_train=False)    
    
    batchify_fn_train = lambda samples, fn=Tuple(
            Pad(0),
            Pad(0),  # attention mask
            Pad(0),  # decoder_input_ids
            Pad(0),  # labels
        ): fn(samples)
    
    batchify_fn_test = lambda samples, fn=Tuple(
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # attention mask
            Stack(dtype="int32"),  # mem_seq_lens
            Stack(dtype="int64"),  # decoder_input_ids
            Stack(dtype="int64"),  # labels
        ): fn(samples)



    
    
    if not is_test:
        train_set = train.map(trans_func)
        dev_set = val.map(trans_func)
        train_batch_sampler = DistributedBatchSampler(
            train_set, batch_size=args.train_batch_size, shuffle=True)

        train_data_loader = DataLoader(
            dataset=train_set,
            batch_sampler=train_batch_sampler,
            num_workers=0,
            collate_fn=batchify_fn_train,
            return_list=True)

        dev_batch_sampler = BatchSampler(
            dev_set, batch_size=args.eval_batch_size, shuffle=False)
        dev_data_loader = DataLoader(
            dataset=dev_set,
            batch_sampler=dev_batch_sampler,
            num_workers=0,
            collate_fn=batchify_fn_train,
            return_list=True)
        return train_data_loader, dev_data_loader
    else:
        dev_set = val.map(trans_func_test)
        dev_batch_sampler = BatchSampler(
            dev_set, batch_size=args.batch_size, shuffle=False)
        dev_data_loader = DataLoader(
            dataset=dev_set,
            batch_sampler=dev_batch_sampler,
            num_workers=0,
            collate_fn=batchify_fn_test,
            return_list=True)
        return dev_data_loader




