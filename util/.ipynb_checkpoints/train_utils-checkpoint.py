import os
import paddle
import time
import random
import argparse
import nltk
from rouge_score import rouge_scorer, scoring

import numpy as np
import pandas as pd

from tqdm import tqdm

from paddlenlp.datasets import load_dataset
from paddlenlp.utils.log import logger
from util.csl_title_public import getDataset
from util.common_utils import set_seed, compute_metrics
import pprint

import paddlenlp
from paddle import nn
from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup





@paddle.no_grad()
def evaluate(model, data_loader, tokenizer, ignore_pad_token_for_loss,
             min_target_length, max_target_length, wandb):
    model.eval()
    all_preds = []
    all_labels = []
    model = model._layers if isinstance(model, paddle.DataParallel) else model
    for batch in tqdm(data_loader, total=len(data_loader), desc="Eval step"):
        input_ids, _, _, labels = batch
        preds = model.generate(
            input_ids=input_ids,
            min_length=min_target_length,
            max_length=max_target_length,
            use_cache=True)[0]
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
    rouge_result, _ = compute_metrics(
        all_preds, all_labels, tokenizer, ignore_pad_token_for_loss)
    logger.info(rouge_result)
    wandb.log(rouge_result)
    model.train()



    

def do_train(args, wandb):
    pprint.pprint(args)
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    model = BartForConditionalGeneration.from_pretrained(
        args.pretrained_model)


    tokenizer = BartTokenizer.from_pretrained(args.pretrained_model)
    model.save_pretrained('./tokenizer')
    train_data_loader, dev_data_loader = getDataset(args, model, tokenizer)


    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)
    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    # decay_params = [
    #     p.name for n, p in model.named_parameters()
    #     if not any(nd in n for nd in ["bias", "norm"])
    # ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,)
        # apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = nn.CrossEntropyLoss()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    global_step = 0
    tic_train = time.time()
    for epoch in tqdm(range(args.num_train_epochs), desc="Epoch"):
        for step, batch in tqdm(
                enumerate(train_data_loader),
                desc="Train step",
                total=len(train_data_loader)):
            global_step += 1
            input_ids, attention_mask, decoder_input_ids, labels = batch
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = model(input_ids, attention_mask, decoder_input_ids)
                loss = loss_fct(logits, labels)
            if args.use_amp:
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.minimize(optimizer, scaled_loss)
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.logging_steps == 0:
                wandb.log({'epoch': epoch, 'step': step, 'loss': loss.item()}, commit = False)
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                evaluate(model, dev_data_loader, tokenizer,
                         args.ignore_pad_token_for_loss, args.min_target_length,
                         args.max_target_length,wandb)

                logger.info("eval done total : %s s" % (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(
                        args.output_dir, "bart_model_%d.pdparams" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            if global_step >= num_training_steps:
                return
    if paddle.distributed.get_rank() == 0:
        output_dir = os.path.join(args.output_dir,
                                  "bart_model_final_%d.pdparams" % global_step)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Need better way to get inner model of DataParallel
        model_to_save = model._layers if isinstance(
            model, paddle.DataParallel) else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)