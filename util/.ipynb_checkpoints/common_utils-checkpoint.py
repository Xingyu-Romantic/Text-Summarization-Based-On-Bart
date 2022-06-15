import nltk
import random
import argparse

import paddle
import numpy as np

from rouge_score import rouge_scorer, scoring


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)

def function_with_args_and_default_kwargs(config):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in config.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args(None)
    return args




    
def compute_metrics(preds, labels, tokenizer, ignore_pad_token_for_loss=True):
    def compute_rouge(predictions,
                      references,
                      rouge_types=None,
                      use_stemmer=True):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeLsum"]

        scorer = rouge_scorer.RougeScorer(
            rouge_types=rouge_types, use_stemmer=use_stemmer)
        aggregator = scoring.BootstrapAggregator()

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            aggregator.add_scores(score)
        result = aggregator.aggregate()
        result = {
            key: round(value.mid.fmeasure * 100, 4)
            for key, value in result.items()
        }
        return result
    def post_process_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def post_process_seq(seq,
                         bos_idx,
                         eos_idx,
                         output_bos=False,
                         output_eos=False):
        """
        Post-process the decoded sequence.
        """
        eos_pos = len(seq) - 1
        for i, idx in enumerate(seq):
            if idx == eos_idx:
                eos_pos = i
                break
        seq = [
            idx for idx in seq[:eos_pos + 1]
            if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
        ]

        return seq
    if ignore_pad_token_for_loss:
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds, decoded_labels = [], []
    # tokenizer.bos_token_id, tokenizer.eos_token_id
    for pred, label in zip(preds, labels):
        pred_id = post_process_seq(pred, tokenizer.bos_token_id, tokenizer.eos_token_id)
        label_id = post_process_seq(label, tokenizer.bos_token_id, tokenizer.eos_token_id)
        # print(pred, label)
        # pred_token = tokenizer.convert_ids_to_tokens(pred_id)
        # label_token = tokenizer.convert_ids_to_tokens(label_id)
        # decoded_preds.append(pred_token)
        # decoded_labels.append(label_token)
        # decoded_preds.append(tokenizer.convert_tokens_to_string(pred_token))
        # decoded_labels.append(tokenizer.convert_tokens_to_string(label_token))
        decoded_preds.append(tokenizer.convert_ids_to_string(pred_id))
        decoded_labels.append(tokenizer.convert_ids_to_string(label_id))
    decoded_preds, decoded_labels = post_process_text(decoded_preds,
                                                      decoded_labels)
    rouge_result = compute_rouge(decoded_preds, decoded_labels)
    print(decoded_preds)
    return rouge_result, decoded_preds