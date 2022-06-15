import time
import nltk
import paddle
from paddle.io import BatchSampler, DataLoader
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Tuple, Stack
from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer


from util.common_utils import set_seed, compute_metrics
from util.csl_title_public import getDataset

@paddle.no_grad()
def generate(args):
    paddle.set_device(args.device)
    set_seed(args)
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path)
    data_loader = getDataset(args, model, tokenizer, is_test=True)
    data_loader.pin_memory = False

    model.eval()
    total_time = 0.0
    start_time = time.time()
    all_preds = []
    all_labels = []
    for step, batch in enumerate(data_loader):
        input_ids, _, mem_seq_lens, _, labels = batch
        preds, _ = model.generate(
            input_ids=input_ids,
            seq_lens=mem_seq_lens,
            max_length=args.max_target_length,
            min_length=args.min_target_length,
            decode_strategy=args.decode_strategy,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            diversity_rate=args.diversity_rate,
            use_faster=args.faster)
        total_time += (time.time() - start_time)
        if step % args.logging_steps == 0:
            print('step %d - %.3fs/step' %
                  (step, total_time / args.logging_steps))
            total_time = 0.0
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        start_time = time.time()

    rouge_result, decoded_preds = compute_metrics(
        all_preds, all_labels, tokenizer, args.ignore_pad_token_for_loss)
    print("Rouge result: ", rouge_result)
    with open(args.output_path, 'w', encoding='utf-8') as fout:
        for decoded_pred in decoded_preds:
            fout.write(decoded_pred + '\n')
    print('Save generated result into: %s' % args.output_path)