import paddle
import nltk
from paddlenlp.transformers import BartForConditionalGeneration, BertTokenizer

def post_process_seq(seq,
                        bos_idx = 101,
                        eos_idx = 102,
                        output_bos=False,
                        output_eos=False):
    """
    Post-process the decoded sequence.
    """
    seq = seq[0]
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


tokenizer = BertTokenizer.from_pretrained('./bart-base-chinese')
model = BartForConditionalGeneration.from_pretrained('./bart-base-chinese')
inputs = tokenizer("我 [mask] 中国人")
preds = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids']))
print(inputs, preds)
inputs = paddle.to_tensor([inputs['input_ids']])
# print(inputs['input_ids'])
# outputs = model(paddle.to_tensor([inputs['input_ids']]))
preds = model.generate(
            input_ids=inputs,
            min_length=0,
            max_length=120,
            use_cache=True)[0]

pred_id = post_process_seq(preds.numpy())
preds = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(pred_id))
print(preds)
# print(outputs)