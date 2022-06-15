import paddle
from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer

model_name = './output/Final_bart.pdparams'

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()


def postprocess_response(seq, bos_idx, eos_idx):
    """Post-process the decoded sequence."""
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1] if idx != bos_idx and idx != eos_idx
    ]
    
    # res = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(seq))
    res = tokenizer.convert_ids_to_string(seq)
    return res


def predcit(inputs):

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id


    input_ids = tokenizer(inputs)["input_ids"]
    input_ids = paddle.to_tensor(input_ids, dtype='int64').unsqueeze(0)

    outputs, _ = model.generate(
        input_ids=input_ids,
        forced_bos_token_id=bos_id,
        decode_strategy="beam_search",
        num_beams=4,
        max_length=50,
        use_faster=True)
    result = postprocess_response(outputs[0].numpy().tolist(), bos_id, eos_id)

    return result

inputs = '''基于Transformer的预训练模型已经在各种下游任务中成为标配，但是在文本生成任务中，效果缺欠佳，BART的出现解决了这一瓶颈。基于BART预训练模型的自动摘要生成，以数据集Abstract\_cs和LCSTS为数据源，试验结果表明，该方法可以提高生成准确率，在小数据集上拟合效果良好，可以应用于小规模单领域的自动文本摘要任务。'''

result = predcit(inputs)

print("Model input:", inputs)
print("Result:", result)