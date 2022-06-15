# 自动文本摘要生成 ![](https://img.shields.io/badge/license-Apache--2.0-blue) ![](https://img.shields.io/badge/PaddlePaddle-v2.3.0-blue) ![](https://img.shields.io/badge/PaddleNLP-v2.0.0-blue)
> 2019 人工智能 芦星宇

[![Typing SVG](https://readme-typing-svg.herokuapp.com?lines=%E5%9F%BA%E4%BA%8EBart%E9%A2%84%E8%AE%AD%E7%BB%83%E7%9A%84%E8%87%AA%E5%8A%A8%E6%96%87%E6%9C%AC%E6%91%98%E8%A6%81%E7%94%9F%E6%88%90)](https://git.io/typing-svg)


## Installation
```bash
pip install -r requirement.txt
```



## Train
**Note**: Introducing [wandb](https://wandb.ai/home) as training visualization tool.


### Fine-tuning
```bash
wandb login # Wandb Token
python task/train.py
```

### Test
```bash
python task/test.py
```

### Predict
```bash
python task/predict.py
```





