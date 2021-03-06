# 自动文本摘要生成 ![](https://img.shields.io/badge/license-Apache--2.0-blue) ![](https://img.shields.io/badge/PaddlePaddle-v2.3.0-blue) ![](https://img.shields.io/badge/PaddleNLP-v2.0.0-blue) ![](https://img.shields.io/badge/Flask-v1.1.2-white) ![](https://img.shields.io/badge/BootStrap-v3.3.7-white) :tada:

> 2019 人工智能 芦星宇


<p align="center">
   <img src="https://readme-typing-svg.herokuapp.com?lines=%E5%9F%BA%E4%BA%8EBart%E9%A2%84%E8%AE%AD%E7%BB%83%E7%9A%84%E8%87%AA%E5%8A%A8%E6%96%87%E6%9C%AC%E6%91%98%E8%A6%81%E7%94%9F%E6%88%90" alt="typing-svg">
</p>


### File Structure :art:
```
├── LICENSE
├── README.md
├── dataset
│   └── csl_title_public
│       ├── csl_title_dev.json
│       ├── csl_title_predict.json
│       ├── csl_title_test.json
│       └── csl_title_train.json
├── deploy
│   ├── demo.html
│   ├── deploy.py
│   ├── predict.py
│   └── static
│       ├── css
│       │   ├── base.css
│       │   ├── bootstrap-theme.min.css
│       │   └── bootstrap.min.css
│       └── templates
│           ├── common
│           │   └── base.html
│           └── index.html
├── generate.txt
├── output
│   ├── Final_bart.pdparams
│   │   ├── merges.txt
│   │   ├── model_config.json
│   │   ├── model_state.pdparams
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   └── LCSTS.pdparams
│       ├── merges.txt
│       ├── model_config.json
│       ├── model_state.pdparams
│       ├── tokenizer_config.json
│       └── vocab.json
├── requirement.txt
├── task
│   ├── predict.py
│   ├── test.py
│   ├── train.py
│   └── unit_test.py
└── util
    ├── __init__.py
    ├── common_utils.py
    ├── csl_title_public.py
    ├── test_utils.py
    └── train_utils.py
```

## Download Weight

Weight File: [Final_bart.pdparams LCSTS.pdparams](https://drive.google.com/drive/folders/1zZNn5mOi8qP1SOuSK_Q9ZuEWxK4GXRGD?usp=sharing)


## Installation :beers:
```bash
pip install -r requirement.txt
```

## Deploy(Show on Web Site) :rocket:
```bash
python deploy/deploy.py

Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
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







