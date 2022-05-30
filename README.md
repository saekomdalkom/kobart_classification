# KoBART for multi-class classification using Korean case

## Requirements

```
pip install -r requirements.txt
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
```

<br/>

## Usage


`mybartcls.ipynb`를 참조하세요.

<br/>


## Description

- datamodule: 모델의 input과 output으로 들어가는 데이터를 전처리합니다. 

    - input: case text
    - output: index from 0 to 5 (0:'민사', 1:'형사', 2:'조세', 3:'가사', 4:'일반행정', 5:'특허')
    
    <br/>

- lightningbase: pytorch lightning의 lightningmodule을 상속해서 optimizer와 save_model을 정의합니다. 

- my_model: lightningbase를 상속해서 BARTcls의 특화된 부분을 정의합니다. 

- train: 실제로 실행하는 main 부분이고 hyper parameter들을 정의합니다. 