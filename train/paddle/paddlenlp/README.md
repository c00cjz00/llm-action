


## PaddleNLP

- https://hub.docker.com/r/paddlecloud/paddlenlp
- https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm


- pytorch轉paddle: https://github.com/PaddlePaddle/PaddleNLP/blob/v2.6.1/docs/community/contribute_models/convert_pytorch_to_paddle.rst



```
pip install --upgrade paddlenlp==2.6.1 -i https://pypi.org/simple



sudo pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```


```
docker run --name dev \
--runtime=nvidia \
-v $PWD:/mnt \
-p 8888:8888 \
-it \
paddlecloud/paddlenlp:develop-gpu-cuda10.2-cudnn7-cdd682 \
/bin/bash
```





---












## 支援的模型

```
bigscience/bloom-560m
bigscience/bloomz-560m/
```


### bloom

```
https://bj.bcebos.com/paddlenlp/models/community/bigscience/bloomz-560m/tokenizer_config.json

```



```
from paddlenlp.transformers.bloom.tokenizer import BloomTokenizer
tokenizer = BloomTokenizer.from_pretrained("bigscience/bloomz-560m")
```





## 推理

```
import paddlenlp
from pprint import pprint
from paddlenlp import Taskflow
schema = ['時間', '選手', '賽事名稱'] # Define the schema for entity extraction
ie = Taskflow('information_extraction', schema=schema)
pprint(ie("2月8日上午北京冬奧會自由式滑雪女子大跳臺決賽中中國選手谷愛凌以188.25分獲得金牌！"))
```



```
https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.1/model_state.pdparams

/Users/liguodong/.paddlenlp/taskflow/information_extraction/uie-base/model_state.pdparams

```

---


```
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", dtype="float32")
input_features = tokenizer("你好！請自我介紹一下。", return_tensors="pd")
outputs = model.generate(**input_features, max_length=128)
tokenizer.batch_decode(outputs[0])




tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", from_hf_hub=True)


tokenizer = AutoTokenizer.from_pretrained("ziqingyang/chinese-llama-7b", from_hf_hub=True)





model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", dtype="float32",from_aistudio = False, from_hf_hub=True, convert_from_torch=True)


tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", from_hf_hub=True)

```


### 動態圖推理

```
# 預訓練&SFT動態圖模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16" \
    --mode "dynamic"
```


### 靜態圖推理



```
# 首先需要執行一下命令將動態圖匯出為靜態圖
# LoRA需要先合併引數，詳見3.7LoRA引數合併
# Prefix Tuning暫不支援
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --output_path ./inference \
    --dtype float16


# 靜態圖模型推理
python predictor.py \
    --model_name_or_path inference \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16" \
    --mode "static"
```


## Flask & Gradio UI服務化部署

```
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" flask_server.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --port 8010 \
    --flask_port 8011 \
    --src_length 1024 \
    --dtype "float16"
```




## 量化


量化演算法可以將模型權重和啟用轉為更低位元數值型別表示，能夠有效減少視訊記憶體佔用和計算開銷。

下面我們提供GPTQ和PaddleSlim自研的PTQ策略，分別實現WINT4和W8A8量化。

- https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md


```


https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

python -m pip install paddlepaddle-gpu==0.0.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html


安裝釋出版本：

pip install paddleslim
安裝develop版本：

git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install

驗證安裝：安裝完成後您可以使用 python 或 python3 進入 python 直譯器，輸入import paddleslim, 沒有報錯則說明安裝成功。
```




```
# PTQ 量化
python  finetune_generation.py ./llama/ptq_argument.json

# GPTQ 量化
python  finetune_generation.py ./llama/gptq_argument.json

```


