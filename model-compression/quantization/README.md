
## 簡介

- https://docs.nvidia.com/deeplearning/tensorrt/tensorflow-quantization-toolkit/docs/docs/qat.html
- Quantization Aware Training (QAT)
- https://github.com/HuangOwen/Awesome-LLM-Compression
- https://blog.csdn.net/jinzhuojun/article/details/106955059
- 閒話模型壓縮之量化（Quantization）篇


一文總結當下常用的大型 transformer 效率最佳化方案
- https://zhuanlan.zhihu.com/p/604118644
- https://lilianweng.github.io/posts/2023-01-10-inference-optimization/


Introduction to Weight Quantization：Reducing the size of Large Language Models with 8-bit quantization
- https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c


大規模 Transformer 模型 8 位元矩陣乘簡介 - 基於 Hugging Face Transformers、Accelerate 以及 bitsandbytes
- https://huggingface.co/blog/zh/hf-bitsandbytes-integration






- 大模型量化原理綜述（一）：概述
- 大模型量化原理綜述（二）：訓練感知量化 LLM-QAT
- 大模型量化原理綜述（二）：SmoothQuant
- 大模型量化原理綜述（二）：AWQ
- 大模型量化原理綜述（二）：GPTQ
- 大模型量化原理綜述（三）：LLM.int8()



---


在深度神經網路上應用量化策略有兩種常見的方法：

- 訓練後量化（PTQ）：首先需要模型訓練至收斂，然後將其權重的精度降低。與訓練過程相比，量化操作起來往往代價小得多；
- 量化感知訓練 (QAT)：在預訓練或進一步微調期間應用量化。QAT 能夠獲得更好的效能，但需要額外的計算資源，還需要使用具有代表性的訓練資料。


值得注意的是，理論上的最優量化策略與實際在硬體核心上的表現存在著客觀的差距。**由於 GPU 核心對某些型別的矩陣乘法（例如 INT4 x FP16）缺乏支援，並非下面所有的方法都會加速實際的推理過程**。





## Post Training Quantization(PTQ)

- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
  - https://www.deepspeed.ai/tutorials/model-compression/
  - 整合在Deepspeed
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
  - https://github.com/mit-han-lab/smoothquant
  - 已經整合在[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
  - https://github.com/mit-han-lab/llm-awq
- LLM.int8()
  - LLM.int8()——在大模型上使用int8量化:https://zhuanlan.zhihu.com/p/586406082
- OWQ: Lessons learned from activation outliers for weight quantization in large language models
	- https://github.com/xvyaward/owq
- Spqr: A sparse-quantized representation for near-lossless LLM weight compression
	- https://arxiv.org/pdf/2306.03078.pdf
	- Tim Dettmers
- LLM.int8()
	- Tim Dettmers
- RPTQ
	- https://github.com/hahnyuan/RPTQ4LLM
- OliVe
- Outlier Suppression+
	- https://github.com/wimh966/outlier_suppression




## Quantization Aware Training（QAT）

- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
  - https://github.com/facebookresearch/LLM-QAT













