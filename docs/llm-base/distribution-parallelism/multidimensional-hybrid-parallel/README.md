- https://huggingface.co/docs/transformers/perf_train_gpu_many
- https://huggingface.co/transformers/v4.12.5/parallelism.html





## 大模型多維混合並行彙總


|    模型          | DP  | TP  | PP  | ZeRO Stage | FSDP（ZeRO Stage 3） | GPUs         |   FP16/BF16        | 訓練框架  |
| ------------ | --- | --- | --- | ---------- | ------------------ | ----------------------- | ------ |  ------ | 
| Bloom-176B   | 8   | 4   | 12  | ZeRO-1     | -                  | 384 張 A100 80GB         | BF16    |  Megatron-DeepSpeed    |
| CodeGeeX-13B | 192 | 8   | -   | ZeRO-2     | -                  | 1,536 張 Ascend 910 32GB | FP16  |   Megatron-LM    |
| GLM-130B     | 24  | 4   | 8   | ZeRO-1     | -                  | 768 張 A100 40G          | FP16 |  Megatron-LM + DeepSpeed    |
| OPT-175B     | 124   | 8   | -   | -          | ✅             | 992 張 80GB A100         | FP16 |   PyTorch + Megatron     |
| Megatron-Turing NLG-530B | 16 | 8   | 35  |  -    | -                  | 4480 張 A100 80G | BF16 |  Megatron-LM  + DeepSpeed     |
| GPT-NeoX-20B | 12 | 2   | 4  |ZeRO-1    | -                  | 96 張 A100 40G |   FP16  |   PyTorch v1.10.0 + NCCL 2.10.3 + CUDA 11.1 + Megatron-LM + DeepSpeed   |







### Bloom-176B

- https://huggingface.co/bigscience/bloom

- https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/tr11-176B-ml.slurm

- https://github.com/bigscience-workshop/bigscience/



```
Hardware:

- 384 A100 80GB GPUs (48 nodes)
  
- Additional 32 A100 80GB GPUs (4 nodes) in reserve

- 8 GPUs per node Using NVLink 4 inter-gpu connects, 4 OmniPath links
  
- CPU: AMD
  
- CPU memory: 512GB per node
  
- GPU memory: 640GB per node
  
- Inter-node connect: Omni-Path Architecture (OPA)
  
- NCCL-communications network: a fully dedicated subnet
  
- Disc IO network: shared network with other types of nodes
  

model:

vocabulary size: 250,680
Total seen tokens: **366B**

Bf16 weights: 329GB
Full checkpoint with optimizer states: 2.3TB



MICRO_BATCH_SIZE=2  # was MBS=1 till GBS=784
GLOBAL_BATCH_SIZE=2048  # 4.2M tokens. It is larger than the initial plan of 3.2M tokens to get higher throughput


NHIDDEN=14336
NLAYERS=70
NHEADS=112
SEQ_LEN=2048



ZERO_STAGE=0 # important: bf16 must use z0! it implements its own zero stage 1 equivalent

```









## CodeGeeX-13B

為了提高訓練效率，我們採用8路模型並行訓練和192路資料並行訓練，啟用 ZeRO-2 進一步減少最佳化器狀態的記憶體消耗。 最後，微批次大小為每個節點 16 個，全域性批次大小達到 3,072。


```
Model parameters： 13B
Vocabulary size： 52224
Position embedding： Learnable
Maximum sequence length： 2048
Hidden size h： 5120
Feed-forward size 4h： 20480
Feed-forward activation： FastGELU
Layernorm epsilon： 1e-5
Layernorm precision： FP32
Number of attention heads hn： 40
Attention softmax precision ：FP32
Dropout rate： 0.1

Global batch size: 3072
```


we use Adam optimizer (Kingma and Ba, 2014) to optimize the loss in Equation 2.
The model weights are under FP16 format, except that we use FP32 for layer-norm and softmax for
higher precision and stability. The model takes about 27GB of GPU memory. We start from an initial
learning rate 1e-4, and apply a cosine learning rate decay 




## GLM-130B

```
adopt 4-way tensor parallelism and 8-way pipeline parallelism
96 臺 A100（40G*8）


fp16 True

glu_activation geglu
hidden_size 12288
ffn_hidden_size 32768
num_layers 70
num_attention_heads 96
seq_length 2048
global_batch_size 4224
learning_rate 8e-05

```





## OPT-175B

- https://github.com/facebookresearch/metaseq/tree/main/projects/OPT

- [GitHub - facebookresearch/metaseq: Repo for external large-scale work](https://github.com/facebookresearch/metaseq/)

- [Fully Sharded Data Parallel | FairScale documentation](https://fairscale.readthedocs.io/en/stable/api/nn/fsdp.html)

- [Getting Started with Fully Sharded Data Parallel(FSDP) — PyTorch Tutorials 2.0.1+cu117 documentation](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)



```
FP16

trained OPT-175B on 992 80GB A100 GPUs, 

by utilizing Fully Sharded Data Parallel with Megatron-LM Tensor Parallelism

透過利用完全分片資料並行與 Megatron-LM 張量並行

roughly ~33 days of continuous training

300B tokens


```



### BloombergGPT



We use the Amazon SageMaker service provided by AWS to train and evaluate BloombergGPT. 

We use the latest version available at the time of training and
train on a total of 64 p4d.24xlarge instances. 

Each p4d.24xlarge instance has 8 NVIDIA 40GB A100 GPUs with NVIDIA NVSwitch intra-node connections (600 GB/s) and NVIDIA GPUDirect using AWS Elastic Fabric Adapter (EFA) inter-node connections (400 Gb/s).

This yields a total of 512 40GB A100 GPUs.




we rely on stage 3 of ZeRO optimization. We utilize the proprietary SageMaker Model Parallelism (SMP) library from AWS, which enables the automatic distribution of large models across multiple GPU devices and instances


ZeRO shards the training state (model parameters, gradients, and optimizer state) across a group of GPUs. We shard a model across 128 GPUs, and we have 4 copies of the model during training






## Megatron-Turing NLG（530B）

我們使用了 Transformer 解碼器的架構 [52]，它是一個從左到右、自迴歸、基於生成 Transformer 的語言模型，並將其擴充套件到 5300 億個引數。 層數、隱藏維度、注意力頭分別為 105、20480 和 128。 序列長度為2048，全域性批次大小為1920。

我們使用 8 路張量和 35 路管道並行。 學習率為5:0e−5。

我們使用 10 億個代幣進行線性學習率預熱。

我們使用餘弦衰減來使學習率目標達到超過 3400 億代幣價值的 10%。

在前 120 億個代幣中，我們從 32 的批次大小開始，並以 32 為增量逐漸增加批次大小，直到達到最終的批次大小 1920。

我們使用 Adam 最佳化器，β1 = 0:9、β2 = 0:95 和 = 10−8。 我們將梯度範數限制為 1.0，並使用 0.1 的權重衰減。 對於權重初始化，我們使用均值為零、標準差為 4:0e−3 的正態分佈。 我們的訓練資料集由 3390 億個令牌組成，我們透過混合上述 15 個訓練資料集在 2700 億個令牌上訓練 MT-NLG。

我們還留出 2% 的資料進行驗證。

對於 MT-NLG 等模型的規模，訓練穩定性是一個根本性的挑戰。 在訓練模型時，我們觀察到學習率、權重初始化和 Adam 最佳化器引數直接影響模型的穩定性。

我們透過在 [9] 中繪製學習率與模型大小來預測 MT-NLG 的學習率。 較高的學習率會增加模型的穩定性。 我們使用大約 p1=(3 ∗ H) 作為權重初始化的標準差，其中 H 表示隱藏維度的大小。 與[45]類似，我們還觀察到使用更高的方差進行權重初始化無法收斂。 我們還降低了 β2 的標準值 0:99，以減少訓練損失的峰值。







訓練過程一共使用了4480塊英偉達A100 GPU


5300億個引數的模型，每個模型副本跨越280個NVIDIA A100 GPU，節點內採用Megatron-LM的8路張量切片（tensor-slicing），節點間採用35路管道並行。


基於NVIDIA DGX SuperPOD的Selene超級計算機上完成混合精度訓練。（該超級計算機由560個DGX A100伺服器提供支援，每個DGX A100有8個 NVIDIA A100 80GB Tensor Core GPU，透過NVLink 和 NVSwitch相互完全連線）。



Model training is done with mixed precision using 16-bit bfloat on NVIDIA’s Selene supercomputer with 560 DGX A100 nodes. 

Each cluster node has 8 NVIDIA 80-GB A100 GPUs, connected to each other by NVLink and NVSwitch. 

Each node has eight NVIDIA Mellanox 200Gbps HDR Infiniband
HCAs for application communication, with an additional two HCAs per node for dedicated storage. 

The nodes are connected in a three-level (leaf, spine, core) fat-tree topology with 850 switches. 
This topology allows efficient all-reduce communication (which is the dominant communication pattern in deep learning
training). The cluster uses an all-NVME shared parallel filesystem for high-performance data access and
storage. The peak device throughput of an A100 GPU with 16-bit precision is 312 teraFLOP/s, resulting in
an aggregate of 1.4 exaFLOP/s of peak 16-bit precision performance



mixed precision using 16-bit bfloat 



## GPT-NeoX-20B


We trained GPT-NeoX-20B on twelve Supermicro AS-4124GO-NART servers, each with eight
NVIDIA A100-SXM4-40GB GPUs and configured with two AMD EPYC 7532 CPUs. 

All GPUs can directly access the InfiniBand switched fabric through one of four ConnectX-6 HCAs for
GPUDirect RDMA. 

Two NVIDIA MQM8700-HS2R switches—connected by 16 links—compose the spine of this InfiniBand network, with one link
per node CPU socket connected to each switch.

Figure 2 shows a simplified overview of a node as configured for training。







