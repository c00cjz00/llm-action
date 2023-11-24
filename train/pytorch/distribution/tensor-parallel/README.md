

```
from utils import cleanup, setup, ToyModel
model = ToyModel().cuda(0)
model
```

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   1521759      C   /opt/conda/bin/python            3820MiB |
+-----------------------------------------------------------------------------+

```


```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   4112660      C   /opt/conda/bin/python            1382MiB |
|    1   N/A  N/A   4112661      C   /opt/conda/bin/python            1382MiB |
|    2   N/A  N/A   4112662      C   /opt/conda/bin/python            1370MiB |
|    3   N/A  N/A   4112663      C   /opt/conda/bin/python            1370MiB |
+-----------------------------------------------------------------------------+




```




Tensor Parallelism(TP) 建立在 DistributedTensor(DTensor) 之上，並提供多種並行格式：Rowwise、Colwise 和 Pairwise Parallelism。


Rowwise，對模組的行進行分割槽。假設輸入是分片的 DTensor，則輸出是仿製的 DTensor。

Colwise，對張量或模組的列進行分割槽。 假設輸入是仿製的 DTensor，則輸出是分片的 DTensor。

PairwiseParallel 將 colwise 和 rowwise 樣式串聯為固定對，就像 [Megatron-LM](https://arxiv.org/abs/1909.08053) 所做的那樣。 我們假設輸入和輸出都需要複製 DTensor。

PairwiseParallel 目前僅支援 `nn.Multihead Attention`、`nn.Transformer` 或偶數層 `MLP`。

由於 Tensor Parallelism 是建立在 DTensor 之上的，因此我們需要使用 DTensor 指定模組的輸入和輸出位置，以便它可以在前後與模組進行預期的互動。 以下是用於輸入/輸出準備的函式：

torch.distributed.tensor.parallel.style.make_input_replicate_1d(input, device_mesh=None)












---

DeviceMesh 裝置網格















