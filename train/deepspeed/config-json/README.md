- https://www.deepspeed.ai/docs/config-json/


## Batch Size 相關的引數


train_batch_size 必須等於 train_micro_batch_size_per_gpu * gradient_accumulation * gpu數量



### train_batch_size


### train_micro_batch_size_per_gpu


### gradient_accumulation_steps

在平均和應用梯度之前進行累積梯度的訓練step數。 

此功能有時對於提高可擴充套件性很有用，因為它會降低step之間梯度通訊的頻率。 

此功能的另一個影響是能夠在每個 GPU 上使用更大的批次大小進行訓練。



## Optimizer 引數

- type：最佳化器名稱。 DeepSpeed 原生支援 Adam、AdamW、OneBitAdam、Lamb 和 OneBitLamb 最佳化器，同時，也可以從 torch 中匯入其他最佳化器。
   - https://deepspeed.readthedocs.io/en/latest/optimizers.html#optimizers
   - https://pytorch.org/docs/stable/optim.html
- params：用於例項化最佳化器的引數字典。引數名稱必須與最佳化器建構函式簽名匹配（例如，Adam）。
   - https://pytorch.org/docs/stable/optim.html#algorithms
   - https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

Adam 最佳化器示例：

```
"optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  }
```



## Scheduler 引數

當執行 model_engine.step() 時，DeepSpeed 在每個訓練步驟呼叫 scheduler 的 step() 方法。

- type：學習率排程器名，DeepSpeed 提供了 LRRangeTest、OneCycle、WarmupLR、WarmupDecayLR 學習率排程器的實現。
   - https://deepspeed.readthedocs.io/en/latest/schedulers.html
- params：用於例項化排程器的引數字典。引數名稱應與排程程式建構函式簽名匹配。

scheduler 示例：

```
 "scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": 0.001,
          "warmup_num_steps": 1000
      }
  }
```

## 通訊選項


### communication_data_type


### prescale_gradients


### gradient_predivide_factor


### sparse_gradients


## FP16 訓練選項

- 注意：此模式不能與下述 amp 模式結合使用。






```
"fp16": {
    "enabled": true,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "consecutive_hysteresis": false,
    "min_loss_scale": 1
}
```





## BFLOAT16 訓練選項

- 注意：此模式不能與下述amp模式結合使用。
- 注意：該模式不能與上述fp16模式結合使用。

使用 bfloat16 浮點格式作為 FP16 替代方案。 

BFLOAT16 需要硬體支援（例如：NVIDIA A100）。 

使用 bfloat16 進行訓練不需要損失縮放。

示例如下所示。 

```
"bf16": {
   "enabled": true
 }
```



## 自動混合精度 (AMP) 訓練選項

注意：該模式不能與上述fp16模式結合使用。 此外，該模式目前與 ZeRO 不相容。

```
"amp": {
    "enabled": true,
    ...
    "opt_level": "O1",
    ...
}
```


## 梯度裁剪(Gradient Clipping)

- gradient_clipping





## 針對 FP16 訓練的 ZeRO 最佳化


## 引數解除安裝（Parameter offloading）


啟用和配置 ZeRO 最佳化，將引數解除安裝到 CPU/NVMe。 僅適用於 ZeRO 階段 3。

- 注意，如果"device"的值未指定或不支援，則會觸發斷言。

```
 "offload_param": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 5,
    "buffer_size": 1e8,
    "max_in_cpu": 1e9
  }
```

## 最佳化器解除安裝

啟用和配置 ZeRO 最佳化，將最佳化器計算解除安裝到 CPU 並將最佳化器狀態解除安裝到 CPU/NVMe。 

CPU 解除安裝適用於 ZeRO 階段 1、2、3。NVMe 解除安裝僅適用於 ZeRO 階段 3。

- 注意，如果"device"的值未指定或不支援，則會觸發斷言。


```
 "offload_optimizer": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 4,
    "fast_init": false
  }
```

## Activation Checkpointing

```
"activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
    }
```


## 稀疏注意力（Sparse Attention）

```
"sparse_attention": {
 "mode": "fixed",
 "block": 16,
 "different_layout_per_head": true,
 "num_local_blocks": 4,
 "num_global_blocks": 1,
 "attention": "bidirectional",
 "horizontal_global_attention": false,
 "num_different_global_patterns": 4,
 "num_random_blocks": 0,
 "local_window_blocks": [4],
 "global_block_indices": [0],
 "global_block_end_indices": None,
 "num_sliding_window_blocks": 3
}
```


## Logging

### steps_per_print


### wall_clock_breakdown


### dump_state





## Flops 分析器（Flops Profiler）

- detailed：是否列印詳細的模型配置。
- output_file：輸出檔案的路徑。 如果沒有，Profiler 將列印到標準輸出。


```
{
  "flops_profiler": {
    "enabled": false,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null,
    }
}
```


## 監控模組（TensorBoard、WandB、CSV）


### tensorboard

TensorBoard配置示例：
```
"tensorboard": {
    "enabled": true,
    "output_path": "output/ds_logs/",
    "job_name": "train_bert"
}
```





## 壓縮（Compression）

### Layer Reduction

### 權重量化（Weight Quantization）


### 啟用量化（Activation Quantization）

### 稀疏剪枝（Sparse Pruning）

### 頭剪枝(Head Pruning)



### 通道剪枝（Channel Pruning）


## Checkpoint 選項

```
"checkpoint": {
    "tag_validation"="Warn",
    "load_universal"=false,
    "use_node_local_storage"=false,
    "parallel_write":{
        "pipeline_stage": false
    }
}
```



## 資料型別選項

```
"data_types": {
    "grad_accum_dtype"=["fp32"|"fp16"|"bf16"]
    }
}
```



## Data Efficiency







