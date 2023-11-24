




## 全量微調


生成用於Ascend晶片分散式通訊的晶片資源資訊配置檔案（RANK_TABLE_FILE）。

Ascend HCCL RANK_TABLE_FILE 檔案提供Ascend分散式訓練作業的叢集資訊。

```
# 如生成8卡的rank_table_file
> python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"

start ./mindformers/tools/hccl_tools.py
visible_devices:['0', '1', '2', '3', '4', '5', '6', '7']
server_id:192.168.1.196
device_num_list: [0, 1, 2, 3, 4, 5, 6, 7]
rank_id:0, device_id:0, device_ip:192.168.100.101
rank_id:1, device_id:1, device_ip:192.168.101.101
rank_id:2, device_id:2, device_ip:192.168.102.101
rank_id:3, device_id:3, device_ip:192.168.103.101
rank_id:4, device_id:4, device_ip:192.168.100.100
rank_id:5, device_id:5, device_ip:192.168.101.100
rank_id:6, device_id:6, device_ip:192.168.102.100
rank_id:7, device_id:7, device_ip:192.168.103.100
Completed: hccl file was save in : /root/workspace/code/mindformers/hccl_8p_01234567_192.168.1.196.json
```


### 修改配置

```
cd /root/workspace/code/mindformers
vim configs/glm/run_glm_6b_finetune.yaml
```


### 啟動訓練任務

```
> bash run_distribute.sh /root/workspace/code/mindformers/hccl_8p_01234567_192.168.1.196.json ../configs/glm/run_glm_6b_finetune.yaml '[0,8]' finetune
start training for rank 0, device 0
start training for rank 1, device 1
start training for rank 2, device 2
start training for rank 3, device 3
start training for rank 4, device 4
start training for rank 5, device 5
start training for rank 6, device 6
start training for rank 7, device 7
```


部分訓練日誌如下所示：

```
...
[INFO] 2023-07-11 10:35:39,223 [run_mindformer.py:71] main: moe config is: <mindformers.modules.transformer.moe.MoEConfig object at 0xffff10297b10>
[INFO] 2023-07-11 10:35:39,223 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:75] __init__: Now Running Task is: text_generation, Model is: glm_6b
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:177] _check_global_batch_size_for_auto_parallel: The current parallel mode is semi_auto_parallel, full batch is True,so global batch size will be changed: global_batch_size = batch_size * data_parallel * micro_batch_interleave_num = 8 * 1 * 1 = 8
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:514] training_process: .........Build Dataset For Train..........
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:268] create_train_dataset: .........Build Dataset From Config..........
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/dataset/causal_language_model_dataset.py:98] __new__: Now Create Causal Language Model Dataset.
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/dataset/base_dataset.py:50] init_dataset_config: Now the semi auto parallel mode is used and full_batch is True,and the shuffle of the dataset is required to be False,so as to ensure that the data loaded on each card is consistent and to avoid the problem of non-convergence of loss.
[INFO] 2023-07-11 10:35:39,231 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/utils.py:133] check_runner_config: Will be Training epochs:1, sink_size:4
[INFO] 2023-07-11 10:35:39,231 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/utils.py:134] check_runner_config: Create training dataset finish, dataset size:125
[INFO] 2023-07-11 10:35:39,231 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:521] training_process: .........Build Net For Train..........
[INFO] 2023-07-11 10:35:39,231 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:282] create_network: .........Build Network From Config..........
[INFO] 2023-07-11 10:38:43,280 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/models/base_model.py:80] load_checkpoint: weights in /root/workspace/model/chatglm-convert/ms_glm_6b.ckpt are loaded
[INFO] 2023-07-11 10:38:43,299 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:425] count_parameters: Network Parameters: 6707 M.
[INFO] 2023-07-11 10:38:43,299 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:544] training_process: .........Build Optimizer For Train..........
[INFO] 2023-07-11 10:38:43,299 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:321] create_optimizer_scheduler: .........Build Optimizer From Config..........
[INFO] 2023-07-11 10:38:43,299 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:354] create_lr_scheduler: .........Build LR Schedule From Config..........
[WARNING] 2023-07-11 10:38:43,306 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/optimizer_grouped_parameters.py:74] get_optimizer_grouped_parameters: dynamic_lr_schedule will be reset and invalid when layer_scale is False.
...
[INFO] 2023-07-11 10:38:43,568 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:550] training_process: .........Build Running Wrapper From Config For Train..........
[INFO] 2023-07-11 10:38:43,568 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:391] create_model_wrapper: .........Build Model Wrapper for Train From Config..........
[INFO] 2023-07-11 10:38:43,582 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:562] training_process: .........Starting Init Train Model..........
[INFO] 2023-07-11 10:38:43,583 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:581] training_process: .........Build Callbacks For Train..........
[INFO] 2023-07-11 10:38:43,583 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:400] create_callbacks: .........Build Callbacks for Train From Config..........
[INFO] 2023-07-11 10:38:43,584 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:340] __init__: Integrated_save is changed to False when using auto_parallel.
[INFO] 2023-07-11 10:38:43,585 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:609] training_process: .........Starting Training Model..........
[INFO] 2023-07-11 10:38:43,585 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:610] training_process: .........Model Compiling, Please Wait a Moment...........
[INFO] 2023-07-11 10:47:36,427 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[    4/  125], loss:[2.244/2.244], time:507844.205 ms, lr:[0.], overflow cond: True, loss_scale: 268435460.0
[INFO] 2023-07-11 10:47:37,342 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 533756.177 ms, per step time: 133439.044 ms, avg loss: 2.244
[INFO] 2023-07-11 10:47:44,861 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[    8/  125], loss:[2.499/2.499], time:7480.938 ms, lr:[0.], overflow cond: True, loss_scale: 16777216.0
[INFO] 2023-07-11 10:47:44,874 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 7518.224 ms, per step time: 1879.556 ms, avg loss: 2.499
...
[INFO] 2023-07-11 10:48:35,199 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 1958.791 ms, per step time: 489.698 ms, avg loss: 2.091
[INFO] 2023-07-11 10:48:37,162 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[  116/  125], loss:[2.220/2.220], time:1951.612 ms, lr:[2.4499998e-06], overflow cond: False, loss_scale: 16384.0
[INFO] 2023-07-11 10:48:37,163 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 1963.915 ms, per step time: 490.979 ms, avg loss: 2.220
[INFO] 2023-07-11 10:48:39,125 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[  120/  125], loss:[2.092/2.092], time:1953.753 ms, lr:[2.5499999e-06], overflow cond: False, loss_scale: 16384.0
[INFO] 2023-07-11 10:48:39,126 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 1962.049 ms, per step time: 490.512 ms, avg loss: 2.092
[INFO] 2023-07-11 10:48:41,083 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[  124/  125], loss:[2.346/2.346], time:1949.995 ms, lr:[2.65e-06], overflow cond: False, loss_scale: 16384.0
[INFO] 2023-07-11 10:48:41,084 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 1958.268 ms, per step time: 489.567 ms, avg loss: 2.346
[INFO] 2023-07-11 10:49:26,307 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:616] training_process: .........Training Over!.............
```



## LoRA微調


### 修改配置
```
cd /root/workspace/code/mindformers
vim configs/glm/run_glm_6b_lora.yaml
```




模型訓練啟動成功，輸出目錄的結構如下所示。
```
output/
├── checkpoint
├── log
└── strategy
```
其中，checkpoint資料夾放置權重檔案，log資料夾方式日誌檔案，strategy資料夾放置模型切分策略檔案。


檢視日誌：
```
# cd /root/workspace/code/mindformers/output/
cd log/rank_0
tail -100f info.log 
```

模型輸出權重檔案：

```
> tree -h checkpoint/
checkpoint/
├── [ 4.0K]  rank_0
│   ├── [ 3.4G]  glm-6b-lora_rank_0-31_4.ckpt
│   └── [ 6.5M]  glm-6b-lora_rank_0-graph.meta
├── [ 4.0K]  rank_1
│   ├── [ 3.4G]  glm-6b-lora_rank_1-31_4.ckpt
│   └── [ 6.5M]  glm-6b-lora_rank_1-graph.meta
├── [ 4.0K]  rank_2
│   ├── [ 3.4G]  glm-6b-lora_rank_2-31_4.ckpt
│   └── [ 6.5M]  glm-6b-lora_rank_2-graph.meta
└── [ 4.0K]  rank_3
    ├── [ 3.4G]  glm-6b-lora_rank_3-31_4.ckpt
    └── [ 6.5M]  glm-6b-lora_rank_3-graph.meta

4 directories, 8 files
```
模型切分策略檔案。
```
> tree -h strategy/
strategy/
├── [  22K]  ckpt_strategy_rank_0.ckpt
├── [  22K]  ckpt_strategy_rank_1.ckpt
├── [  22K]  ckpt_strategy_rank_2.ckpt
└── [  22K]  ckpt_strategy_rank_3.ckpt
```



## 權重合並


### 全量微調



```
python3 merge_ckpt.py --src_postfix=31_4 \
> --src_checkpoints_dir=/root/workspace/output/fullft_output \
> --src_strategy_file=/root/workspace/output/fullft_output/strategy/ckpt_strategy_rank_0.ckpt \
> --dst_checkpoints_dir=/root/workspace/output/fullft_merge_checkpoint/

args_opt.src_strategy_file:  /root/workspace/output/fullft_output/strategy/ckpt_strategy_rank_0.ckpt
checkpoint_file_map {7: '/root/workspace/output/fullft_output/checkpoint/rank_7/glm-6b_rank_7-31_4.ckpt', 6: '/root/workspace/output/fullft_output/checkpoint/rank_6/glm-6b_rank_6-31_4.ckpt', 5: '/root/workspace/output/fullft_output/checkpoint/rank_5/glm-6b_rank_5-31_4.ckpt', 4: '/root/workspace/output/fullft_output/checkpoint/rank_4/glm-6b_rank_4-31_4.ckpt', 3: '/root/workspace/output/fullft_output/checkpoint/rank_3/glm-6b_rank_3-31_4.ckpt', 2: '/root/workspace/output/fullft_output/checkpoint/rank_2/glm-6b_rank_2-31_4.ckpt', 1: '/root/workspace/output/fullft_output/checkpoint/rank_1/glm-6b_rank_1-31_4.ckpt', 0: '/root/workspace/output/fullft_output/checkpoint/rank_0/glm-6b_rank_0-31_4.ckpt'}
save_checkpoint_path /root/workspace/output/fullft_merge_checkpoint/transformed.ckpt

[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:32:38.347.469 [mindspore/parallel/_parallel_serialization.py:351] The parameter scale_sense is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:32:38.347.863 [mindspore/parallel/_parallel_serialization.py:351] The parameter global_step is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.541.985 [mindspore/parallel/_parallel_serialization.py:351] The parameter current_iterator_step is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.542.313 [mindspore/parallel/_parallel_serialization.py:351] The parameter last_overflow_iterator_step is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.542.392 [mindspore/parallel/_parallel_serialization.py:351] The parameter epoch_num is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.542.460 [mindspore/parallel/_parallel_serialization.py:351] The parameter step_num is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.542.523 [mindspore/parallel/_parallel_serialization.py:351] The parameter loss_scale is not in src_strategy.

transform ckpt done.
Filtering ckpt, this may take a while.

100%|###############################################################################| 1027/1027 [00:35<00:00, 28.57it/s]
```

合併之後的權重檔案如下所示：
```
> tree -h /root/workspace/output/fullft_merge_checkpoint
/root/workspace/output/fullft_merge_checkpoint
├── [  13G]  filtered_transformed.ckpt
└── [  63G]  transformed.ckpt
```


### LoRA 微調
```
python3 merge_ckpt.py --src_postfix=31_4 \
--src_checkpoints_dir=/root/workspace/output/lora_output \
--src_strategy_file=/root/workspace/code/mindformers/output/strategy/ckpt_strategy_rank_0.ckpt \
--dst_checkpoints_dir=/root/workspace/output/lora_merge_checkpoint_v2/
```


## 模型評估

### 全量微調模型評估

<details><summary>詳細輸出</summary><p>

```
python run_mindformer.py \
> --config ./configs/glm/run_glm_6b_infer.yaml \
> --run_mode eval \
> --load_checkpoint /root/workspace/output/fullft_merge_checkpoint/filtered_transformed.ckpt \
> --eval_dataset_dir /root/workspace/data/AdvertiseGen-ms/eval_0711_256.mindrecord \
> --device_id 7
2023-07-11 16:52:59,073 - mindformers - INFO - full_batch will be forced to False when the parallel mode is stand_alone or data_parallel
2023-07-11 16:52:59,075 - mindformers - INFO - .........Build context config..........
2023-07-11 16:52:59,075 - mindformers - INFO - initial moe_config from dict: {'expert_num': 1, 'capacity_factor': 1.05, 'aux_loss_factor': 0.05, 'num_experts_chosen': 1}
2023-07-11 16:52:59,075 - mindformers - INFO - initial recompute_config from dict: {'recompute': False, 'parallel_optimizer_comm_recompute': False, 'mp_comm_recompute': True, 'recompute_slice_activation': False}
2023-07-11 16:52:59,075 - mindformers - INFO - initial parallel_config from dict: {'data_parallel': 1, 'model_parallel': 1, 'pipeline_stage': 1, 'expert_parallel': 1, 'optimizer_shard': False, 'micro_batch_num': 1, 'vocab_emb_dp': True, 'gradient_aggregation_group': 4}
2023-07-11 16:52:59,075 - mindformers - INFO - context config is: [ParallelConfig]
_recompute:[ParallelConfig]
_recompute:False
_parallel_optimizer_comm_recompute:False
_mp_comm_recompute:True
_recompute_slice_activation:False

_optimizer_shard:False
_gradient_aggregation_group:4
_embed_dp_mp_config:[ParallelConfig]
_dp_mp_config:[ParallelConfig]
_data_parallel:1
_model_parallel:1

_vocab_emb_dp:True

_pp_config:[ParallelConfig]
_pipeline_stage:1
_micro_batch_num:1

_moe_config:[ParallelConfig]
_dpmp:[ParallelConfig]
_data_parallel:1
_model_parallel:1

_expert_parallel:1


2023-07-11 16:52:59,076 - mindformers - INFO - moe config is: <mindformers.modules.transformer.moe.MoEConfig object at 0xffff73d74690>
{'auto_trans_ckpt': False,
 'auto_tune': False,
 'autotune_per_step': 10,
 'callbacks': [OrderedDict([('type', 'MFLossMonitor')]),
               OrderedDict([('type', 'SummaryMonitor'),
                            ('keep_default_action', True)]),
               OrderedDict([('type', 'CheckpointMointor'),
                            ('prefix', 'glm-6b'),
                            ('save_checkpoint_steps', 500),
                            ('keep_checkpoint_max', 2),
                            ('integrated_save', False),
                            ('async_save', False)]),
               OrderedDict([('type', 'ObsMonitor'), ('keep_last', False)])],
 'context': {'device_id': 7,
             'device_target': 'Ascend',
             'enable_graph_kernel': False,
             'graph_kernel_flags': '--disable_expand_ops=Softmax,Dropout '
                                   '--enable_parallel_fusion=true '
                                   '--reduce_fuse_depth=8 '
                                   '--enable_auto_tensor_inplace=true',
             'max_call_depth': 10000,
             'save_graphs': False,
             'save_graphs_path': './graph'},
 'device_num': 1,
 'eval_callbacks': [OrderedDict([('type', 'ObsMonitor'),
                                 ('keep_last', False)])],
 'eval_dataset': {'batch_size': 1,
                  'data_loader': {'dataset_dir': '/root/workspace/data/AdvertiseGen-ms/eval_0711_256.mindrecord',
                                  'shuffle': True,
                                  'type': 'MindDataset'},
                  'drop_remainder': True,
                  'input_columns': ['input_ids', 'label'],
                  'num_parallel_workers': 8,
                  'numa_enable': False,
                  'prefetch_size': 1,
                  'python_multiprocessing': False,
                  'repeat': 1,
                  'seed': 0},
 'eval_dataset_task': {'dataset_config': {'batch_size': 1,
                                          'data_loader': {'dataset_dir': '',
                                                          'shuffle': True,
                                                          'type': 'MindDataset'},
                                          'drop_remainder': True,
                                          'input_columns': ['input_ids',
                                                            'label'],
                                          'num_parallel_workers': 8,
                                          'numa_enable': False,
                                          'prefetch_size': 1,
                                          'python_multiprocessing': False,
                                          'repeat': 1,
                                          'seed': 0},
                       'type': 'CausalLanguageModelDataset'},
 'filepath_prefix': './autotune',
 'init_start_profile': True,
 'load_checkpoint': None,
 'local_rank': 0,
 'lr_schedule': {'learning_rate': 5e-05,
                 'lr_end': 1e-06,
                 'total_steps': -1,
                 'type': 'polynomial',
                 'warmup_steps': 2000},
 'metric': {'tokenizer_type': 'glm_6b', 'type': 'ADGENMetric'},
 'micro_batch_interleave_num': 1,
 'model': {'arch': {'type': 'GLMChatModel'},
           'model_config': {'activation_func': 'GELU',
                            'attention_dropout_rate': 0.0,
                            'bos_token_id': 130004,
                            'checkpoint_name_or_path': '/root/workspace/output/fullft_merge_checkpoint/filtered_transformed.ckpt',
                            'compute_dtype': 'float16',
                            'do_sample': True,
                            'embedding_dropout_prob': 0.0,
                            'eos_token_id': 130005,
                            'gmask_token_id': 130001,
                            'hidden_dropout_rate': 0.0,
                            'hidden_size': 4096,
                            'hidden_size_per_attention_head': None,
                            'inner_hidden_size': 16384,
                            'is_enhanced_encoder': True,
                            'is_npu_acceleration': True,
                            'layernorm_compute_type': 'float32',
                            'layernorm_epsilon': 1e-05,
                            'layernorm_order': 'post',
                            'mask_token_id': 130000,
                            'max_decode_length': 2048,
                            'num_heads': 32,
                            'num_layers': 28,
                            'pad_token_id': 3,
                            'param_init_type': 'float16',
                            'position_encoding_2d': True,
                            'repetition_penalty': 1,
                            'seq_length': 512,
                            'softmax_compute_type': 'float32',
                            'top_k': 1,
                            'top_p': 1,
                            'type': 'GLMConfig',
                            'use_final_layernorm': True,
                            'use_past': True,
                            'vocab_size': 130528}},
 'moe_config': <mindformers.modules.transformer.moe.MoEConfig object at 0xffff73d74690>,
 'only_save_strategy': False,
 'optimizer': {'beta1': 0.9,
               'beta2': 0.95,
               'eps': 1e-08,
               'type': 'FusedAdamWeightDecay',
               'weight_decay': 0.1},
 'output_dir': './output',
 'parallel': {'enable_alltoall': False,
              'enable_parallel_optimizer': False,
              'full_batch': True,
              'gradients_mean': False,
              'loss_repeated_mean': True,
              'parallel_mode': 0,
              'search_mode': 'sharding_propagation',
              'strategy_ckpt_save_file': './output/strategy/./ckpt_strategy_rank_0.ckpt'},
 'parallel_config': <mindformers.modules.transformer.transformer.TransformerOpParallelConfig object at 0xffff3085a690>,
 'processor': {'return_tensors': 'ms',
               'tokenizer': {'bos_token': '<sop>',
                             'end_token': '</s>',
                             'eos_token': '<eop>',
                             'gmask_token': '[gMASK]',
                             'mask_token': '[MASK]',
                             'pad_token': '<pad>',
                             'padding_side': 'left',
                             'type': 'ChatGLMTokenizer',
                             'unk_token': '<unk>'},
               'type': 'GLMProcessor'},
 'profile': False,
 'profile_communication': True,
 'profile_memory': True,
 'profile_start_step': 1,
 'profile_stop_step': 10,
 'recompute_config': <mindformers.modules.transformer.transformer.TransformerRecomputeConfig object at 0xffff302b7bd0>,
 'remote_save_url': 'Please input obs url on AICC platform.',
 'resume_training': False,
 'run_mode': 'eval',
 'runner_config': {'batch_size': 1,
                   'epochs': 1,
                   'sink_mode': True,
                   'sink_size': 4},
 'runner_wrapper': {'scale_sense': {'loss_scale_value': 4294967296,
                                    'scale_factor': 2,
                                    'scale_window': 1000,
                                    'type': 'DynamicLossScaleUpdateCell'},
                    'type': 'MFTrainOneStepCell',
                    'use_clip_grad': True},
 'seed': 0,
 'train_dataset': {'batch_size': 1,
                   'data_loader': {'dataset_dir': '',
                                   'shuffle': True,
                                   'type': 'MindDataset'},
                   'drop_remainder': True,
                   'input_columns': ['input_ids',
                                     'label',
                                     'position_ids',
                                     'attention_mask'],
                   'num_parallel_workers': 8,
                   'numa_enable': False,
                   'prefetch_size': 1,
                   'python_multiprocessing': False,
                   'repeat': 1,
                   'seed': 0},
 'train_dataset_task': {'dataset_config': {'batch_size': 1,
                                           'data_loader': {'dataset_dir': '',
                                                           'shuffle': True,
                                                           'type': 'MindDataset'},
                                           'drop_remainder': True,
                                           'input_columns': ['input_ids',
                                                             'label',
                                                             'position_ids',
                                                             'attention_mask'],
                                           'num_parallel_workers': 8,
                                           'numa_enable': False,
                                           'prefetch_size': 1,
                                           'python_multiprocessing': False,
                                           'repeat': 1,
                                           'seed': 0},
                        'type': 'CausalLanguageModelDataset'},
 'trainer': {'model_name': 'glm_6b', 'type': 'CausalLanguageModelingTrainer'},
 'use_parallel': False}
2023-07-11 16:52:59,081 - mindformers - INFO - Now Running Task is: text_generation, Model is: glm_6b
2023-07-11 16:52:59,082 - mindformers - INFO - The current parallel mode is stand_alone, batch size per card will not be changed: batch_size_per_card = 1
2023-07-11 16:52:59,082 - mindformers - INFO - global_batch_size = batch_size_per_card * device_num = 1 * 1 = 1
2023-07-11 16:52:59,082 - mindformers - INFO - parallel_config will be change to default config: [ParallelConfig]
_recompute:[ParallelConfig]
_recompute:False
_parallel_optimizer_comm_recompute:False
_mp_comm_recompute:True
_recompute_slice_activation:False

_optimizer_shard:False
_gradient_aggregation_group:4
_embed_dp_mp_config:[ParallelConfig]
_dp_mp_config:[ParallelConfig]
_data_parallel:1
_model_parallel:1

_vocab_emb_dp:True

_pp_config:[ParallelConfig]
_pipeline_stage:1
_micro_batch_num:1

_moe_config:[ParallelConfig]
_dpmp:[ParallelConfig]
_data_parallel:1
_model_parallel:1

_expert_parallel:1

.
2023-07-11 16:52:59,082 - mindformers - INFO - .........Build Dataset For Evaluate..........
2023-07-11 16:52:59,082 - mindformers - INFO - .........Build Dataset From Config..........
2023-07-11 16:52:59,083 - mindformers - INFO - Now Create Causal Language Model Dataset.
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:52:59.839.42 [mindspore/dataset/core/validator_helpers.py:806] 'TypeCast' from mindspore.dataset.transforms.c_transforms is deprecated from version 1.8 and will be removed in a future version. Use 'TypeCast' from mindspore.dataset.transforms instead.
2023-07-11 16:52:59,084 - mindformers - INFO - .........Build Network From Config..........
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:53:02.472.83 [mindspore/common/_decorator.py:40] 'TensorAdd' is deprecated from version 1.1 and will be removed in a future version, use 'Add' instead.
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:56:17.765.324 [mindspore/train/serialization.py:1058] For 'load_param_into_net', 56 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict', please check whether the network structure is consistent when training and loading checkpoint.
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:56:17.765.833 [mindspore/train/serialization.py:1060] transformer.layers.0.key_past is not loaded.
...
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:56:17.768.487 [mindspore/train/serialization.py:1060] transformer.layers.27.value_past is not loaded.
2023-07-11 16:56:17,768 - mindformers - INFO - weights in /root/workspace/output/fullft_merge_checkpoint/filtered_transformed.ckpt are loaded
2023-07-11 16:56:17,878 - mindformers - INFO - Network Parameters: 6825 M.
2023-07-11 16:56:17,878 - mindformers - INFO - .........Build Compute Metrics For Evaluate..........
2023-07-11 16:56:17,878 - mindformers - INFO - Config in the yaml file ./checkpoint_download/glm/glm_6b.yaml are used for tokenizer building.
2023-07-11 16:56:18,492 - mindformers - INFO - Load the tokenizer name ChatGLMTokenizer from the ./checkpoint_download/glm/glm_6b.yaml
2023-07-11 16:56:18,528 - mindformers - INFO - config in the yaml file ./checkpoint_download/glm/glm_6b.yaml are used for tokenizer building.
2023-07-11 16:56:18,564 - mindformers - WARNING - Can't find the tokenizer_config.json in the file_dict. The content of file_dict is : {}
2023-07-11 16:56:18,565 - mindformers - INFO - build tokenizer class name is: ChatGLMTokenizer using args {'bos_token': '<sop>', 'eos_token': '<eop>', 'end_token': '</s>', 'mask_token': '[MASK]', 'gmask_token': '[gMASK]', 'padding_side': 'left', 'pad_token': '<pad>', 'unk_token': '<unk>', 'vocab_file': './checkpoint_download/glm/ice_text.model'}.
2023-07-11 16:56:18,969 - mindformers - INFO - ChatGLMTokenizer Tokenizer built successfully!
2023-07-11 16:56:18,969 - mindformers - INFO - .........Starting Init Evaluate Model..........
2023-07-11 16:56:18,970 - mindformers - INFO - .........Starting Evaluate Model..........

2023-07-11 17:00:33,480 - mindformers - INFO - Epoch 1 Finished, cost time 254.49472093582153,  every example cost time is 254.49472093582153, generate speed: 0.13359805608138378 tokens/s, avg speed: 0.0 tokens/s
pred is:
 以白色為底,以清新、淡雅的刺繡花朵為裝飾,將v領與抽褶的元素融入其中,將簡約與浪漫完美演繹。
 label is:
 簡單大氣純白色連衣裙,是開春季節最美好的穿搭單品。簡單的小v領點綴領部,加以獨特的花邊繡花點綴,滿滿的清新活力悠然散發。加以純粹的白色選料,上身親膚透氣,自帶自然的褶皺肌理。同時,中長款式,修飾好身材,十分美膩。
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 1.166 seconds.
Prefix dict has been built successfully.
2023-07-11 17:00:40,003 - mindformers - INFO - Epoch 2 Finished, cost time 5.341211557388306,  every example cost time is 5.341211557388306, generate speed: 24.151823722757985 tokens/s, avg speed: 24.151371552472405 tokens/s
pred is:
 一款吊帶裙,以清新的白色為主色調,簡約而優雅。採用薄而透氣的雪紡面料,在細節處加入精緻的花邊,讓整件裙子在細節處盡顯女性的柔美。 v領設計,露出性感的鎖骨,性感又迷人。 腰部的拼接設計,將整件裙子的層次感演繹的恰到好處,既不失簡約的大氣,又凸顯了女性的柔美。 下襬的寬鬆設計,在行走中自然流動,將整件裙子的優雅氣質展現的淋淋盡致。 整體設計以休閒為主,在細節處加入精緻的蝴蝶結,讓整件裙子更顯甜美。
 label is:
 優美而動感的上衣。採用半透的雪紡材質工藝,深黑色系給您以非常魅惑的穿著體驗,內裡需要搭配深黑色的吊帶。花邊v字領口連襟拼接,舉手投足更加優雅迷人,適合搭配各種半身裙和休閒長褲。
2023-07-11 17:00:41,904 - mindformers - INFO - Epoch 3 Finished, cost time 1.8842175006866455,  every example cost time is 1.8842175006866455, generate speed: 23.88259316326333 tokens/s, avg speed: 24.08128160602231 tokens/s
pred is:
 以白色為底,以黑色線條為裝飾,將高腰與a字裙的設計元素融合在一起,打造出一種獨特的時尚風格。將簡約與個性元素相結合,讓穿著者更加有型有款。
 label is:
 這款裙子採用黑色的顏色打底,裙身上裝飾著白色的線條以及釦子裝飾,豐富視覺上的變化。另外整體上a字裙裙型搭配高腰的設計,修身效果出眾,還有著不規則的裙襬,展現出十足的設計感。
2023-07-11 17:00:43,475 - mindformers - INFO - Epoch 4 Finished, cost time 1.561234951019287,  every example cost time is 1.561234951019287, generate speed: 23.699187605199157 tokens/s, avg speed: 24.013391025594462 tokens/s
pred is:
 以蕾絲為底,以宮廷刺繡為裝飾,以大裙襬和泡泡袖為特點,將時尚與復古元素相結合,打造一款華麗、浪漫的蕾絲裙。
 label is:
 宮廷風的甜美蕾絲設計,清醒的蕾絲拼縫處,刺繡定製的貝殼花邊,增添了裙子的精緻感覺。超大的裙襬,加上精細的小花邊設計,上身後既帶著仙氣撩人又很有女人味。泡泡袖上的提花面料,在細節處增加了浪漫感,春日的仙女姐姐。浪漫蕾絲布滿整個裙身,美麗明豔,氣質超仙。
...
2023-07-11 17:02:40,918 - mindformers - INFO - Epoch 45 Finished, cost time 5.272047996520996,  every example cost time is 5.272047996520996, generate speed: 25.98610636519352 tokens/s, avg speed: 24.71114413227348 tokens/s
pred is:
 黑色條紋褲是經典的時尚單品,不僅簡約大方,還非常百搭,可以搭配各種上衣,讓你時尚又氣質。

搭配白色T恤,簡約清新,白色與黑色條紋的搭配非常顯瘦,還很有層次感。

搭配印花T恤,印花元素非常可愛,搭配黑色條紋褲,很有小清新的感覺。

搭配牛仔襯衫,牛仔襯衫的休閒感與黑色條紋褲的時尚感相結合,非常時髦。

搭配皮質外套,皮質外套的帥氣與黑色條紋褲的休閒感相結合,很有時尚感。

黑色條紋褲可以搭配各種上衣,非常百搭,而且簡約大方,讓你時尚又氣質。
 label is:
 傳承動感簡約氣質的條紋衣身,結合包邊圓領和半開襟設計,造型顯得活力有範,又不失男孩子的時尚帥氣。胸前單側小口袋點綴,讓男寶寶帥氣加倍。搭配純黑色的底褲,整體顯得層次十足,視覺也十分有美感,男寶寶穿起來獨特魅力盡顯。
2023-07-11 17:02:42,877 - mindformers - INFO - Epoch 46 Finished, cost time 1.9328925609588623,  every example cost time is 1.9328925609588623, generate speed: 25.350607162403406 tokens/s, avg speed: 24.720831918537765 tokens/s
pred is:
 以簡約為靈魂,以純色為基石,以條紋為元素,將時尚與舒適結合,將簡約與個性展現。這款外套,時尚與實用並存,在細節處展現品質,在簡約中彰顯個性。
 label is:
 來自巴拉巴拉的女童長款外套,設計師採用直筒式衣袖裁剪,並在袖口加飾有純色條紋,在打破了整體的單一性的同時,還增添了一絲簡約時尚氣息。再加上對稱的斜插口袋,既能給予嬌嫩雙手溫暖,同時還可放置孩子的隨身物品,暖心又很實用呢。
2023-07-11 17:02:45,750 - mindformers - INFO - Epoch 47 Finished, cost time 2.860628366470337,  every example cost time is 2.860628366470337, generate speed: 25.16929526530534 tokens/s, avg speed: 24.730666589943375 tokens/s
pred is:
 以蝴蝶結為元素,將層疊的網紗與繫帶進行搭配,將整體設計打造為半裙的半身裙,在腰部加入門襟設計,將整體設計打造為繫帶的蝴蝶結,在腰部的層疊層疊的網紗與蝴蝶結的點綴下,讓整件裙子在細節處盡顯盡顯的時尚感。
 label is:
 層疊網紗,仙氣飄飄,卻不會過於膨脹。腰間的蝴蝶結繫帶,恰到好處的增添了柔美感。膝蓋以下,長度剛剛好的半身裙,比起“一覽無遺魅力盡顯”,專注於“完美隱藏”
2023-07-11 17:02:47,855 - mindformers - INFO - Epoch 48 Finished, cost time 2.094264030456543,  every example cost time is 2.094264030456543, generate speed: 24.829725022142583 tokens/s, avg speed: 24.732231816627035 tokens/s
pred is:
 焦糖色連衣裙,簡約而不失優雅,寬鬆的版型,穿上身更顯氣質。百褶的裙襬,在腰部的收腰設計,更顯身材的纖細。搭配寬鬆的腰帶,將身材比例修飾的更好。整體的設計,更顯女性的柔美。
 label is:
 來自<UNK>自制的連衣裙採用今年大熱的焦糖色,就像巧克力一樣,甜蜜又不膩人。腰帶的貼心設計,讓寬鬆的版型也能擁有s曲線。上身簡約的襯衫式翻領,襯托小v臉,帶來一股職場ol風,加以百褶下襬的點綴,一起述說無盡溫柔。
2023-07-11 17:02:50,774 - mindformers - INFO - Epoch 49 Finished, cost time 2.9023842811584473,  every example cost time is 2.9023842811584473, generate speed: 25.15173489392764 tokens/s, avg speed: 24.74122134217671 tokens/s
pred is:
 一款時尚舒適的羊毛九分微喇褲,採用優質羊毛面料打造,柔軟親膚,保暖舒適。微喇褲的褲口設計,修飾腿型,展現優美曲線。九分的長度,修飾腳踝,顯瘦顯高挑。褲身的流線型設計,修飾腰部,展現身材比例。搭配一件簡約的毛衣,即可穿出氣質。
 label is:
 不同於一般的西服褲。這款<UNK>小喇叭羊毛褲在樣式上顯得更加時髦優雅,特地採用微微的九分喇叭褲腿設計,視覺上將腳踝處顯得更加纖細。並且特地甄選柔軟的羊毛材質,就算直接貼膚穿著,也不會覺得寒冷,比較適合初秋穿噢。
2023-07-11 17:02:53,838 - mindformers - INFO - Epoch 50 Finished, cost time 3.0505833625793457,  every example cost time is 3.0505833625793457, generate speed: 25.241073869521973 tokens/s, avg speed: 24.75223162149713 tokens/s
pred is:
 以綠色為主色調,將復古與時尚相結合,設計寬鬆的版型,在細節處加入復古圖案,讓整件裙子更加有層次感。在腰部加入的褶皺設計,讓腰部更加修飾,同時讓整件裙子更加有型。在裙襬處加入的燈籠袖設計,讓整件裙子更加有層次感,同時讓裙襬更加修飾身材。
 label is:
 袖子有燈籠袖的既視感,中世紀的復古韻味輕鬆展現,版型寬鬆舒適,上身貼合身材,不會顯胖。超級百搭,秋季單穿,搭配裙子褲子都ok!冬天也能做打底,外搭毛呢大衣,氣質滿滿。
2023-07-11 17:02:53,853 - mindformers - INFO - metric: Text Generation Metric 
rouge-1: 25.986109999999993 
rouge-2: 4.22121 
rouge-l: 21.202881999999995 
bleu-4:  4.540767999999999 
2023-07-11 17:02:53,854 - mindformers - INFO - ...........Evaluate Over!...............
```

</p></details>



### LoRA 微調模型評估

```
python run_mindformer.py \
--config ./configs/glm/run_glm_6b_lora_infer.yaml \
--run_mode eval \
--load_checkpoint /root/workspace/output/lora_merge_checkpoint_v2/filtered_transformed.ckpt \
--eval_dataset_dir /root/workspace/data/AdvertiseGen-ms/eval_0711_256.mindrecord \
--device_id 0
```




