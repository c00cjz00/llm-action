


# DDP

- https://zhuanlan.zhihu.com/p/343951042



## 資料並行

當一張 GPU 可以儲存一個模型時，可以採用資料並行得到更準確的梯度或者加速訓練，即每個 GPU 複製一份模型，將一批樣本分為多份輸入各個模型平行計算。因為求導以及加和都是線性的，資料並行在數學上也有效。

## DP


model = nn.DataParallel(model)



## DDP


DDP 透過 Reducer 來管理梯度同步。為了提高通訊效率， Reducer 會將梯度歸到不同的桶裡（按照模型引數的 reverse order， 因為反向傳播需要符合這樣的順序），一次歸約一個桶。其中，桶的大小為引數 bucket_cap_mb 預設為 25，可根據需要調整。


可以看到每個程序裡，模型引數都按照倒序放在桶裡，每次歸約一個桶。


DDP 透過在構建時註冊 autograd hook 進行梯度同步。反向傳播時，當一個梯度計算好後，相應的 hook 會告訴 DDP 可以用來歸約。

當一個桶裡的梯度都可以了，Reducer 就會啟動非同步 allreduce 去計算所有程序的平均值。allreduce 非同步啟動使得 DDP 可以邊計算邊通訊，提高效率。

當所有桶都可以了，Reducer 會等所有 allreduce 完成，然後將得到的梯度寫到 param.grad。




### DDP+MP

DDP與流水線並行。







## launch



透過 launch.py 啟動，在 8 個 GPU 節點上，每個 GPU 一個程序：
```
python /home/guodong.li/virtual-venv/megatron-ds-venv-py310-cu117/lib/python3.10/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=8 example.py --local_world_size=8

python /home/guodong.li/virtual-venv/megatron-ds-venv-py310-cu117/lib/python3.10/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=1 example.py --local_world_size=1

```




## torch elastic/torchrun


```
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py
```

我們在兩臺主機上執行 DDP 指令碼，每臺主機執行 8 個程序，也就是說，我們在 16 個 GPU 上執行它。 

請注意，所有節點上的 $MASTER_ADDR 必須相同。

這裡torchrun將啟動8個程序，並在啟動它的節點上的每個程序上呼叫elastic_ddp.py，
但使用者還需要應用slurm等叢集管理工具才能在2個節點上實際執行此命令。


例如，在啟用 SLURM 的叢集上，我們可以編寫一個指令碼來執行上面的命令並將 MASTER_ADDR 設定為：

```
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
```

然後我們可以使用 SLURM 命令執行此指令碼：

```
srun --nodes=2 ./torchrun_script.sh
```

當然，這只是一個例子； 您可以選擇自己的叢集排程工具來啟動torchrun作業。



- 詳細啟動命令：https://pytorch.org/docs/stable/elastic/quickstart.html




```
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 elastic_ddp.py
```


```
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=10.xx.2.46:29400 multigpu_torchrun.py --batch_size 32  10 5
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=10.xx.2.46:29400 multigpu_torchrun.py --batch_size 32  10 5
```





## 官方文件
- DDP 教程：https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- DDP 設計：https://pytorch.org/docs/master/notes/ddp.html
- DDP 示例：
	- https://github.com/pytorch/examples/tree/main/distributed/ddp (官方沒有更新了)
	- https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series




# FSDP



- GETTING STARTED WITH FULLY SHARDED DATA PARALLEL(FSDP)：https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- ADVANCED MODEL TRAINING WITH FULLY SHARDED DATA PARALLEL (FSDP)：https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html



















