

- srun文件： https://slurm.schedmd.com/srun.html


## pytorch



### 單機多卡
```

```


### 多機多卡


```

```


## deepspeed


### 單機多卡

```
deepspeed --include localhost:0,1,2,3 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```


### 多機多卡

```
python -m torch.distributed.run --nproc_per_node=2 --nnode=2 --node_rank=0 --master_addr=10.99.2.xx \
--master_port=9901 train.py --deepspeed_config=ds_config.json -p 2 --steps=200


python -m torch.distributed.run --nproc_per_node=2 --nnode=2 --node_rank=1 --master_addr=10.99.2.xx \
--master_port=9901 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```


### 單機多卡+docker

```
sudo docker run -it --rm --gpus all \
--network=host \
--shm-size 4G \
-v /data/hpc/home/guodong.li/:/workspaces \
-v /data/hpc/home/guodong.li/.cache/:/root/.cache/ \
-w /workspaces/DeepSpeedExamples-20230430/training/pipeline_parallelism \
deepspeed/deepspeed:v072_torch112_cu117 /bin/bash

deepspeed --include localhost:4,5,6,7 --master_port 29001 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```


### 單機多卡+singularity


```
docker tag deepspeed/deepspeed:v072_torch112_cu117 harbor.aip.io/base/deepspeed:torch112_cu117
sudo docker push harbor.aip.io/base/deepspeed:torch112_cu117
SINGULARITY_NOHTTPS=1 singularity build deepspeed.sif docker://harbor.aip.io/base/deepspeed:torch112_cu117


singularity run --nv \
--pwd /workspaces/DeepSpeedExamples-20230430/training/pipeline_parallelism \
-B /data/hpc/home/guodong.li/:/workspaces:rw \
deepspeed.sif


export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond0 && export CC=/opt/hpcx/ompi/bin/mpicc && deepspeed --include localhost:4,5,6,7 --master_port 29001 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```

### 單機多卡+singularity+slurm 


```
sbatch pp-standalone-singularity.slurm


squeue
scancel -v xx
```


---


```
srun --mpi=list 
```




### 多機多卡+singularity+slurm 


- --mpi：指定mpi型別為pmi2

```
sbatch pp-multinode-singularity.slurm
```








