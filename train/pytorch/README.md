

- 映象：https://hub.docker.com/r/pytorch/pytorch


- https://github.com/pytorch/examples
- https://github.com/pytorch/examples.git
- https://github.com/pytorch/pytorch

---



- torch.distributed.get_rank() # 取得當前程序的全域性序號
- torch.distributed.get_world_size() # 取得全域性程序的個數
- torch.cuda.set_device(device) # 為當前程序分配GPU
- torch.distributed.new_group(ranks) # 設定組
- torch.cuda.current_device()


---








---


- https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- 



## PyTorch 分散式訓練



- PyTorch 分散式訓練（一）：概述
- PyTorch 分散式訓練（二）：資料並行
- PyTorch 分散式訓練（三）：分散式自動微分
- PyTorch 分散式訓練（四）：分散式最佳化器
- PyTorch 分散式訓練（五）：分散式 RPC 框架
- 






## 問題排查


- 將環境變數 NCCL_DEBUG 設定為 INFO 以列印有助於診斷問題的詳細日誌。（export NCCL_DEBUG=INFO）
- 顯式設定網路介面。（export NCCL_SOCKET_IFNAME=eth0）


