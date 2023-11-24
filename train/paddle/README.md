



```
nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4

docker run -dt \
--name paddle \
--restart=always \
--gpus all \
--network=host \
--shm-size 4G \
-v /home/guodong.li/workspace:/paddle \
registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4 \
/bin/bash
```

```
docker exec -it paddle bash
```


```
sudo docker run -it --rm \
--gpus all \
--network=host \
--shm-size 4G \
-v /home/guodong.li/workspace:/workspace \
registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4 \
/bin/bash
```



CUDA 工具包 11.7 配合 cuDNN v8.4.1, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.4.2.4

如需使用分散式多卡環境，需配合 NCCL>=2.7

GPU 運算能力超過 3.5 的硬體裝置


```
conda create -n paddle python=3.8 -y
conda activate paddle
```



```
python -m pip install paddlepaddle-gpu==2.5.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```


```
import paddle
paddle.utils.run_check()
```


```
python3 -m pip uninstall paddlepaddle-gpu
```



---



## 安裝develop版本


```
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html

pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
sudo pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

```












