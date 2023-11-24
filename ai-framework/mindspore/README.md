

- https://ascendhub.huawei.com/#/detail/ascend-mindspore





```
docker login -u 15708484031 ascendhub.huawei.com


docker pull ascendhub.huawei.com/public-ascendhub/ascend-mindspore:23.0.RC2-centos7
```


```
chmod +x Ascend-docker-runtime_{version}_linux-{arch}.run
./Ascend-docker-runtime_{version}_linux-{arch}.run  --install



docker run -it -e ASCEND_VISIBLE_DEVICES=0 映象id  /bin/bash

bash test_model.sh

```











```
docker run -dt --name mindspore_env --restart=always --gpus all \
--network=host \
--shm-size 4G \
-v /home/guodong/workspace:/workspace \
-w /workspace \
ascendhub.huawei.com/public-ascendhub/ascend-mindspore:23.0.RC2-centos7 \
/bin/bash

docker exec -it mindspore_env bash
```







```
FROM ascendhub.huawei.com/public-ascendhub/ascend-mindspore:23.0.RC1-ubuntu18.04

USER root
COPY ./glm0609 /home/chatglm6b
COPY ./hccl_tools /home/hccl_tools
COPY ./mindpet /home/mindpet
COPY ./web_demo.gif /home/

WORKDIR /home

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -U pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ 

RUN wget https://sqlite.org/2019/sqlite-autoconf-3290000.tar.gz
RUN tar zxvf sqlite-autoconf-3290000.tar.gz && cd sqlite-autoconf-3290000 && ./configure && make -j${proc} && make install

RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
RUN tar xzvf Python-3.7.5.tgz && cd Python-3.7.5 && \
./configure LDFLAGS="-L/usr/local/lib" CPPFLAGS="-I/usr/local/include" --prefix=/usr/local/python3.7.5 && \
make -j${proc} && \
make install

RUN /usr/local/python3.7.5/bin/pip install torch transformers numpy sentencepiece ftfy regex tqdm \
pyyaml rouge_chinese nltk jieba datasets gradio==3.23.0 pandas openpyxl et-xmlfile mdtex2html

ENV LC_ALL='C.UTF-8'
ENV PYTHONPATH=/home/mindpet/:$PYTHONPATH
ENV LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1
```



```
docker run -it -u root --ipc=host \
               --device=/dev/davinci0 \
               --device=/dev/davinci1 \
               --device=/dev/davinci2 \
               --device=/dev/davinci3 \
               --device=/dev/davinci4 \
               --device=/dev/davinci5 \
               --device=/dev/davinci6 \
               --device=/dev/davinci7 \
               --device=/dev/davinci_manager \
               --device=/dev/devmm_svm \
               --device=/dev/hisi_hdc \
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /var/log/npu/:/usr/slog \
               swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag} \
               /bin/bash
```



```
docker run -it --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /var/log/npu/:/usr/slog \
ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag} \
/bin/bash
```


```
docker run -itd --name mindspore_dev -u root --network=host --ipc=host \
            --device=/dev/davinci0 \
            --device=/dev/davinci1 \
            --device=/dev/davinci2 \
            --device=/dev/davinci3 \
            --device=/dev/davinci4 \
            --device=/dev/davinci5 \
            --device=/dev/davinci6 \
            --device=/dev/davinci7 \
            --device=/dev/davinci_manager \
            --device=/dev/devmm_svm \
            --device=/dev/hisi_hdc \
            -v /root/ChatGLM6B_MS_0613/huggingface-glm-6b:/home/huggingface-glm-6b \
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
            -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
            -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
            -v /var/log/npu/:/usr/slog \
            chatglm6b:v1 \
            /bin/bash


docker exec -it  mindspore_dev bash

```










## 安裝(mac m1)

```
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/cpu/aarch64/mindspore-2.2.0-cp39-cp39-macosx_11_0_arm64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```





## mindformers(docker)



```
docker pull --platform=arm64 swr.cn-central-221.ovaijisuan.com/mindformers/mindformers0.8.0_mindspore2.2.0:aarch_20231025
```


```
# --device用於控制指定容器的執行NPU卡號和範圍
# -v 用於對映容器外的目錄
# --name 用於自定義容器名稱

docker run -it -u root \
--ipc=host \
--network host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--name {請手動輸入容器名稱} \
swr.cn-central-221.ovaijisuan.com/mindformers/mindformers0.8.0_mindspore2.2.0:aarch_20231025 \
/bin/bash
```









