

# chatglm-6b

- https://huggingface.co/THUDM/chatglm-6b/tree/main
- https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py


自迴歸填空





ChatGLM藉助編碼器-解碼器架構思想，前半部分採用類似於Bert的雙向注意力進行掩碼，後半部分採用類似於GPT的自迴歸架構進行預測。







說明：

- gelu
- LayerNorm



- 重新排列了LN和殘差連線的順序，具體來講就是將Post-LN改成Pre-LN。
- 使用一個線性層來預測輸出詞；
- 將ReLU啟用函式替換為GeLU啟用函式。