


- https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py


-更強大的效能：ChatGLM2-6B 使用了 GLM 的混合目標函式，經過了 1.4T 中英識別符號的預訓練與人類偏好對齊訓練。
- 更長的上下文：基於 FlashAttention 技術，我們將基座模型的上下文長度（Context Length）由 ChatGLM-6B 的 2K 擴充套件到了 32K，並在對話階段使用 8K 的上下文長度訓練。對於更長的上下文，我們釋出了 ChatGLM2-6B-32K 模型。
- 更高效的推理：基於 Multi-Query Attention 技術，ChatGLM2-6B 有更高效的推理速度和更低的視訊記憶體佔用。








## 說明


- F.silu：
- RMSNorm





## chatglm 與 chatglm2 不同支援


- 啟用函式不同
- RotaryEmbedding 位置不同。