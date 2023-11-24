<p align="center">
  <img src="https://github.com/liguodongiot/llm-action/blob/main/pic/llm-action.png" >
</p>

## 目錄

- 🔥 [LLM訓練](#llm訓練)
  - 🐫 [LLM訓練實戰](#llm訓練實戰)
  - 🐼 [LLM引數高效微調技術原理綜述](#llm微調技術原理)
  - 🐰 [LLM引數高效微調技術實戰](#llm微調實戰)
  - 🐘 [LLM分散式訓練並行技術](#llm分散式訓練並行技術)
  - 🌋 [分散式AI框架](#分散式ai框架)
  - 📡 [分散式訓練網路通訊](#分散式訓練網路通訊)
- 🐎 [LLM推理](#llm推理)
  - 🚀 [LLM推理框架](#llm推理框架)
  - ✈️ [LLM推理最佳化技術](#llm推理最佳化技術)
- ♻️ [LLM壓縮](#llm壓縮)
  - 📐 [LLM量化](#llm量化)
  - 🔰 [LLM剪枝](#llm剪枝)
  - 💹 [LLM知識蒸餾](#llm知識蒸餾)
  - ♑️ [低秩分解](#低秩分解)
- ♍️ [LLM演算法架構](#llm演算法架構)
- :jigsaw: [LLM應用開發](#llm應用開發)
- 🀄️ [LLM國產化適配](#llm國產化適配)
- 🔯 [AI編譯器](#ai編譯器)
- 🔘 [AI基礎設施](#ai基礎設施)
- 💟 [LLMOps](#llmops)
- 🍄 [LLM生態相關技術](#llm生態相關技術)
- 🔨 [伺服器基礎環境安裝及常用工具](#伺服器基礎環境安裝及常用工具)
- 💬 [LLM學習交流群](#llm學習交流群)
- 👥 [微信公眾號](#微信公眾號)
- ⭐️ [Star History](#star-history)

## LLM訓練

### LLM訓練實戰

下面彙總了我在大模型實踐中訓練相關的所有教程。從6B到65B，從全量微調到高效微調（LoRA，QLoRA，P-Tuning v2），再到RLHF（基於人工反饋的強化學習）。

| LLM                         | 預訓練/SFT/RLHF...            | 引數     | 教程                                                                                                                                                                                                                     | 程式碼                                                                                     |
| --------------------------- | ----------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Alpaca                      | full fine-turning             | 7B       | [從0到1復現斯坦福羊駝（Stanford Alpaca 7B）](https://zhuanlan.zhihu.com/p/618321077)                                                                                                                                        | [配套程式碼](https://github.com/liguodongiot/llm-action/tree/main/train/alpaca)               |
| Alpaca(LLaMA)               | LoRA                          | 7B~65B   | 1.[足夠驚豔，使用Alpaca-Lora基於LLaMA(7B)二十分鐘完成微調，效果比肩斯坦福羊駝](https://zhuanlan.zhihu.com/p/619426866)<br>2. [使用 LoRA 技術對 LLaMA 65B 大模型進行微調及推理](https://zhuanlan.zhihu.com/p/632492604)    | [配套程式碼](https://github.com/liguodongiot/llm-action/tree/main/train/alpaca-lora)          |
| BELLE(LLaMA/Bloom)          | full fine-turning             | 7B       | 1.[基於LLaMA-7B/Bloomz-7B1-mt復現開源中文對話大模型BELLE及GPTQ量化](https://zhuanlan.zhihu.com/p/618876472) <br> 2. [BELLE(LLaMA-7B/Bloomz-7B1-mt)大模型使用GPTQ量化後推理效能測試](https://zhuanlan.zhihu.com/p/621128368) | N/A                                                                                      |
| ChatGLM                     | LoRA                          | 6B       | [從0到1基於ChatGLM-6B使用LoRA進行引數高效微調](https://zhuanlan.zhihu.com/p/621793987)                                                                                                                                      | [配套程式碼](https://github.com/liguodongiot/llm-action/tree/main/train/chatglm-lora)         |
| ChatGLM                     | full fine-turning/P-Tuning v2 | 6B       | [使用DeepSpeed/P-Tuning v2對ChatGLM-6B進行微調](https://zhuanlan.zhihu.com/p/622351059)                                                                                                                                     | [配套程式碼](https://github.com/liguodongiot/llm-action/tree/main/train/chatglm)              |
| Vicuna(LLaMA)               | full fine-turning             | 7B       | [大模型也內卷，Vicuna訓練及推理指南，效果碾壓斯坦福羊駝](https://zhuanlan.zhihu.com/p/624012908)                                                                                                                            | N/A                                                                                      |
| OPT                         | RLHF                          | 0.1B~66B | 1.[一鍵式 RLHF 訓練 DeepSpeed Chat（一）：理論篇](https://zhuanlan.zhihu.com/p/626159553) <br> 2. [一鍵式 RLHF 訓練 DeepSpeed Chat（二）：實踐篇](https://zhuanlan.zhihu.com/p/626214655)                                 | [配套程式碼](https://github.com/liguodongiot/llm-action/tree/main/train/deepspeedchat)        |
| MiniGPT-4(LLaMA)            | full fine-turning             | 7B       | [大殺器，多模態大模型MiniGPT-4入坑指南](https://zhuanlan.zhihu.com/p/627671257)                                                                                                                                             | N/A                                                                                      |
| Chinese-LLaMA-Alpaca(LLaMA) | LoRA（預訓練+微調）           | 7B       | [中文LLaMA&amp;Alpaca大語言模型詞表擴充+預訓練+指令精調](https://zhuanlan.zhihu.com/p/631360711)                                                                                                                            | [配套程式碼](https://github.com/liguodongiot/llm-action/tree/main/train/chinese-llama-alpaca) |
| LLaMA                       | QLoRA                         | 7B/65B   | [高效微調技術QLoRA實戰，基於LLaMA-65B微調僅需48G視訊記憶體，真香](https://zhuanlan.zhihu.com/p/636644164)                                                                                                                         | [配套程式碼](https://github.com/liguodongiot/llm-action/tree/main/train/qlora)                |

**[⬆ 一鍵返回目錄](#目錄)**

### LLM微調技術原理

對於普通大眾來說，進行大模型的預訓練或者全量微調遙不可及。由此，催生了各種引數高效微調技術，讓科研人員或者普通開發者有機會嘗試微調大模型。

因此，該技術值得我們進行深入分析其背後的機理，本系列大體分七篇文章進行講解。

- [大模型引數高效微調技術原理綜述（一）-背景、引數高效微調簡介](https://zhuanlan.zhihu.com/p/635152813)
- [大模型引數高效微調技術原理綜述（二）-BitFit、Prefix Tuning、Prompt Tuning](https://zhuanlan.zhihu.com/p/635686756)
- [大模型引數高效微調技術原理綜述（三）-P-Tuning、P-Tuning v2](https://zhuanlan.zhihu.com/p/635848732)
- [大模型引數高效微調技術原理綜述（四）-Adapter Tuning及其變體](https://zhuanlan.zhihu.com/p/636038478)
- [大模型引數高效微調技術原理綜述（五）-LoRA、AdaLoRA、QLoRA](https://zhuanlan.zhihu.com/p/636215898)
- [大模型引數高效微調技術原理綜述（六）-MAM Adapter、UniPELT](https://zhuanlan.zhihu.com/p/636362246)
- [大模型引數高效微調技術原理綜述（七）-最佳實踐、總結](https://zhuanlan.zhihu.com/p/649755252)

### LLM微調實戰

下面給大家分享**大模型引數高效微調技術實戰**，該系列主要針對 HuggingFace PEFT 框架支援的一些高效微調技術進行講解，共6篇文章。

| 教程                                                                                                | 程式碼                                                                                                      | 框架             |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------- |
| [大模型引數高效微調技術實戰（一）-PEFT概述及環境搭建](https://zhuanlan.zhihu.com/p/651744834)          | N/A                                                                                                       | HuggingFace PEFT |
| [大模型引數高效微調技術實戰（二）-Prompt Tuning](https://zhuanlan.zhihu.com/p/646748939)               | [配套程式碼](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_prompt_tuning_clm.ipynb) | HuggingFace PEFT |
| [大模型引數高效微調技術實戰（三）-P-Tuning](https://zhuanlan.zhihu.com/p/646876256)                    | [配套程式碼](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_p_tuning_clm.ipynb)      | HuggingFace PEFT |
| [大模型引數高效微調技術實戰（四）-Prefix Tuning / P-Tuning v2](https://zhuanlan.zhihu.com/p/648156780) | [配套程式碼](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_p_tuning_v2_clm.ipynb)   | HuggingFace PEFT |
| [大模型引數高效微調技術實戰（五）-LoRA](https://zhuanlan.zhihu.com/p/649315197)                        | [配套程式碼](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_lora_clm.ipynb)          | HuggingFace PEFT |
| [大模型引數高效微調技術實戰（六）-IA3](https://zhuanlan.zhihu.com/p/649707359)                         | [配套程式碼](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_ia3_clm.ipynb)           | HuggingFace PEFT |

**[⬆ 一鍵返回目錄](#目錄)**

### [LLM分散式訓練並行技術](https://github.com/liguodongiot/llm-action/tree/main/docs/llm-base/distribution-parallelism)

近年來，隨著Transformer、MOE架構的提出，使得深度學習模型輕鬆突破上萬億規模引數，傳統的單機單卡模式已經無法滿足超大模型進行訓練的要求。因此，我們需要基於單機多卡、甚至是多機多卡進行分散式大模型的訓練。

而利用AI叢集，使深度學習演算法更好地從大量資料中高效地訓練出效能優良的大模型是分散式機器學習的首要目標。為了實現該目標，一般需要根據硬體資源與資料/模型規模的匹配情況，考慮對計算任務、訓練資料和模型進行劃分，從而進行分散式訓練。因此，分散式訓練相關技術值得我們進行深入分析其背後的機理。

下面主要對大模型進行分散式訓練的並行技術進行講解，本系列大體分九篇文章進行講解。

- [大模型分散式訓練並行技術（一）-概述](https://zhuanlan.zhihu.com/p/598714869)
- [大模型分散式訓練並行技術（二）-資料並行](https://zhuanlan.zhihu.com/p/650002268)
- [大模型分散式訓練並行技術（三）-流水線並行](https://zhuanlan.zhihu.com/p/653860567)
- [大模型分散式訓練並行技術（四）-張量並行](https://zhuanlan.zhihu.com/p/657921100)
- [大模型分散式訓練並行技術（五）-序列並行](https://zhuanlan.zhihu.com/p/659792351)
- [大模型分散式訓練並行技術（六）-多維混合並行](https://zhuanlan.zhihu.com/p/661279318)
- [大模型分散式訓練並行技術（七）-自動並行](https://zhuanlan.zhihu.com/p/662517647)
- [大模型分散式訓練並行技術（八）-MOE並行](https://zhuanlan.zhihu.com/p/662518387)
- [大模型分散式訓練並行技術（九）-總結](https://juejin.cn/post/7290740395913969705)

**[⬆ 一鍵返回目錄](#目錄)**

### 分散式AI框架

- [PyTorch](https://github.com/liguodongiot/llm-action/tree/main/train/pytorch/)
  - PyTorch 單機多卡訓練
  - PyTorch 多機多卡訓練
- [Megatron-LM](https://github.com/liguodongiot/llm-action/tree/main/train/megatron)
  - Megatron-LM 單機多卡訓練
  - Megatron-LM 多機多卡訓練
  - [基於Megatron-LM從0到1完成GPT2模型預訓練、模型評估及推理](https://juejin.cn/post/7259682893648724029)
- [DeepSpeed](https://github.com/liguodongiot/llm-action/tree/main/train/deepspeed)
  - DeepSpeed 單機多卡訓練
  - DeepSpeed 多機多卡訓練
- [Megatron-DeepSpeed](https://github.com/liguodongiot/llm-action/tree/main/train/megatron-deepspeed)
  - 基於 Megatron-DeepSpeed 從 0 到1 完成 LLaMA 預訓練
  - 基於 Megatron-DeepSpeed 從 0 到1 完成 Bloom 預訓練





**[⬆ 一鍵返回目錄](#目錄)**

## [LLM推理](https://github.com/liguodongiot/llm-action/tree/main/inference)

### LLM推理框架

- [大模型推理框架概述](https://www.zhihu.com/question/625415776/answer/3243562246)
- [大模型的好夥伴，淺析推理加速引擎FasterTransformer](https://zhuanlan.zhihu.com/p/626008090)
- [模型推理服務化框架Triton保姆式教程（一）：快速入門](https://zhuanlan.zhihu.com/p/629336492)
- [模型推理服務化框架Triton保姆式教程（二）：架構解析](https://zhuanlan.zhihu.com/p/634143650)
- [模型推理服務化框架Triton保姆式教程（三）：開發實踐](https://zhuanlan.zhihu.com/p/634444666)
- [TensorRT-LLM保姆級教程（一）-快速入門](https://zhuanlan.zhihu.com/p/666849728)
- [TensorRT-LLM保姆級教程（二）-開發實踐](https://zhuanlan.zhihu.com/p/667572720)
- TensorRT-LLM保姆級教程（三）-基於Triton完成模型服務化
- TensorRT-LLM保姆級教程（四）-新模型適配


### LLM推理最佳化技術

- [LLM推理最佳化技術概述]()
- PageAttention
- FlashAttention

## LLM壓縮

- 模型壓縮技術原理（一）：知識蒸餾 
- 模型壓縮技術原理（二）：模型量化
- 模型壓縮技術原理（三）：模型剪枝 


### [LLM量化](https://github.com/liguodongiot/llm-action/tree/main/model-compression/quantization)

- [大模型量化概述](https://www.zhihu.com/question/627484732/answer/3261671478)

訓練後量化：

- SmoothQuant
- ZeroQuant
- GPTQ
- LLM.int8()
- AWQ


量化感知訓練：

- [大模型量化感知訓練開山之作：LLM-QAT](https://zhuanlan.zhihu.com/p/647589650)

量化感知微調：

- QLoRA
- PEQA

### LLM剪枝

**結構化剪枝**：

- LLM-Pruner

**非結構化剪枝**：

- SparseGPT
- LoRAPrune
- Wanda

### LLM知識蒸餾

- [大模型知識蒸餾概述](https://www.zhihu.com/question/625415893/answer/3243565375)

**Standard KD**:

使學生模型學習教師模型(LLM)所擁有的常見知識，如輸出分佈和特徵資訊，這種方法類似於傳統的KD。

- MINILLM
- GKD

**EA-based KD**:

不僅僅是將LLM的常見知識轉移到學生模型中，還涵蓋了蒸餾它們獨特的湧現能力。具體來說，EA-based KD又分為了上下文學習（ICL）、思維鏈（CoT）和指令跟隨（IF）。

In-Context Learning：

- In-Context Learning distillation

Chain-of-Thought：

- MT-COT
- Fine-tune-CoT
- DISCO
- SCOTT
- SOCRATIC CoT

Instruction Following：

- Lion

### 低秩分解

低秩分解旨在透過將給定的權重矩陣分解成兩個或多個較小維度的矩陣，從而對其進行近似。低秩分解背後的核心思想是找到一個大的權重矩陣W的分解，得到兩個矩陣U和V，使得W≈U V，其中U是一個m×k矩陣，V是一個k×n矩陣，其中k遠小於m和n。U和V的乘積近似於原始的權重矩陣，從而大幅減少了引數數量和計算開銷。

在LLM研究的模型壓縮領域，研究人員通常將多種技術與低秩分解相結合，包括修剪、量化等。

- ZeroQuant-FP（低秩分解+量化）
- LoRAPrune（低秩分解+剪枝）

## [LLM演算法](https://github.com/liguodongiot/llm-action/tree/main/docs/llm-base/ai-algo)

- [大模型演算法演進](https://zhuanlan.zhihu.com/p/600016134)
- ChatGLM / ChatGLM2 / ChatGLM3 大模型解析
- Bloom 大模型解析
- LLaMA / LLaMA2 大模型解析
- [百川智慧開源大模型baichuan-7B技術剖析](https://www.zhihu.com/question/606757218/answer/3075464500)
- [百川智慧開源大模型baichuan-13B技術剖析](https://www.zhihu.com/question/611507751/answer/3114988669)


## [LLM國產化適配](https://github.com/liguodongiot/llm-action/tree/main/docs/llm_localization)

隨著 ChatGPT 的現象級走紅，引領了AI大模型時代的變革，從而導致 AI 算力日益緊缺。與此同時，中美貿易戰以及美國對華進行AI晶片相關的制裁導致 AI 算力的國產化適配勢在必行。本系列將對一些國產化 AI 加速卡進行講解。

- [大模型國產化適配1-華為昇騰AI全棧軟硬體平臺總結](https://zhuanlan.zhihu.com/p/637918406)
- [大模型國產化適配2-基於昇騰910使用ChatGLM-6B進行模型推理](https://zhuanlan.zhihu.com/p/650730807)
- [大模型國產化適配3-基於昇騰910使用ChatGLM-6B進行模型訓練](https://zhuanlan.zhihu.com/p/651324599)
- [大模型國產化適配4-基於昇騰910使用LLaMA-13B進行多機多卡訓練](https://juejin.cn/post/7265627782712901686)
- [大模型國產化適配5-百度飛漿PaddleNLP大語言模型工具鏈總結](https://juejin.cn/post/7291513759470960679)

**[⬆ 一鍵返回目錄](#目錄)**

## LLM應用開發

大模型是基座，要想讓其變成一款產品，我們還需要一些其他相關的技術，比如：向量資料庫（Pinecone、Milvus、Vespa、Weaviate），LangChain等。

- [雲原生向量資料庫Milvus（一）-簡述、系統架構及應用場景](https://zhuanlan.zhihu.com/p/476025527)
- [雲原生向量資料庫Milvus（二）-資料與索引的處理流程、索引型別及Schema](https://zhuanlan.zhihu.com/p/477231485)
- [關於大模型驅動的AI智慧體Agent的一些思考](https://zhuanlan.zhihu.com/p/651921120)


## AI編譯器

AI編譯器是指將機器學習演算法從開發階段，透過變換和最佳化演算法，使其變成部署狀態。

- [AI編譯器技術原理（一）-概述]()
- [AI編譯器技術原理（二）-編譯器前端]()
- [AI編譯器技術原理（三）-編譯器後端]()


框架：
- TVM
- MLIR
- TensorRT



## AI基礎設施

### AI加速卡

- [AI晶片技術原理剖析（一）：國內外AI晶片概述](https://zhuanlan.zhihu.com/p/667686665)
- AI晶片技術原理剖析（二）：英偉達GPU 
- AI晶片技術原理剖析（三）：谷歌TPU

### AI叢集

待更新...


### [AI叢集網路通訊](https://github.com/liguodongiot/llm-action/tree/main/docs/llm-base/network-communication)

待更新...

- 分散式訓練網路通訊原語
- AI 叢集通訊軟硬體


## LLMOps

待更新...

## LLM生態相關技術

- [大模型詞表擴充必備工具SentencePiece](https://zhuanlan.zhihu.com/p/630696264)
- [大模型實踐總結](https://www.zhihu.com/question/601594836/answer/3032763174)
- [ChatGLM 和 ChatGPT 的技術區別在哪裡？](https://www.zhihu.com/question/604393963/answer/3061358152)
- [現在為什麼那麼多人以清華大學的ChatGLM-6B為基座進行試驗？](https://www.zhihu.com/question/602504880/answer/3041965998)
- [為什麼很多新發布的大模型預設使用BF16而不是FP16？](https://www.zhihu.com/question/616600181/answer/3195333332)

**[⬆ 一鍵返回目錄](#目錄)**

## 伺服器基礎環境安裝及常用工具

基礎環境安裝：

- [英偉達A800加速卡常見軟體包安裝命令](https://github.com/liguodongiot/llm-action/blob/main/docs/llm-base/a800-env-install.md)
- [英偉達H800加速卡常見軟體包安裝命令](https://github.com/liguodongiot/llm-action/blob/main/docs/llm-base/h800-env-install.md)
- [昇騰910加速卡常見軟體包安裝命令](https://github.com/liguodongiot/llm-action/blob/main/docs/llm_localization/ascend910-env-install.md)

常用工具：

- [Linux 常見命令大全](https://juejin.cn/post/6992742028605915150)
- [Conda 常用命令大全](https://juejin.cn/post/7089093437223338015)
- [Poetry 常用命令大全](https://juejin.cn/post/6999405667261874183)
- [Docker 常用命令大全](https://juejin.cn/post/7016238524286861325)
- [Docker Dockerfile 指令大全](https://juejin.cn/post/7016595442062327844)
- [Kubernetes 常用命令大全](https://juejin.cn/post/7031201391553019911)
- [叢集環境 GPU 管理和監控工具 DCGM 常用命令大全](https://github.com/liguodongiot/llm-action/blob/main/docs/llm-base/dcgmi.md)

## LLM學習交流群

我建立了大模型學習交流群，供大家一起學習交流大模型相關的最新技術，目前已有5個群，每個群都有上百人的規模，**可加我微信進群**（加微信請備註來意，如：進大模型學習交流群+GitHub）。**一定要備註喲，否則不予透過**。

PS：**成都有個本地大模型交流群，想進可以另外單獨備註下。**

<p align="center">
  <img src="https://github.com/liguodongiot/llm-action/blob/main/pic/wx.jpg">
</p>

## 微信公眾號

微信公眾號：**吃果凍不吐果凍皮**，該公眾號主要分享AI工程化（大模型、MLOps等）相關實踐經驗，免費電子書籍、論文等。

<p align="center">
  <img src="https://github.com/liguodongiot/llm-action/blob/main/pic/wx-gzh.png" >
</p>

**[⬆ 一鍵返回目錄](#目錄)**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=liguodongiot/llm-action&type=Date)](https://star-history.com/#liguodongiot/llm-action&Date)
