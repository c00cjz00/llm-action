


## 演算法

CLIP

BLIP


BLIP2 

LLaVA 

miniGPT4

InstructBLIP


MDETR



### Stable Diffusion  

擴散模型 ， 多模態任務：文生圖 圖生圖

- https://huggingface.co/docs/peft/task_guides/dreambooth_lora
- https://github.com/huggingface/peft/tree/v0.6.2/examples/lora_dreambooth
- 推理：https://github.com/huggingface/peft/blob/v0.6.2/examples/lora_dreambooth/lora_dreambooth_inference.ipynb
- 資料：https://huggingface.co/datasets/diffusers/docs-images

支援多種微調技術：LoRA、	LoHa、	LoKr


- 擴散模型庫：https://huggingface.co/docs/diffusers/tutorials/tutorial_overview









### Blip2 
image to text (Multi-modal models)

- 微調：https://github.com/huggingface/peft/blob/v0.6.2/examples/int8_training/fine_tune_blip2_int8.py
- 模型及示例：https://huggingface.co/Salesforce/blip2-opt-2.7b
- 使用 BLIP-2 零樣本“圖生文：https://huggingface.co/blog/zh/blip-2







## 任務

文生圖，還能實現圖生文、圖文聯合生成、無條件圖文生成、圖文改寫


文生圖(Generation) （文字->影象）
視覺問答(Visual Question Answering) (影象+文字 ->文字)
多模態分類 (Multimodal classification) (影象+文字 -> 標籤)
最佳化理解/生成(Better understanding/generation) (影象+文字 ->標籤/文字)

零樣本影象描述生成


通用視覺問答

文字導向的視覺問答

細粒度視覺定位



例如給定一張圖片，可以完成以下任務：

一、VQA（Visual Question Answering）視覺問答輸入：一張圖片、一個自然語言描述的問題輸出：答案（單詞或短語）

二、Image Caption 影象字幕輸入：一張圖片輸出：圖片的自然語言描述（一個句子）

三、Referring Expression Comprehension 指代表達輸入：一張圖片、一個自然語言描述的句子輸出：判斷句子描述的內容（正確或錯誤）

四、Visual Dialogue 視覺對話輸入：一張圖片輸出：兩個角色進行多次互動、對話

五、VCR (Visual Commonsense Reasoning) 視覺常識推理輸入：1個問題，4個備選答案，4個理由輸出：正確答案，和理由


六、NLVR(Natural Language for Visual Reasoning)自然語言視覺推理

輸入：2張圖片，一個分佈

輸出：true或false



七、Visual Entailment 視覺蘊含

輸入：影象、文字

輸出：3種label的機率。（entailment、neutral、contradiction）蘊含、中性、矛盾




八、Image-Text Retrieval 圖文檢索

有3種方式。

1）以圖搜文。輸入圖片，輸出文字

2）以文搜圖。輸入文字，輸出圖片

3）以圖搜圖，輸入圖片，輸出圖片








## 多模態通用模型 FLAVA

https://github.com/facebookresearch/multimodal/tree/main/examples/flava














