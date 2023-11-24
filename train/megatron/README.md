




## 基於Megatron-LM實現的專案

- [CodeGeeX](https://github.com/THUDM/CodeGeeX)

- [如何使用 Megatron-LM 訓練語言模型](https://huggingface.co/blog/zh/megatron-training)：資料預處理，訓練，模型轉換，推理等







### 資料載入

Megatron-LM 帶有一個高效的 DataLoader，其中資料在訓練前被 tokenize 和 shuffle。它還將資料拆分為帶有索引的編號序列，並將索引儲存，因此 tokenize 只需要計算一次。為了構建索引，首先根據訓練引數計算每個 epoch 的數量，並建立一個排序，然後對資料進行 shuffle 操作。這與大多數情況不同，我們通常迭代整個資料集直到其用盡，然後重複第二個 epoch 。這平滑了學習曲線並節省了訓練時間。


### 融合 CUDA 核心
當一個計算在 GPU 上執行時，必要的資料會從記憶體中取出並載入到 GPU 上，然後計算結果被儲存回記憶體。簡單來說，融合核心的思想是: 將通常由 PyTorch 單獨執行的類似操作組合成一個單獨的硬體操作。因此可以將多個離散計算合併為一個，從而減少在多個離散計算中的記憶體移動次數。


當 f、g 和 h 融合在一個核心中時，f 和 g 的中間結果 x' 和 y' 儲存在 GPU 暫存器中並立即被 h 使用。但是如果不融合，x' 和 y' 就需要複製到記憶體中，然後由 h 載入。因此，融合 CUDA 核心顯著加快了計算速度。此外，Megatron-LM 還使用 Apex 的 AdamW 融合實現，它比 PyTorch 實現更快。

雖然我們可以在 transformers 中自定義 Megatron-LM 中的 DataLoader 和 Apex 的融合最佳化器，但自定義融合 CUDA 核心對新手來說太不友好了。


