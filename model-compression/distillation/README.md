
近年來，隨著Transformer、MOE架構的提出，使得深度學習模型輕鬆突破上萬億規模引數，從而導致模型變得越來越大，因此，我們需要一些大模型壓縮技術來降低模型部署的成本，並提升模型的推理效能。而大模型壓縮主要分為如下幾類：

-   剪枝（Pruning）
-   知識蒸餾（Knowledge Distillation）
-   量化（Quantization）
-   低秩分解（Low-Rank Factorization）

下面主要針對大模型蒸餾技術進行相應的講解，本系列一共分六篇文章進行講解。

- 大模型知識蒸餾原理綜述（一）：概述
- 大模型知識蒸餾原理綜述（二）：MINILLM、GDK
- 大模型知識蒸餾原理綜述（三）：In-Context Learning distillation 
- 大模型知識蒸餾原理綜述（四）：SCOTT、DISCO、MT-COT
- 大模型知識蒸餾原理綜述（五）：Lion
- 大模型知識蒸餾原理綜述（六）：總結

本文為大模型知識蒸餾原理綜述第一篇，主要講述當前大模型蒸餾相關的一些工作。






## 知識蒸餾簡介


知識蒸餾，也被稱為教師-學生神經網路學習演算法，已經受到業界越來越多的關注。大型深度網路在實踐中往往會獲得良好的效能，因為當考慮新資料時，過度引數化會提高泛化效能。在知識蒸餾中，小網路（學生網路）通常是由一個大網路（教師網路）監督，演算法的關鍵問題是如何將教師網路的知識傳授給學生網路。通常把一個全新的更深的更窄結構的深度神經網路當作學生神經網路，然後把一個預先訓練好的神經網路模型當作教師神經網路。

Hinton等人([@Distill])首先提出了教師神經網路-學生神經網路學習框架，透過最小化兩個神經網路之間的差異來學習一個更窄更深的神經網路。記教師神經網路為
，它的引數為
，同時記學生神經網路為
，相應的引數為
。一般而言，學生神經網路相較於教師神經網路具有更少的引數。

文獻([@Distill])提出的知識蒸餾（knowledge distillation，KD）方法，同時令學生神經網路的分類結果接近真實標籤並且令學生神經網路的分類結果接近於教師神經網路的分類結果，即，

(10.3.5)
其中，
是交叉熵函式，
和
分別是學生網路和教師網路的輸出，
是標籤。公式 (10.3.5)中的第一項使得學生神經網路的分類結果接近預期的真實標籤，而第二項的目的是提取教師神經網路中的有用資訊並傳遞給學生神經網路，
是一個權值引數用來平衡兩個目標函式。
是一個軟化（soften）函式，將網路輸出變得更加平滑。

公式 (10.3.5)僅僅從教師神經網路分類器輸出的資料中提取有價值的資訊，並沒有從其他中間層去將教師神經網路的資訊進行挖掘。因此，Romero等人[@FitNet]）進一步地開發了一種學習輕型學生神經網路的方法，該演算法可以從教師神經網路中任意的一層來傳遞有用的資訊給學生神經網路。此外，事實上，並不是所有的輸入資料對卷積神經網路的計算和完成後續的任務都是有用的。例如，在一張包含一個動物的影象中，對分類和識別結果比較重要的是動物所在的區域，而不是那些無用的背景資訊。所以，有選擇性地從教師神經網路的特徵圖中提取資訊是一個更高效的方式。於是，Zagoruyko和Komodakis（[@attentionTS]）提出了一種基於感知（Attention）損失函式的學習方法來提升學生神經網路的效能，該方法在學習學生神經網路的過程中，引入了感知模組（Attention），選擇性地將教師神經網路中的資訊傳遞給學生神經網路，並幫助其進行訓練。感知圖用來表達輸入影象不同位置對最終分類結果的重要性。感知模組從教師網路生成感知圖，並遷移到學生網路，如圖


圖10.3.3 一種基於感知（attention）的教師神經網路-學生神經網路學習演算法

知識蒸餾是一種有效的幫助小網路最佳化的方法，能夠進一步和剪枝、量化等其他壓縮方法結合，訓練得到精度高、計算量小的高效模型。



標準知識蒸餾

基於湧現能力的知識蒸餾
- 上下文學習
- 思維鏈
- 指令遵循



