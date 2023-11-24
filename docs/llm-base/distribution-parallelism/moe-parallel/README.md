

- https://github.com/laekov/fastmoe
- SmartMoE: https://github.com/zms1999/SmartMoE





- 飛漿-MOE：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/moe_cn.html
- 
- https://blog.csdn.net/qq_41185868/article/details/103219988

- [GShard-MoE](https://arxiv.org/abs/2006.16668)





GShard，Switch-Transformer， GLaM



- Mixture-of-Experts (MoE) 經典論文一覽：https://zhuanlan.zhihu.com/p/542465517


```
GShard，按照文章的說法，是第一個將MoE的思想拓展到Transformer上的工作。
具體的做法是，把Transformer的encoder和decoder中，每隔一個（every other）的FFN層，替換成position-wise 的 MoE層，使用的都是 Top-2 gating network。




跟其他MoE模型的一個顯著不同就是，Switch Transformer 的 gating network 每次只 route 到 1 個 expert，而其他的模型都是至少2個。
這樣就是最稀疏的MoE了，因此單單從MoE layer的計算效率上講是最高的了。
```





- Google的 Pathways（理想）與 PaLM（現實）：https://zhuanlan.zhihu.com/p/541281939

```
當前模型的主要問題：

基本都是一個模型做一個任務；
在一個通用的模型上繼續fine-tune，會遺忘很多其他知識；
基本都是單模態；
基本都是 dense 模型，在完成一個任務時（不管難易程度），網路的所有引數都被啟用和使用；



Pathways 的願景 —— 一個跟接近人腦的框架：

一個模型，可以做多工，多模態
sparse model，在做任務時，只是 sparsely activated，只使用一部分的引數
```




- GShard論文筆記（1）-MoE結構：https://zhuanlan.zhihu.com/p/344344373

```
Mixture-of-Experts結構的模型更像是一個智囊團，裡面有多個專家，你的問題會分配給最相關的一個或多個專家，綜合他們的意見得到最終結果。

為了實現這個結構，顯而易見需要兩部分：

1）分發器：根據你的問題決定應該問哪些專家
2）一群各有所長的專家：根據分發器分過來的問題做解答
3）（可選）綜合器：很多專家如果同事給出了意見，決定如何整合這些意見，這個東西某種程度上和分發器是一樣的，其實就是根據問題，給各個專家分配一個權重



左邊部分展示了普通的Transformer模型，右邊展示了引入MoE結構的Transformer模型：其實就是把原來的FFN（兩層全連線）替換成了紅框裡的MoE結構。不過MoE裡面的“專家”依舊是FFN，只是從單個FFN換成了一群FFN，又加了一個分發器（圖中的Gating）。分發器的任務是把不同的token分發給不同的專家。




看完了專家和分發器的作用，我們再進一步看看GShard裡面他們的具體實現：

對於分發器來說，在訓練過程中，最好把token平均分配給各個專家：不然有些專家閒著，有些專家一堆事，會影響訓練速度，而且那些整天無所事事的專家肯定最後訓練的效果不好。。。因此分發器有一個很重要的任務，就是儘可能把token均分給各個專家。

為了完成這個目標，有一些繁瑣的設定：

1）引入了一個loss，專門用來控制我分發器分發的怎麼樣：如果我把token都分給一個人，loss就很高，分的越均勻（最好是徹底均分），loss越小
2）每個token最多分配給兩個專家。如果我每個token哐嘰一下發給了所有人，那我多專家有什麼意義？（專家之間的差別主要就是訓練資料的不同引起的）
3）每個專家每次最多接手C個token。和2類似，如果一個專家成天：“教練，我想打籃球”，“教練，我想唱”，“教練，我想rapper”。。。那估計最後學出來也是四不像


```








