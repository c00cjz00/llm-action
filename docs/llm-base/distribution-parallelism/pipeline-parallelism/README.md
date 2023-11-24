



DP 將批次（global batch size）拆分為小批次（mini-batch）。PP 將一個小批次切分為多個塊 (chunks)，因此，PP 引入了微批次(micro-batch，MBS) 的概念。

計算 DP + PP 設定的全域性批次大小的公式為: `mbs*chunks*dp_degree` ， 比如：DP並行度為4，微批次大小為8，塊為32，則全域性批次大小為：`8*32*4=1024`。



