



Megatron-LM 的張量並行，通訊量很大，同時，計算和通訊沒辦法同時進行。



需要特別考慮的是：由於前向和後向傳播中每層都有兩個 all reduce，因此 TP 需要裝置間有非常快速的互聯。因此，除非你有一個非常快的網路，否則不建議跨多個節點進行 TP。
