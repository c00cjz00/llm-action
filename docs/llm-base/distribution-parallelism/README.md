



- One weird trick for parallelizing convolutional neural networks
  - 不同的層適合用不同的並行方式，具體的，卷積層資料比引數大，適合資料並行，全連線層引數比資料大，適合模型並行。
- Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks
  - 這篇文章在抽象上更進一步，發現數據並行，模型並行都只是張量切分方式的不同罷了，有的是切資料，有的是切模型，而且對於多維張量，在不同的維度上切分，效果也不同，譬如在sample, channel, width, length等維度都可以切分。
  - 其次，不同的切分方式，都是一種構型（configuration)，不同的構型會導致不同的效果，所以尋找最優的並行方式，其實就是在構型空間裡面搜尋最優的構型而已，問題形式化成一個搜尋問題。
  - 最後，引入了代價模型來衡量每個構型的優劣，並提出了一系列對搜尋空間剪枝的策略，並實現了原型系統。

- BEYOND DATA AND MODEL PARALLELISM FOR DEEP NEURAL NETWORKS（FlexFlow）
  - 主要是提出了execution simulator來完善cost model。
  
- Supporting Very Large Models using Automatic Dataflow Graph Partitioning（Tofu）
  - tofu 提出了一套DSL，方便開發者描述張量的劃分策略，使用了類似poly的integer interval analysis來描述並行策略，同樣，並行策略的搜尋演算法上也做了很多很有特色的工作。
  - Tofu與所有其它工作的不同之處在於，它的關注點是operator的劃分，其它工作的關注點是tensor的劃分，二者當然是等價的。
  - 不過，我認為關注點放在tensor的劃分上更好一些，這不需要使用者修改operator的實現，Tofu需要在DSL裡描述operator的實現方式。
  - 
- Mesh-TensorFlow: Deep Learning for Supercomputers
  - Mesh-TensorFlow的作者和GShard的作者幾乎是重疊的，Mesh-TensorFlow甚至可以被看作GShard的前身。
  - Mesh-TensorFlow的核心理念也是beyond batch splitting，資料並行是batch splitting，模型並行是張量其它維度的切分。這篇文章把叢集的加速卡抽象成mesh結構，提出了一種把張量切分並對映到這個mesh結構的辦法。



- Unity: Accelerating DNN Training Through Joint Opt of Algebraic Transform and Parallelization：https://zhuanlan.zhihu.com/p/560247608
  - Unity 在 FlexFlow、TASO 和 MetaFlow 的基礎上，提出在平行計算圖（PCG）中代數變換和並行化的統一表示（OP，Operator）和共最佳化（圖替代，Substitution）方法，可以同時考慮分散式訓練中的計算、並行和通訊過程。對於共最佳化，Unity 使用一個多級搜尋演算法來高效搜尋效能最好的圖替代組合以及相應的硬體放置策略。




- https://github.com/DicardoX/Individual_Paper_Notes
- https://jeongseob.github.io/readings_mlsys.html
- https://paperswithcode.com/methods/category/distributed-methods





