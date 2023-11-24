







- [預訓練中文語料彙總（附資料）](https://zhuanlan.zhihu.com/p/163616279)





好的資料： 書籍 、 維基百科、 程式碼











## 業界大模型訓練資料


### OPT-175B




Meta AI 團隊希望在儘可能大的語料庫上訓練這個模型。 它由以下 5 個經過過濾的文字文件資料集的並集組成：

BookCorpus，由超過 10K 未出版的書籍組成，

CC-Stories，其中包含 CommonCrawl 資料的子集，經過過濾以匹配 Winograd 模式的故事風格，

The Pile，其中包括 Pile-CC、OpenWebText2、USPTO、Project Gutenberg、OpenSubtitles、Wikipedia、DM Mathematics 和 HackerNews。

Baumgartner 等人開發的 Pushshift.io Reddit 資料集。 並由 Roller 等人處理。

CCNewsV2 包含 RoBERTa 中使用的 CommonCrawl 新聞資料集英文部分的更新版本


最終的訓練資料包含180B個token，對應800GB的資料。 驗證分割由 200MB 的預訓練資料組成，根據預訓練語料庫中每個資料集的大小按比例進行取樣。

該資料集可能包含令人反感的內容，因為資料集的一部分是公共 Common Crawl 資料的子集以及公共 Reddit 資料的子集，其中可能包含如果直接檢視可能具有侮辱性、威脅性或可能導致焦慮的句子 。






### Bloom-176B



41.5TB 經過大量去重和清洗的文字，包含 46 種語言，最終轉換為 350B 個詞元


46種自然語言，13種程式語言。


模型的詞彙表含 250,680 個詞元








資料的多樣性很重要，涉及領域越豐富越好

有害資訊生成與有害資訊鑑別能力難以兩全

低質量資料過濾可以提高有害資訊鑑別能力，以及下游任務的表現，有害資訊過濾會起到相反作用

預訓練資料的來源時間，與下游任務資料的來源時間越接近，模型表現就越好。不光預訓練資料過時會有負面影響，過於超前也會。





















