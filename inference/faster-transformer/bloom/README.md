






## 統計效能指標

### 資料格式

```
[
    {
        "id":0,
        "input":"#姓名：何#性別：女士#剩餘額度：一萬七千#當前額度：一萬七千#繫結銀行：無#活動：D#初始額度：nan#手機尾號：七八八#註冊時間：二零二三年一月三日#借款日期：nan#優惠券數：0.0#電銷時間：二零二三年四月二十三日#登入型別：無#提額時間：二零二三年四月二十二日#offer型別：提額#cust_type：授信T91-180\n\nHuman: 你好。\nAssistant: ",
        "answer":"Assistant: 好，請問一下是何女士對吧？<\/s>"
    },
    {
        "id":1,
        "input":"#姓名：嶽#性別：先生#剩餘額度：五萬#當前額度：零#繫結銀行：農業銀行#活動：E#初始額度：一萬#手機尾號：六四七三#註冊時間：二零一九年十一月七日#借款日期：nan#優惠券數：1.0#電銷時間：二零二三年二月二十五日#登入型別：無#提額時間：nan#offer型別：nan#cust_type：授信T31-90\n\nHuman: 喂。\nAssistant: 您好，請問就是嶽先生對吧？<\/s>Human: 行。對呀。\nAssistant: 誒你好，嶽先生，不好意思打擾到你，我這邊是xxx的回訪專員哈，來電話給你做個提醒，關注到你這邊在我們ttt裡面申請的這個額度還打算用嗎？誒已經過了六天了，ttt。<\/s>Human: 嗯。我用，但是我我用是可以用，但是我不我我上次問他，他不都不跟我講，我這個錢，假如我這個一萬塊錢，我用一天大概要多少錢利息知道吧？\nAssistant: 啊就是我這邊給你說一下哈，可能之前之前可能是有同事就是就是沒有給你說，對吧？<\/s>Human: 對呀，說我不敢用呀，你。\nAssistant: 啊跟你說一下。<\/s>Human: 嗯。\nAssistant: 一萬塊錢一萬塊錢，你要分一年，他總共的利息下來是在一千三左右哈，你平均到每個月的話，你下來的一百塊錢左右，每天每個月在一百塊錢左右的利息，平均到每天就在在三塊錢左右哈。<\/s>Human: 就是五萬塊錢。\nAssistant: 我一萬塊錢哈。<\/s>Human: 呃一一天三塊錢左右應該就是一百塊錢，但是的話說就是。\nAssistant: 一百塊錢左右嘛。<\/s>Human: 一你我一百塊錢就是一就是假的話，就是利大概是六千多塊錢。\nAssistant: 我看一下你。<\/s>Human: 一年。\nAssistant: 分的話差不多六千塊錢左右哈。<\/s>Human: 對對對，因為對我所以說你你給我講一百塊錢，我說大概六千多塊，我就大概六千塊錢左右了，一年。\nAssistant: 對。<\/s>Human: 啊啊\nAssistant: 嗯嗯差這一五萬塊錢一年就是六千塊錢左右哈。<\/s>Human: 哦對對對，因為因為因為我這我這這這個錢還有個問題，就是假的話的。就是我我說的我拿的是我我的就是分十二期還，但是我提前還就是提前還就沒有違約，沒有什麼利息的啦。\nAssistant: 嗯。<\/s>Human: 是\nAssistant: 誒你提前還款的話，它就是就是你你就是我們就是滿足條件沒有逾期的情況下就可以申請提前還款了，提前還款，您用幾個月收取你幾個月的利息哈，就是滿足條件還滿三期之後就可以申請提前還款了哈，就比如說哈，因為先生比如說你分了一年啊，就你只用了半年哈，你提前還款了，你嗯。<\/s>Human: 嗯。對。對對對。對。\nAssistant: ",
        "answer":"Assistant: 就是後你提前還款了之後，後面是不會收取這利息的哈。<\/s>"
    },
    {
        "id":2,
        "input":"#姓名：李#性別：先生#剩餘額度：兩萬元#當前額度：零#繫結銀行：無#活動：C#初始額度：nan#手機尾號：五零零七#註冊時間：nan#借款日期：nan#優惠券數：nan#電銷時間：nan#登入型別：nan#提額時間：nan#offer型別：nan#cust_type：nan\n\nAssistant: 好，請問是李先生是吧？<\/s>Human: 啊行。\nAssistant: 誒李先生您好，我是xxx的工作人員。<\/s>Human: 哦\nAssistant: 來電呢是給您確認一下，您之前在ttt還有個剩餘額度，兩萬元是您本人在自己使用的，對吧？<\/s>Human: 啊？\nAssistant: 誒是這樣的先生，因為呢我們系統顯示您之前還款記錄保持得比較好，現在公司在回饋優質客戶專門來電給您確認一下，<\/s>Human: 好\nAssistant: 如果您打算以後長期使用ttt，還需要提高您的額度和降低利息的話。<\/s>Human: 嗯\nAssistant: ",
        "answer":"Assistant: 您現在呢只需要將您的剩餘額度兩萬元按去暫時借出來。<\/s>"
    }
]
```


### 推理


推理耗時，平均每個token生成時長等。

單卡：
```
CUDA_VISIBLE_DEVICES=1 python examples/pytorch/gpt/firefly_lambada_dianxiao_1w_stat_token.py \
--checkpoint-path /workspace/model/firefly-2b6-dx-1tp/belle7b/1/1-gpu \
--tokenizer-path /workspace/model/firefly-2b6-dx \
--dataset-path /workspace/data/lambada_test.jsonl \
--lib-path  /workspace/lib/libth_transformer.so \
--inference-data-type fp16 --show-progress --input-token-len 64 --output-token-len 256 \
--dianxiao-path-stat /workspace/output/firefly_random_sample_1w_256_stat_ft.json
```

雙卡張量並行：
```
CUDA_VISIBLE_DEVICES=2,3  mpirun -n 2 python examples/pytorch/gpt/firefly_lambada_dianxiao_1w_stat_token.py \
--checkpoint-path /workspace/model/firefly-2b6-dx-2tp/belle7b/1/2-gpu \
--tokenizer-path /workspace/model/firefly-2b6-dx \
--dataset-path /workspace/data/lambada_test.jsonl \
--lib-path  /workspace/lib/libth_transformer.so \
--inference-data-type fp16 \
--tensor-para-size 2 \
--pipeline-para-size 1 \
--show-progress \
--input-token-len 64 \
--output-token-len 256 \
--dianxiao-path-stat  /workspace/output/firefly_random_sample_1w_256_stat_ft_tp2.json
```

