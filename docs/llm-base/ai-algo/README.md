- https://zhuanlan.zhihu.com/p/644815089

## 模型對比

| 模型                                                       | GPT2 Medium（345M） | Bloom-7b1                | LLaMA-7B                      | LLaMA2-7B                     | ChatGLM-6B  | ChatGLM2-6B |
| -------------------------------------------------------- | ----------------- | ------------------------ | ----------------------------- | ----------------------------- | ----------- | ----------- |
| 詞表大小（vocab_size）                                         | 50257             | 250880                   | 32000                         | 32000                         | 130528      | 65024       |
| Transformer層（n_layer, num_layers, num_hidden_layers）     | 24                | 30                       | 32                            | 32                            | 28          | 28          |
| 注意力頭數（num_attention_heads, n_head）                       | 16                | 32                       | 32                            | 32                            | 32          | 32          |
| key_value頭數（num_key_value_heads）                         | N/A               | N/A                      | N/A                           | N/A                           | N/A         | N/A         |
| 隱藏層大小（hidden_size）                                       | 1024(n_embd)      | 4096(n_embed)            | 4096                          | 4096                          | 4096        | 4096        |
| 前饋神經網路的隱藏層大小（ffn_hidden_size, intermediate_size,n_inner） | 4*n_embd          | 4 * hidden_size          | 11008                         | 11008                         | 16384       | 13696       |
| seq_length, n_ctx                                        | 1024              | 2048                     | 2048(max_position_embeddings) | 2048(max_position_embeddings) | 2048        | 32768       |
| n_positions,max_position_embeddings,n_embed              | 1024(default)     | 2048(4096,bloomz-7b1-hf) | 2048                          | 2048(4096,llama2-chat-hf)     | hidden_size | hidden_size |

- https://huggingface.co/gpt2-medium/resolve/main/config.json
- https://huggingface.co/bigscience/bloom-7b1/blob/main/config.json
- https://huggingface.co/bigscience/bloomz-7b1-mt/blob/main/config.json
- https://huggingface.co/yahma/llama-7b-hf/blob/main/config.json
- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json
- https://huggingface.co/THUDM/chatglm2-6b-32k
- https://huggingface.co/THUDM/chatglm-6b

說明：

- 通常 seq_length 與 max_position_embeddings 相等。
- key_value頭數：This is the number of key_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group.



## LLaMA

| 模型                                                       | LLaMA-7B                      | LLaMA-2-7B                    | LLaMA-13B | LLaMA-2-13B | LLaMA-30B | LLaMA-65B | LLaMA-2-70B |
| -------------------------------------------------------- | ----------------------------- | ----------------------------- | --------- | ----------- | --------- | --------- | ----------- |
| 詞表大小（vocab_size）                                         | 32000                         | 32000                         | 32000     | 32000       | 32000     | 32000     | 32000       |
| Transformer層（n_layer, num_layers, num_hidden_layers）     | 32                            | 32                            | 40        | 40          | 60        | 80        | 80          |
| 注意力頭數（num_attention_heads, n_head）                       | 32                            | 32                            | 40        | 40          | 52        | 64        | 64          |
| key_value頭數（num_key_value_heads）                         | N/A                           | 32                         | N/A       | 40          | N/A       | N/A       | 8           |
| 隱藏層大小（hidden_size）                                       | 4096                          | 4096                          | 5120      | 5120        | 6656      | 8192      | 8192        |
| 前饋神經網路的隱藏層大小（ffn_hidden_size, intermediate_size,n_inner） | 11008                         | 11008                         | 13824     | 13824       | 17920     | 22016     | 28672       |
| seq_length, n_ctx                                        | 2048(max_position_embeddings) | 2048(max_position_embeddings) | 2048      | N/A         | 2048      |           | N/A         |
| n_positions,max_position_embeddings,n_embed              | 2048                          | 2048(4096,llama2-chat-hf)     | N/A       | 4096        | N/A       | N/A       | 4096        |

- https://huggingface.co/decapoda-research/llama-13b-hf

- 
