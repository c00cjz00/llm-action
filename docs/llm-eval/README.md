
- How to Evaluate a Large Language Model (LLM)?：https://www.analyticsvidhya.com/blog/2023/05/how-to-evaluate-a-large-language-model-llm/



## 評估指標

### 困惑度 perplexity

語言模型的效果好壞的常用評價指標是困惑度(perplexity),在一個測試集上得到的perplexity 越低，說明建模的效果越好。

PPL是用在自然語言處理領域（NLP）中，衡量語言模型好壞的指標。它主要是根據每個詞來估計一句話出現的機率，並用句子長度作normalize。

PPL越小越好，PPL越小，p(wi)則越大，也就是說這句話中每個詞的機率較高，說明這句話契合的表較好。

- https://blog.csdn.net/hxxjxw/article/details/107722646








## helm


- https://crfm.stanford.edu/helm/latest/

### 59 metrics

```

# Accuracy
none
Quasi-exact match
F1
Exact match
RR@10
NDCG@10
ROUGE-2
Bits/byte
Exact match (up to specified indicator)
Absolute difference
F1 (set match)
Equivalent
Equivalent (chain of thought)
pass@1


# Calibration
Max prob
1-bin expected calibration error
10-bin expected calibration error
Selective coverage-accuracy area
Accuracy at 10% coverage
1-bin expected calibration error (after Platt scaling)
10-bin Expected Calibration Error (after Platt scaling)
Platt Scaling Coefficient
Platt Scaling Intercept


# Robustness
Quasi-exact match (perturbation: typos)
F1 (perturbation: typos)
Exact match (perturbation: typos)
RR@10 (perturbation: typos)
NDCG@10 (perturbation: typos)
Quasi-exact match (perturbation: synonyms)
F1 (perturbation: synonyms)
Exact match (perturbation: synonyms)
RR@10 (perturbation: synonyms)
NDCG@10 (perturbation: synonyms)


# Fairness
Quasi-exact match (perturbation: dialect)
F1 (perturbation: dialect)
Exact match (perturbation: dialect)
RR@10 (perturbation: dialect)
NDCG@10 (perturbation: dialect)
Quasi-exact match (perturbation: race)
F1 (perturbation: race)
Exact match (perturbation: race)
RR@10 (perturbation: race)
NDCG@10 (perturbation: race)
Quasi-exact match (perturbation: gender)
F1 (perturbation: gender)
Exact match (perturbation: gender)
RR@10 (perturbation: gender)
NDCG@10 (perturbation: gender)
Bias
Stereotypical associations (race, profession)
Stereotypical associations (gender, profession)
Demographic representation (race)
Demographic representation (gender)
Toxicity
Toxic fraction
Efficiency
Observed inference runtime (s)
Idealized inference runtime (s)
Denoised inference runtime (s)
Estimated training emissions (kg CO2)
Estimated training energy cost (MWh)

---
# General information

# eval
# train
truncated
# prompt tokens
# output tokens
# trials

---

# Summarization metrics
SummaC
QAFactEval
BERTScore (F1)
Coverage
Density
Compression
HumanEval-faithfulness
HumanEval-relevance
HumanEval-coherence

# APPS metrics
Avg. # tests passed
Strict correctness


# BBQ metrics
BBQ (ambiguous)
BBQ (unambiguous)


# Copyright metrics
Longest common prefix length
Edit distance (Levenshtein)
Edit similarity (Levenshtein)


# Disinformation metrics
Self-BLEU
Entropy (Monte Carlo)

# Classification metrics
Macro-F1
Micro-F1
```


## lm-evaluation-harness

- https://github.com/EleutherAI/lm-evaluation-harness






## Chatbot Arena

- https://chat.lmsys.org/



### 現有的 LLM 基準框架

儘管存在 HELM 和 lm-evaluation-harness 等基準測試，但由於缺乏成對比較相容性，它們在評估自由形式問題時存在不足。 

這就是 Chatbot Arena 等眾包基準測試平臺發揮作用的地方。





## CLEVA

中文語言模型評估平臺

- https://github.com/LaVi-Lab/CLEVA
- http://www.lavicleva.com/#/homepage/overview










