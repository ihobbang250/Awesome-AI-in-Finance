# Awesome AI-In-Finance

![https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> Welcome to **Awesome-AI-In-Finance**, a curated collection of high-quality papers exploring the intersection of Artificial Intelligence (AI) and Finance. Contributions are highly encouraged—feel free to suggest additions or improvements!

---

## Contents
1. [Financial AI](#financial-ai)
   - [📈Prediction](#prediction)
   - [💰Investment Management](#investment-management)
   - [🔁Trading](#trading)
   - [🤖Simulation](#simulation)
   - [🤔Reasoning](#reasoning)
   - [📁Datasets](#datasets)
   - [📚Survey](#survey)
   
---

## Financial AI

### 📈Prediction

- Double-Path Adaptive-correlation Spatial-Temporal Inverted Transformer for Stock Time Series Forecasting, *KDD'25* [[Paper](https://arxiv.org/pdf/2409.15662)]
- CI-STHPAN: Pre-trained Attention Network for Stock Selection with Channel-Independent Spatio-Temporal Hypergraph, *AAAI'24* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28770)]
- MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction, *AAAI'24* [[Paper](https://arxiv.org/pdf/2402.06633)]
- MASTER: Market-Guided Stock Transformer for Stock Pricing Forecasting, *AAAI'24* [[Paper](https://arxiv.org/pdf/2312.15235)] [[Code](https://github.com/SJTU-DMTai/MASTER)]
- Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models, *ArXiv'24* [[Paper](https://arxiv.org/pdf/2304.07619)]
- Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes, *ArXiv'24* [[Paper](https://www.arxiv.org/pdf/2410.17266)]
- MANA-Net: Mitigating Aggregated Sentiment Homogenization with News Weighting for Enhanced Market Prediction, *CIKM'24* [[Paper](https://arxiv.org/pdf/2409.05698)]
- Automatic De-Biased Temporal-Relational Modeling for Stock Investment Recommendation, *IJCAI'24* [[Paper](https://www.ijcai.org/proceedings/2024/0221.pdf)]
- RSAP-DFM: Regime-Shifting Adaptive Posterior Dynamic Factor Model for Stock Returns Prediction, *IJCAI'24* [[Paper](https://www.ijcai.org/proceedings/2024/0676.pdf)]
- Trade When Opportunity Comes: Price Movement Forecasting via Locality-Aware Attention and Iterative Refinement Labeling, *IJCAI'24* [[Paper](https://www.ijcai.org/proceedings/2024/0678.pdf)]
- Modeling and Detecting Company Risks from News, *NAACL’24* [[Paper](https://aclanthology.org/2024.naacl-industry.6.pdf)]
- From News to Forecast: Iterative Event Reasoning in LLM-Based Time Series Forecasting, *NeurIPS'24* [[Paper](https://arxiv.org/pdf/2409.17515v1)]
- Re(Visiting) Large Language Models in Finance, *SSRN’24* [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4963618)]
- Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models, *WWW'24* [[Paper](https://arxiv.org/abs/2402.03659)]
- FinReport: Explainable Stock Earnings Forecasting via News Factor Analyzing Model, *WWW'24* [[Paper](https://arxiv.org/abs/2403.02647)]

### 💰Investment Management

- Mitigating Extremal Risks: A Network-Based Portfolio Strategy, *ArXiv'24* [[Paper](https://arxiv.org/pdf/2409.12208v1)]
- Temporal Representation Learning for Stock Similarities and Its Applications in Investment Management, *Arxiv’24* [[Paper](https://arxiv.org/pdf/2407.13751)]
- Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools, *WWW'24* [[paper](https://arxiv.org/pdf/2311.10801)]
- FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition, *KDD'24* [[paper](https://dl.acm.org/doi/10.1145/3637528.3671668)]

### 🔁Trading

- EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading, *AAAI'24* [[Paper](https://arxiv.org/pdf/2309.12891)]
- Can Large Language Models Mine Interpretable Financial Factors More Effectively? A Neural-Symbolic Factor Mining Agent Model, *ACL’24* [[Paper](https://aclanthology.org/2024.findings-acl.233.pdf)]
- Automate Strategy Finding with LLM in Quant investment, *ArXiv'24* [[Paper](https://arxiv.org/pdf/2409.06289)]
- Hierarchical Reinforced Trader(HRT): A Bi-Level Approach for Optimizing Stock Selection and Execution, *ArXiv'24* [[Paper](https://arxiv.org/pdf/2410.14927)]
- QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model, *Arxiv’24* [[Paper](https://arxiv.org/pdf/2402.03755)]
- IMM: An Imitative Reinforcement Learning Approach with Predictive Representation Learning for Automatic Market Making, *IJCAI'24* [[Paper](https://www.ijcai.org/proceedings/2024/0663.pdf)]
- MacMic: Executing Iceberg Orders via Hierarchical Reinforcement Learning, *IJCAI'24* ([Paper](https://www.ijcai.org/proceedings/2024/0664.pdf))
- A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist, *KDD'24* ([Paper](https://arxiv.org/pdf/2402.18485))
- MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading, *KDD'24* ([Paper](https://arxiv.org/pdf/2406.14537))
- StockFormer: Learning Hybrid Trading Machines with Predictive Coding, *IJCAI'23* ([Paper](https://www.ijcai.org/proceedings/2023/0530.pdf))

### 🤖Simulation

- EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities, *ACL'24* ([Paper](https://aclanthology.org/2024.acl-long.829/), [Code](https://github.com/tsinghua-fib-lab/ACL24-EconAgent))
- When AI Meets Finance (StockAgent): Large Language Model-based Stock Trading in Simulated Real-world Environments, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2407.18957), [Code](https://github.com/MingyuJ666/Stockagent))
- The Effect of Liquidity on the Spoofability of Financial Markets, *ICAIF’24* [[Paper](https://strategicreasoning.org/wp-content/uploads/2024/11/ICAIF24proceedings_Spoofing.pdf)]

### 🤔Reasoning

- Evaluating LLMs’ Mathematical Reasoning in Financial Document Question Answering, *ACL’24* [[Paper](https://aclanthology.org/2024.findings-acl.231.pdf)]
- LLM economicus? Mapping the Behavioral Biases of LLMs via Utility Theory, *COLM’24* [Paper]
- FinQAPT: Empowering Financial Decisions with End-to-End LLM-driven Question Answering Pipeline, *ICAIF’24* [[Paper](https://arxiv.org/pdf/2410.13959)]

### 📁Datasets

- Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context, *AAAI'24* ([Paper](https://arxiv.org/pdf/2309.07708), [Code](https://github.com/kah-ve/MarketGAN))
- DO WE NEED DOMAIN-SPECIFIC EMBEDDING MODELS? AN EMPIRICAL INVESTIGATION, *Arxiv’24* [[Paper](https://arxiv.org/pdf/2409.18511v1)] [[Code](https://github.com/yixuantt/FinMTEB)]
- UCFE: A User-Centric Financial Expertise Benchmark for Large Language Models, Arxiv’24 [[Paper](https://arxiv.org/pdf/2410.14059)] [[Code](https://github.com/TobyYang7/UCFE-Benchmark)]
- Large Language Models as Financial Data Annotators: A Study on Effectiveness and Efficiency, *COLING'24* [[Paper](https://arxiv.org/pdf/2403.18152)]
- FNSPID: A Comprehensive Financial News Dataset in Time Series, *KDD'24* ([Paper](https://arxiv.org/abs/2402.06698)] [[Code](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)]
- StockEmotions: Discover Investor Emotions for Financial Sentiment Analysis and Multivariate Time Series, *AAAI'23* [[Paper](https://arxiv.org/pdf/2301.09279)] [[Code](https://github.com/adlnlp/StockEmotions)]

### 📚Survey

- A Survey of Large Language Models in Finance (FinLLMs), *ArXiv'24* ([Paper](https://arxiv.org/pdf/2402.02315))
- A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2406.11903))
- Large Language Model Agent in Financial Trading: A Survey, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2408.06361))
- Revolutionizing Finance with LLMs: An Overview of Applications and Insights, *ArXiv'24* [[Paper](https://arxiv.org/pdf/2401.11641)]

---

## Contribute

Welcome to star & submit a PR to this repo!

## Citations

```
@article{liu2024survey,
  title={A Survey of Financial AI: Architectures, Advances and Open Challenges},
  author={Liu, Junhua},
  journal={arXiv:2411.12747},
  url={https://github.com/junhua/awesome-finance-ai-papers/},
  year={2024},
  publisher={arXiv}
}
```

```
@misc{Awesome-AI-In-Finance@2024,
  title={Awesome-AI-In-Finance},
  url={https://github.com/ihobbang250/Awesome-AI-in-Finance},
  author={Hoyoung Lee},
  year={2024}
}
```
