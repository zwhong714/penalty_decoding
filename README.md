## open-ended generation

- conditional story generation, contextual text continuation, dialogue system

## directed generation

- machine translation, data-to-text generation, summarization 
- ususally beam search, since output is tightly scoped by the input, repetition and genericness are not as problematic.
- QA



**Problem:** 

-  repetition
    - **token-level** 
    - **phrase-level** 
    - **sentence-level** 
- inconsistency
- Anisotropic:  their representations reside in a narrow subset of the entire space 

**Metric**: 

- language modeling quality 

    - ppl 

    - predicition accuracy

    - prediction repetition 

- generation quality: several axes to evaluate:  liklihood, diversity, repetition

    - Zipf's law: there is an exponential relationship between the rank of a word and its frequency in text. 

    - generation quality
        $$
        \mathbf{r e p - n}=100 \times\left(1.0-\frac{\mid \text { unique } \text { n-grams }(\hat{\boldsymbol{x}}) \mid}{\mid \operatorname{total} \text {-grams }(\hat{\boldsymbol{x}}) \mid}\right)
        $$

    - diversity 
        $$
        \text { diversity }=\prod_{n=2}^4\left(1.0-\frac{\text { rep-n }}{100}\right)
        $$

    - MAUVE:  a metric that measures the token distribution closeness between the generated text and human-written text

    - Semantic Coherence 
        $$
        \text { coherence }=v_{\boldsymbol{x}}^{\top} v_{\hat{\boldsymbol{x}}} /\left(\left\|v_{\boldsymbol{x}}\right\| \cdot\left\|v_{\hat{\boldsymbol{x}}}\right\|\right)
        $$

    - Perplexity of genereated text
        $$
        \text{gen-ppl} = 2^{f(D, \theta)}
        $$

        $$
        f(D, \theta) = \frac{1}{\sum_{\boldsymbol{x} \in \mathcal{D}}|\hat{\boldsymbol{x}}|} \sum_{\boldsymbol{x} \in \mathcal{D}} \log _2 p_\theta(\hat{\boldsymbol{x}} \mid \boldsymbol{x})
        $$

# Paper list

## Summary

[Trading Off Diversity and Quality in Natural Language Generation](https://aclanthology.org/2021.humeval-1.3.pdf) [OpenReview] [Cite: 41]





## Deterministic decoding algorithm 

- beam search

- [CTRL: A Conditional Transformer Language Model for Controllable Generation(Arxiv 2019)](https://arxiv.org/abs/1909.05858) [OpenReview] [Cite: 719]

    - **Dataset:** Wikipedia, OpenWebText,question-answer pairs ,  Europarl and UN data from WMT (En-De, En-Es, En-Fr) etc. 
    - **Model:** transformer
    - **Metric:** 
    - **Task:** open-ended generation
    - **Type**: token-level
    - **Drawback:** Parameter $\theta$ Settings have no regular effect on the results

    

#### Variants of beam search

- [A Simple, Fast Diverse Decoding Algorithm for Neural Generation (Arxiv 2016)](https://arxiv.org/pdf/1611.08562.pdf) [OpenReview] [Cite: 204]
- [Diverse beam search for improved description of complex scenes (AAAI 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/12340/12199) [OPenReview] [Cite: 163]

- [Importance of Search and Evaluation Strategies in Neural Dialogue Modeling (Arxiv 2018)](https://aclanthology.org/W19-8609.pdf) [OPenReview] [Cite: 163]





##  Stochastic decoding algorithm 

- [THE CURIOUS CASE OF NEURAL TEXT DeGENERATION (ICLR2020)](https://arxiv.org/abs/1904.09751)   [[OpenReview](https://openreview.net/forum?id=rygGQyrFvH)] [Cite: 1383]
    - top-p algorithm.
    - **Dataset:** WebText
    - **Model:** GPT2- large 
    - **Metric:** Perplexity, HUSE, Self-BLEU, Repetition, Zipf Coefficient
    - **Task:** open-ended generation
    - **Type:** token level

- [MIROSTAT: A NEURAL TEXT DECODING ALGORITHM THAT DIRECTLY CONTROLS PERPLEXITY (ICLR 2021)](https://arxiv.org/abs/2007.14966) [[OpenReview](https://openreview.net/forum?id=W1G1JZEIy5_)] [Cite: 19]
    - Adaptive top-k algorithm.
    - **Dataset:** Not-found
    - **Model:** GPT2- small
    - **Metric:** Supprise($-\log P(x)$), cross-entropy, perplexity
    - **Task:** open-ended generation
    - **Type:** token level
    - method: Our method uses statistics of previously-generated tokens as input to generate the next token, by distorting the probability distribution so it helps control the overall statistics of the generated text
- [Truncation Sampling as Language Model Desmoothing(EMNLP2022)](https://arxiv.org/abs/2210.15191) [OpenReview] [Cite: 3]
    - **Dataset:** WebText 
    - **Model:** GPT2- small, GPT2- medium, GPT2- large, GPT2-xl
    - **Metric:** MAUVE, Repetition percent
    - **Task:** open-ended generation
    - **Type:** token level

- [A Contrastive Framework for Neural Text Generation(NeurIPS 2022)](https://arxiv.org/abs/2202.06417) [[OpenReview](https://openreview.net/forum?id=V88BafmH9Pj)]  [Cite: 18]
    - contrastive-search algorithm 
    - **Dataset:** Wikitext-103[docunment-level], LCCC, DailyDialog
    - **Model:** GPT-2 small fine-tuned 
    - **Metric:** generation repetition, diversity, mauve, semantic coherence, perplexity
    - **Task:** Document generation, dialogue generation
    - **Type:** token level

- [Momentum Decoding: Open-ended Text Generation As Graph Exploration (Arxiv 2022) ](https://arxiv.org/abs/2212.02175) [OpenReview] [Cite: 0]
    - momentum decoding algorithm
    - **Dataset:** Wikinews, Wikitext-103, Book-corpus (story domain)
    - **Model:** GPT-2 XL 
    - **Metric:** diversity, mauve, semantic coherence, perplexity, greedy ratio, Flops
    - **Task:** Open-ended generation 
    - **Type:** token level

- [Contrastive Decoding: Open-ended Text Generation as Optimization (Arxiv 2022)](https://arxiv.org/abs/2210.15097) [OpenReview] [Cite: 6]
    - contrastive decoding algorithm
    - **Dataset:** Wikinews, Wikitext-103, Book-corpus (story domain)
    - **Model:** GPT-2 XL , OPT
    - **Metric:** Repetition, Diversity, MAUVE, Coherence 
    - **Task:** Open-ended generation 
    - **Type:** token level
- [Locally Typical Sampling (TACL2022)](https://arxiv.org/abs/2202.00666) [OpenReview] [Cite: 6]
    - **Dataset:** Wikitext-103
    - **Model:** GPT-2 large 
    - **Metric:** PPL, MAUVE, Reptition, Zipf , Diverisity
    - **Task:** Abstractive summarization, story generation 
    - **Type:** token level
- [PENALIZING THE HIGH-LIKELIHOOD: A NOVEL SAMPLING METHOD FOR OPEN-ENDED NEURAL TEXT GENERATION VIA INVERSE PROBABILITY WEIGHTING (ICLR 2023 Reject) ](https://openreview.net/pdf?id=e9CKiV6pgBD) [[OpenReview](https://openreview.net/forum?id=e9CKiV6pgBD)]
    - **Dataset:** just one promt "She walks in beauty"
    - **Model:** GPT-2 xl 
    - **Metric:** PPL,  Self-BLEU, n-gram erntropy, zipf 
    - **Task:** Open-ended  generation
    - **Type:** token level





## Loss function

- [A Contrastive Framework for Neural Text Generation(NeurIPS 2022)](https://arxiv.org/abs/2202.06417) [[OpenReview](https://openreview.net/forum?id=V88BafmH9Pj)]  [Cite: 18]

    - SimCTG training Loss 
        $$
        \mathcal{L}_{\mathrm{CL}}=\frac{1}{|\boldsymbol{x}| \times(|\boldsymbol{x}|-1)} \sum_{i=1}^{|\boldsymbol{x}|} \sum_{j=1, j \neq i}^{|\boldsymbol{x}|} \max \left\{0, \rho-s\left(h_{x_i}, h_{x_i}\right)+s\left(h_{x_i}, h_{x_j}\right)\right\}
        $$

        $$
        s\left(h_{x_i}, h_{x_j}\right)=\frac{h_{x_i}^{\top} h_{x_j}}{\left\|h_{x_i}\right\| \cdot\left\|h_{x_j}\right\|}
        $$

        $$
        \mathcal{L}_{\text {SimCTG }}=\mathcal{L}_{\mathrm{MLE}}+\mathcal{L}_{\mathrm{CL}}
        $$

    - **Metric:** Perplexity, prediction accuracy, prediction repetition

- [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319) 
    - decreases training loss on repeated tokens and thus implicitly reduces gradients on frequent tokens as well

- [Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation (NeurIPS 2022)](https://arxiv.org/pdf/2206.02369.pdf) [[OpenReview](https://openreview.net/forum?id=sexfswCc7B)] [Cite: 3]



## Theoretical prove

[A Theoretical Analysis of the Repetition Problem in Text Generation(AAAI 2021)](https://arxiv.org/abs/2012.14660) [OpenReview] [Cite: 25]





## Controllable text generation

Plug and play language models: a simple approach to controlled text generation (ICLR 2020) . 



