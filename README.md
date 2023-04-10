## Problem 

-  repetition
    - token-level 
    - phrase-level 
    - sentence-level 
- inconsistency
- Anisotropic:  their representations reside in a narrow subset of the entire space 

## open-ended generation

- conditional story generation, contextual text continuation, dialogue system

## directed generation

- machine translation, data-to-text generation, summarization 
- ususally beam search, since output is tightly scoped by the input, repetition and genericness are not as problematic.



## Metric 

#### language modeling quality 

- ppl 
- predicition accuracy
- prediction repetition 

#### generation quality

several axes to evaluate:  liklihood, diversity, repetition

- Zipf's law: there is an exponential relationship between the rank of a word and its frequency in text. 

- generation quality

    - $$
        \mathbf{r e p - n}=100 \times\left(1.0-\frac{\mid \text { unique } \text { n-grams }(\hat{\boldsymbol{x}}) \mid}{\mid \operatorname{total} \text {-grams }(\hat{\boldsymbol{x}}) \mid}\right)
        $$

- diversity 

    - $$
        \text { diversity }=\prod_{n=2}^4\left(1.0-\frac{\text { rep-n }}{100}\right)
        $$

- MAUVE:  a metric that measures the token distribution closeness between the generated text and human-written text

- Semantic Coherence 

    - $$
        \text { coherence }=v_{\boldsymbol{x}}^{\top} v_{\hat{\boldsymbol{x}}} /\left(\left\|v_{\boldsymbol{x}}\right\| \cdot\left\|v_{\hat{\boldsymbol{x}}}\right\|\right)
        $$

- Perplexity of genereated text

    - $$
        \text{gen-ppl} = 2^{f(D, \theta)}
        $$

    - $$
        f(D, \theta) = \frac{1}{\sum_{\boldsymbol{x} \in \mathcal{D}}|\hat{\boldsymbol{x}}|} \sum_{\boldsymbol{x} \in \mathcal{D}} \log _2 p_\theta(\hat{\boldsymbol{x}} \mid \boldsymbol{x})
        $$

        

#### Deterministic decoding algorithm 

- beam search





#### Stochastic decoding algorithm 

- [THE CURIOUS CASE OF NEURAL TEXT DeGENERATION (ICLR2020)](https://arxiv.org/abs/1904.09751)   [[OpenReview](https://openreview.net/forum?id=rygGQyrFvH)] [Cite: 1383]
    - top-p algorithm.
    - **Dataset:** WebText
    - **Model:** GPT2- large 
    - **Metric:** Perplexity, HUSE, Self-BLEU, Repetition, Zipf Coefficient
    - **Task:** open-ended generation
    - **Type:** token level
- [A Contrastive Framework for Neural Text Generation(NeurIPS 2022)](https://arxiv.org/abs/2202.06417) [[OpenReview](https://openreview.net/forum?id=V88BafmH9Pj)]  [Cite: 18]
    - contrastive-search algorithm 
    - **Dataset:** Wikitext-103[docunment-level]
    - **Model:** GPT-2 small fine-tuned 
    - **Task:** open-ended generation





#### Loss function

- [A Contrastive Framework for Neural Text Generation(NeurIPS 2022)](https://arxiv.org/abs/2202.06417) [[OpenReview](https://openreview.net/forum?id=V88BafmH9Pj)]  [Cite: 18]

    - SimCTG training Loss 

    - $$
        \mathcal{L}_{\mathrm{CL}}=\frac{1}{|\boldsymbol{x}| \times(|\boldsymbol{x}|-1)} \sum_{i=1}^{|\boldsymbol{x}|} \sum_{j=1, j \neq i}^{|\boldsymbol{x}|} \max \left\{0, \rho-s\left(h_{x_i}, h_{x_i}\right)+s\left(h_{x_i}, h_{x_j}\right)\right\}
        $$

    - $$
        s\left(h_{x_i}, h_{x_j}\right)=\frac{h_{x_i}^{\top} h_{x_j}}{\left\|h_{x_i}\right\| \cdot\left\|h_{x_j}\right\|}
        $$

    - $$
        \mathcal{L}_{\text {SimCTG }}=\mathcal{L}_{\mathrm{MLE}}+\mathcal{L}_{\mathrm{CL}}
        $$

        

- [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319) 
    - decreases training loss on repeated tokens and thus implicitly reduces gradients on frequent tokens as well















