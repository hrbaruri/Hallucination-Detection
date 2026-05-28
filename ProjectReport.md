# Report: Hallucination detection in tool calling
Code for this project can be found in this [repository](https://github.com/hrbaruri/Hallucination-Detection/tree/main).
## Team Members: 
**Yakupova Valeria**

**Dmitrij Korogod** 

**Weerathep Rattanajaratkul**

**Hritendu Russo Baruri**

## 1. Motivation 

### Hallucinations and Tools
#### Tools
In the context of LLMs, a tool is an external service that an LLM receives information from upon a query from the user. 

For example: A weather API. The user requests to know the weather in Moscow, however, the LLM does not have this information readily available. So, it calls on a `weather api (the tool)` to get the current weather in Moscow and forwards the information to the user.
#### Hallucinations in tool calling
LLMs grounded by tool APIs can still fabricate or distort facts in their final response, even when correct data was returned. Such distortions or incorrect facts are called Hallucinations. In the context of tool-calling, Hallucinations can be roughly divided into three categories:
- 1.**Tool Output Contradiction:** Model response directly contradicts a fact returned by the tool. For example: 
        
        `Tool says "sunny" → Answer says "rainy"`
- 2.**Overgeneration:** Response contains information not present in the tool output i.e. unverified speculative additions.

        `…and the weather has been good past few months`
- 3.**Missing Tool:** Response suggests actions that require a tool not available in the current session.

        `"Would you like me to book a ticket?"` 
    but no booking tool is present.

Existing hallucination detection methods were not designed for tool-calling dialogues and have no span-level benchmarks for this setting, which is exactly the problem this project addresses.

--- 

# 2 Methodology

## About the baselines
### Baseline A: LettuceDetect
`LettuceDetect` (KR Labs, 2025) is a token-classification model built on ModernBERT. Given a (question, context, answer) triple it assigns each answer token a hallucination probability.
- Model: KRLabsOrg/lettuce-detect-base-v2 (or -large-v2)
- Input: question, list of context passages, answer
- Output: per-token hallucination scores + hallucinated spans

Reference: [LettuceDetect](https://arxiv.org/pdf/2502.17125)
### Baseline B: LookBackLens
**LookBackLens** (Tang et al., 2024) uses the *lookback ratio* as a hallucination signal. For each answer token at position $t$:

$$\text{LookbackRatio}(t) = \frac{\sum_{k < |\text{context}|} A_{t,k}}{\sum_{k < |\text{context}|} A_{t,k} \;+\; \sum_{|\text{context}| \le k < t} A_{t,k}}$$

where $A_{t,k}$ is the averaged attention weight from the token at position $t$ to position $k$ (averaged over all layers and heads).

- **High ratio** → the model is attending to the provided context → likely grounded
- **Low ratio** → the model is attending to its own prior generation → potential hallucination

The original paper trains a linear classifier on per-layer/per-head lookback ratio vectors using Llama-2-7B. Below we implement the core algorithm for **GPT-2** (accessible without large-model downloads) and use a simple logistic-regression classifier trained on aggregate ratio features. To reproduce the paper's exact numbers, replace `MODEL_NAME` with `meta-llama/Llama-2-7b-chat-hf` and provide the paper's pre-trained classifier weights.

Reference: [original paper](https://arxiv.org/abs/2407.07071); [Code](https://github.com/amazon-science/lookback-lens)

### Improved Hallucination Detection Baselines

We upgrade the original baseline evaluation framework by introducing a multi-layered benchmark that incorporates lexical token alignment, transformer-based token classification backbones, and attention-guided contextual lens ratios. To evaluate the resilience of dialogue systems operating under complex tool-assisted workflows, our methodology focuses on both sample-level and span-level detection of fine-grained hallucinations.

The methodology is structured across the following core pillars:

1. **Dataset Rehabilitation and Schema Integration**: We rebuild the clean evaluation splits directly from the raw ToolACE dataset. The dialogue turns from the original dataset are framed as structural triplets consisting of the user Query, the execution output of the tools as Context, and the model's final response as Output. Ground-truth span annotations are structured in alignment with the RAGTruth schema across three target corruption scenarios:
   - **Tool Output Contradiction**: Direct factual discrepancies between the model's natural language response and the tool payload. To generate such hallucinations, the large language model (gpt-o4-mini) was asked to mildly modify the final response such that it contradicts the tool output.
   
   - **Overgeneration**: Unverified assertions or speculative facts introduced by the model that are completely absent from the tool context. Hallucinations of these type were generated in two steps. First, Llama-3.1 (8B) was asked to generate additional fact related to the user request but not supported by the tool response. Then, the generated fact was added to the final response programmatically to ensure the correctness of the extracted hallucinated span.
   
   - **Missing Tool**: Actions suggested by the model that imply or require the activation of an unavailable tool API. Hallucinations of these type were generated similar to the overgeneration data, but this time the LLM (Llama-3.1 (8B)) was given the list of available tools and asked to generate an offer to do something that would require tool absent from this list. 

2. **Multi-Model Baseline Extensions**: Rather than relying strictly on raw out-of-the-box predictions, we evaluate and optimize several architectural archetypes:
   - **Lexical Span Verifier**: A non-parametric, exact-match lookup baseline that evaluates token-level overlap and flags sub-strings failing lexical alignment with tool contexts.
   
   - **Lettuce-Span Supervised Detector**: Utilizes a modern token-classification transformer backbone (`KRLabsOrg/lettucedect-base-modernbert-en-v1`) based on the BERT architecture, optimized to directly map token positions to boundary probabilities.
   
   - **LookBack-Span Supervised Detector**: Powered by a dense generative language model backbone (`Qwen/Qwen2.5-0.5B`), this approach extracts inner attention layers to compute lookback ratios. It evaluates how heavily the model attends to the input context versus its own historical generation tokens.
   
   - **Soft-Vote Ensemble**: A collaborative combination that synthesizes sample-level probabilities across the individual supervised models to maximize precision, control false-positive rates, and boost the overall Area Under the Receiver Operating Characteristic curve (AUROC).


---

## 3. Discussion of Results
**Overall Test Set Metrics:**

| Method | Backbone | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| **Lexical Span Verifier** | N/A (Non-parametric) | 56.6% | 0.529 | 0.517 | 0.523 |
| **LookBack-Span Supervised** | Qwen2.5-0.5B | 48.9% | 0.463 | 0.708 | 0.560 |
| **Lettuce-Span Supervised** | ModernBERT | **67.7%** | **0.665** | 0.601 | 0.631 |
| **Soft-Vote Ensemble** | Combined | 67.6% | 0.639 | **0.679** | **0.659** |

The comparative performance across the initial baselines and our optimized supervised implementations reveals clear trade-offs between detection recall and calibration precision.

An examination of the initial out-of-the-box logistic regression baselines evaluated on the combined benchmark ($n=828$) highlights the inherent difficulty of zero-shot alignment:

- **LettuceDetect (LogReg Baseline)**: Achieved an absolute recall of 1.000, but suffered from extreme over-flagging, resulting in a low precision of 0.460 and an overall accuracy of 0.460. This indicates that without threshold calibration or fine-tuning, the token classifier collapses into a trivial majority-positive state.

- **LookBackLens (LogReg Baseline)**: Demonstrated superior structural calibration compared to the raw token baseline, elevating accuracy to 0.506 and precision to 0.476, while preserving a solid recall of 0.714.

- **Baseline Agreement**: Agreement statistics show that at least one method flags a hallucination in 100.0% of cases, with a mutual overlap on 69.1% of samples, confirming that both models catch overlapping signal boundaries but suffer from high false-positive rates when left uncalibrated.


By introducing the supervised span training paradigm and merging their outputs into a **Soft-Vote Ensemble**, performance increases significantly:

- **Overall Ensemble Performance**: The soft-voting ensemble establishes the highest robustness on the benchmark, yielding an overall accuracy of **67.6%** (0.6763), an overall F1-score of **65.9%** (0.6590), and well-balanced precision/recall curves (0.6395 and 0.6798 respectively).

- **Specificity on Clean Inputs**: On uncorrupted data (`clean`), the supervised Lettuce model shows strong specificity, leaving true responses unflagged with an accuracy of **78.3%**. The Soft-Vote Ensemble tracks close behind at **71.0%** accuracy on clean inputs, indicating a successful suppression of the false-positive errors that plagued the initial zero-shot baselines.

- **Robustness Against Missing Tools**: For the challenging `missing_tool` hallucination category, the ensemble manages an accuracy of **63.3%** and an F1 score of **54.8%** (Precision = 0.5679, Recall = 0.5287). This proves that synthesizing inner attention-based contextual awareness with dense token-level representations allows the detector to reliably capture abstract structural anomalies—such as an assistant proposing actions without tool support—alongside explicit factual contradictions.

---

## 4. Repository Content

- `hallucination_detection_final.ipynb`: Jupyter notebook with the code for this project. It
1. Rebuilds the clean ToolACE split directly from the original dataset.
2. Evaluates on a combined benchmark made of `clean + 3 hallucination datasets` from `datasets/`.
3. Trains stronger sample-level detectors on real labels and reports both overall and per-type metrics.
- `datasets`: folder containing generated datasets


   

