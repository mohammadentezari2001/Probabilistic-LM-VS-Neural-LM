## Language Modeling from Scratch: Traditional vs. Neural Approaches

This repository contains the implementation and comparison of various Language Models (LMs), covering traditional count-based methods with smoothing and a modern neural network approach using word embeddings.

The project is structured into two main Jupyter Notebooks:
1.  **`ProbabilisticLM.ipynb`**: Implementation of traditional N-gram Language Models with Dirichlet and Kneser-Ney smoothing, including hyperparameter tuning.
2.  **`NeuralLM.ipynb`**: Implementation of a Feed-Forward Network (FFN) Language Model using pre-trained GloVe embeddings, learning rate tuning, and a final prediction task.

## Data

The models are trained and evaluated on a Reuters news dataset split into four files:
| File Name | Role | Record Count | Column |
| :---: | :---: | :---: | :---: |
| `train.csv` | Training | 8550 | `Text` |
| `val.csv` | Validation (Hyperparameter Tuning) | 1069 | `Text` |
| `test.csv` | Final Evaluation | 1069 | `Text` |
| `sample.csv` | Prediction Task (Truncated Sentences) | 100 | `Truncated Text` |

## 1. Traditional Probabilistic Language Models (`ProbabilisticLM.ipynb`)

This section implements Unigram, Bigram, and Trigram models with smoothing techniques.

### Implemented Smoothing Methods

| Model | Smoothing Technique | Description |
| :---: | :---: | :---: |
| **UnigramModel** | Dirichlet Smoothing | A form of additive smoothing, controlled by the hyperparameter $\alpha$. |
| **NGramModelKN** | Kneser-Ney Smoothing | A state-of-the-art interpolation smoothing technique for N-grams, controlled by a discount factor $D$. |

### Hyperparameter Tuning Results (on `val.csv`)

| Model | Hyperparameter | Optimal Value | Validation Perplexity |
| :---: | :---: | :---: | :---: |
| Unigram (Dirichlet) | $\alpha$ | **1.0000** | 1005.28 |
| Bigram (Kneser-Ney) | $D$ | **0.60** | 254.54 |
| Trigram (Kneser-Ney) | $D$ | **0.75** | 214.69 |

### Test Set Perplexity (Default Parameters)

| Model | Smoothing Parameters | Perplexity (Test Set) |
| :---: | :---: | :---: |
| Unigram | $\alpha=0.1$ (Default) | 1065.29 |
| Bigram (KN) | $D=0.75$ (Default) | 254.23 |
| Trigram (KN) | $D=0.75$ (Default) | **210.86** |

## 2. Neural Feed-Forward Language Models (`NeuralLM.ipynb`)

This section implements an FFN language model using pre-trained word embeddings for the input layer.

### Model Architecture

*   **Embeddings:** Pre-trained **GloVe-50** (`glove-wiki-gigaword-50`).
*   **Context Size:** $N-1$ (e.g., 1 word for Bigram, 2 words for Trigram).
*   **Network:**
    *   Input Layer: Concatenated $N-1$ embeddings (size: $(N-1) \times 50$).
    *   Hidden Layer: `nn.Linear(input, 128)` followed by `nn.ReLU()`.
    *   Output Layer: `nn.Linear(128, vocab_size)`.

### Initial Test Set Perplexity (LR = 0.001)

| Model | Context Size ($N$) | Perplexity (Test Set) |
| :---: | :---: | :---: |
| 1-gram FFN | 0 | 805.84 |
| 2-gram FFN | 1 | **195.57** |
| 3-gram FFN | 2 | 212.27 |

### Learning Rate Tuning (on `val.csv`)

| Model | Optimal Learning Rate | Validation Perplexity |
| :---: | :---: | :---: |
| 1-gram FFN | **0.001** | 806.11 |
| 2-gram FFN | **0.001** | **243.12** |
| 3-gram FFN | 0.001 (based on run) | 247.96 |

## 3. Conclusion & Final Comparison

| Approach | Best Model | Perplexity (Test Set) |
| :---: | :---: | :---: |
| **Traditional** | Trigram (Kneser-Ney) | 210.86 |
| **Neural** | **2-gram FFN + GloVe** | **195.57** |

The **2-gram FFN model** achieved the lowest perplexity, demonstrating superior performance over the best traditional N-gram model. The use of dense word embeddings allows the model to generalize better and capture semantic similarity, which is crucial for reducing perplexity.
```
