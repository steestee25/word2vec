# Word2Vec 

A pure NumPy implementation of Word2Vec using the **Skip-gram** model with **Negative Sampling (SGNS)**, trained on a custom text corpus.

## Overview

This project trains word embeddings from scratch - no deep learning frameworks. Given a corpus of text, the model learns dense vector representations of words such that semantically similar words end up close together in the embedding space.

## How It Works

The model uses the **Skip-gram objective**: for each center word in a sentence, predict its surrounding context words within a sliding window. Training uses **Negative Sampling** to make this efficient - rather than normalizing over the full vocabulary at each step, the model contrasts each positive (center, context) pair against a small number of randomly sampled "negative" words.

### Architecture

- **Input embeddings** `W_in` - one vector per vocabulary word (the learned representations)
- **Output embeddings** `W_out` - a second matrix used only during training
- Both matrices are initialized with small random values and updated via gradient descent

### Loss Function

For each (center, context) pair, the loss is:

```
L = -log σ(u_o · v_c) - Σ log σ(-u_k · v_c)
```

where `v_c` is the center word vector, `u_o` is the positive context vector, and `u_k` are the negative sample vectors.

## Project Structure

```
.
├── data/
│   └── corpus.txt         # One sentence per line
└── word2vec.py            # Full training script
```

## Configuration

| Parameter | Value | Description |
|---|---|---|
| `embedding_dim` | 100 | Size of each word vector |
| `window_size` | 5 | Context window radius |
| `epochs` | 300 | Training passes over the corpus |
| `learning_rate` | 0.025 | SGD step size |
| `num_negatives` | 10 | Negative samples per positive pair |
| `min_count` | 3 | Minimum word frequency to include in vocabulary |

## Usage

1. Place your corpus in `data/corpus.txt` (one sentence per line).
2. Run the training script:

```bash
python word2vec.py
```

Training prints the average loss per epoch. After training, the script evaluates the embeddings by finding the nearest neighbors of a set of test words.

## Vocabulary

Stop words (`a`, `an`, `the`, `and`, `or`, etc.) are excluded, and words appearing fewer than `min_count` times are filtered out. The negative sampling distribution uses a smoothed unigram distribution raised to the **0.75 power**, which reduces the dominance of very frequent words.

## Nearest Neighbor Evaluation

After training, similarity is computed via **cosine similarity** between normalized `W_in` vectors:

```python
nearest_words_vectorized("neural", k=10)
```

Default test words: `neural`, `embeddings`, `language`, `learning`.

## Requirements

```
numpy
```

Install with:

```bash
pip install numpy
```

## Example Output

Nearest neighbors after training, computed via cosine similarity:

```
Nearest to: neural
  networks 0.5911
  training 0.4821
  like 0.3596
  deep 0.3594
  natural 0.3203
  learning 0.3141
  language 0.261
  tasks 0.2526
  models 0.2351
  machine 0.187

Nearest to: embeddings
  word 0.554
  words 0.4748
  networks 0.4471
  training 0.3943
  processing 0.3551
  learning 0.2605
  language 0.2594
  deep 0.2347
  natural 0.2304
  performance 0.169
```

The results show the model has picked up meaningful semantic relationships - `neural` clusters with NLP/ML terminology, and `embeddings` associates closely with `word` and `words`, reflecting how the term is used in context.

## Notes

- All randomness is seeded (`numpy` seed `42`, `random` seed `42`) for reproducibility.
