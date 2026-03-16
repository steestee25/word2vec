import numpy as np
import random
import math
from collections import Counter

np.random.seed(42)
random.seed(42)

# Load dataset
print("Loading dataset...")

with open('data/corpus.txt', 'r') as f:
    corpus_text = [line.strip() for line in f if line.strip()]

tokens = [w.lower() for sent in corpus_text for w in sent.split()]

print("Total tokens:", len(tokens))

# Build vocabulary
min_count = 3

word_counts = Counter(tokens)

stop_words = set([
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "for", "to", "with", "is", "are"
])

vocab = [
    word for word, count in word_counts.items()
    if count >= min_count and word not in stop_words
]

word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}

vocab_size = len(vocab)

print("Vocabulary size:", vocab_size)

# Convert tokens → ids
token_ids = []

for w in tokens:
    if w in word_to_id:
        token_ids.append(word_to_id[w])

print("Tokens after filtering:", len(token_ids))

corpus_ids = [token_ids]

# Negative sampling distribution
print("Building negative sampling distribution...")

counts = np.array([word_counts[w] for w in vocab])

dist = counts ** 0.75
dist = dist / np.sum(dist)

cdf = np.cumsum(dist)


def sample_negative(k, forbidden):

    samples = []

    while len(samples) < k:

        r = np.random.rand()
        idx = np.searchsorted(cdf, r)

        if idx in forbidden:
            continue

        samples.append(idx)

    return samples


# Initialize embeddings
embedding_dim = 100

W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
W_out = np.random.randn(vocab_size, embedding_dim) * 0.01

# Helper function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training parameters
window_size = 5
epochs = 300
learning_rate = 0.025
num_negatives = 10

# Training loop 
print("Starting training...")

for epoch in range(epochs):

    total_loss = 0
    pair_count = 0

    for sentence in corpus_ids:

        length = len(sentence)

        for i, center_id in enumerate(sentence):

            start = max(0, i - window_size)
            end = min(length, i + window_size + 1)

            for j in range(start, end):

                if i == j:
                    continue

                context_id = sentence[j]

                v_c = W_in[center_id]
                u_o = W_out[context_id]

                negative_ids = sample_negative(num_negatives, {context_id})
                u_neg = W_out[negative_ids]

                # Forward
                pos_score = np.dot(u_o, v_c)
                pos_sig = sigmoid(pos_score)

                neg_scores = np.dot(u_neg, v_c)
                neg_sig = sigmoid(-neg_scores)

                # Loss
                loss = -math.log(pos_sig + 1e-10) \
                       -np.sum(np.log(neg_sig + 1e-10))

                total_loss += loss
                pair_count += 1

                # Gradients
                grad_v_c = np.zeros_like(v_c)

                coef_pos = pos_sig - 1

                grad_u_o = coef_pos * v_c
                grad_v_c += coef_pos * u_o

                coef_negs = sigmoid(neg_scores)

                grad_u_negs = coef_negs[:, None] * v_c[None, :]

                grad_v_c += np.sum(coef_negs[:, None] * u_neg, axis=0)

                # Parameter updates
                W_in[center_id] -= learning_rate * grad_v_c
                W_out[context_id] -= learning_rate * grad_u_o

                for k, neg_id in enumerate(negative_ids):
                    W_out[neg_id] -= learning_rate * grad_u_negs[k]

    print("Epoch", epoch + 1, "Loss:", total_loss / pair_count)

# Nearest neighbors
def nearest_words_vectorized(word, k=10):
    if word not in word_to_id:
        print("Word not in vocabulary")
        return

    vec = W_in[word_to_id[word]]

    vec_norm = vec / np.linalg.norm(vec)

    W_norm = W_in / np.linalg.norm(W_in, axis=1, keepdims=True)

    sims = W_norm @ vec_norm  

    top_k_idx = np.argsort(-sims)[1:k+1]

    for idx in top_k_idx:
        print(id_to_word[idx], round(sims[idx], 4))

# Test embeddings
test_words = ["neural", "embeddings", "language", "learning"]

for w in test_words:

    print("\nNearest to:", w)
    nearest_words_vectorized(w)