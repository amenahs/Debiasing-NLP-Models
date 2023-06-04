# Amenah Syed
# CS 995 Capstone Project

import numpy as np
from scipy import stats
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer
from typing import List
import matplotlib.pyplot as plt


def weat_effect_size(X: List[str], Y: List[str], A: List[str], B: List[str], embeddings: np.ndarray):
    # Convert words to embedding vectors
    X_embed = np.array([embeddings[word] for word in X if word in embeddings])
    Y_embed = np.array([embeddings[word] for word in Y if word in embeddings])
    A_embed = np.array([embeddings[word] for word in A if word in embeddings])
    B_embed = np.array([embeddings[word] for word in B if word in embeddings])
    
    # Calculate the mean embedding vectors for each set
    X_mean = np.mean(X_embed, axis=0)
    Y_mean = np.mean(Y_embed, axis=0)
    A_mean = np.mean(A_embed, axis=0)
    B_mean = np.mean(B_embed, axis=0)
    
    # Calculate the differences
    diff1 = X_mean - Y_mean
    diff2 = A_mean - B_mean
    
    # Project the differences onto a common space
    X = np.vstack([X_embed, Y_embed, A_embed, B_embed])
    diff = np.dot(X, diff1 - diff2)
    
    # Calculate the effect size
    effect_size = np.mean(diff) / np.std(diff)
    
    return effect_size

def get_debiasing_matrix(target_vectors, attribute_vectors):
    target_vectors = target_vectors.T
    attribute_vectors = attribute_vectors.T
    target_vectors -= np.mean(target_vectors, axis=0)
    attribute_vectors -= np.mean(attribute_vectors, axis=0)

    # Get the dimensions of the target and attribute matrices
    d_target, n_target = target_vectors.shape
    d_attr, n_attr = attribute_vectors.shape

    # Concatenate the target and attribute matrices
    V = np.hstack((target_vectors, attribute_vectors))

    # Compute the SVD of the concatenated matrix V
    U, S, _ = np.linalg.svd(V, full_matrices=False)

    # Construct the debiasing matrix
    S_inv = np.diag(1.0 / S)
    M = np.dot(U[:, :d_target], np.dot(S_inv[:d_target, :d_target], U[:, :d_target].T))

    return M


# Load embeddings
embeddings = {}
with open('glove.6B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word, *vector = line.split()
        embeddings[word] = np.array(vector, dtype=float)

# Load word lists
with open('religious_words.txt', 'r') as f:
    X = [line.strip() for line in f]
with open('non_religious_words.txt', 'r') as f:
    Y = [line.strip() for line in f]
with open('positive_words.txt', 'r') as f:
    A = [line.strip() for line in f]
with open('negative_words.txt', 'r') as f:
    B = [line.strip() for line in f]

# Calculate the WEAT effect size
effect_size = weat_effect_size(X, Y, A, B, embeddings)

# Perform statistical significance test
n = 10000
sample1 = [weat_effect_size(X, Y, np.random.choice(A, size=len(A), replace=False), np.random.choice(B, size=len(B), replace=False), embeddings) for _ in range(n)]
sample2 = [weat_effect_size(np.random.choice(X, size=len(X), replace=False), np.random.choice(Y, size=len(Y), replace=False), np.random.choice(A, size=len(A), replace=False), np.random.choice(B, size=len(B), replace=False), embeddings) for _ in range(n)]
p_value = (np.sum(sample1 > effect_size) + np.sum(sample2 < effect_size)) / n

# Print results
print(f"WEAT effect size: {effect_size:.3f}")
print(f"p-value: {p_value:.3f}")

religious_vectors = np.array([embeddings[word] for word in X if word in embeddings])
negative_vectors = np.array([embeddings[word] for word in B if word in embeddings])

# Apply debiasing
if effect_size > 0:
    target_words = X
    attribute_words = B
else:
    target_words = Y
    attribute_words = A

target_vectors = np.array([embeddings[word] for word in target_words if word in embeddings])
attribute_vectors = np.array([embeddings[word] for word in attribute_words if word in embeddings])

Q = np.array([embeddings[word] for word in X if word in embeddings])
W = np.array([embeddings[word] for word in Y if word in embeddings])
E = np.array([embeddings[word] for word in A if word in embeddings])
R = np.array([embeddings[word] for word in B if word in embeddings])

Q_words = [word for word in X if word in embeddings]
W_words = [word for word in Y if word in embeddings]
E_words = [word for word in A if word in embeddings]
R_words = [word for word in B if word in embeddings]

fig, ax = plt.subplots(figsize=(12, 8))

# Plot the not debiased embeddings of the target and attribute words
ax.scatter(Q[:, 0], Q[:, 1], color='red', label='Target: Religious vs. Non-Religious')
ax.scatter(W[:, 0], W[:, 1], color='blue', label='Target: Non-Religious vs. Religious')
ax.scatter(E[:, 0], E[:, 1], color='green', label='Attribute: Positive')
ax.scatter(R[:, 0], R[:, 1], color='orange', label='Attribute: Negative')

for word, coords in zip(Q_words, Q):
    ax.annotate(word, (coords[0], coords[1]))
for word, coords in zip(W_words, W):
    ax.annotate(word, (coords[0], coords[1]))
for word, coords in zip(E_words, E):
    ax.annotate(word, (coords[0], coords[1]))
for word, coords in zip(R_words, R):
    ax.annotate(word, (coords[0], coords[1]))

# apply autosizing
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

ax.legend()
plt.show()

transformation_matrix = get_debiasing_matrix(target_vectors, attribute_vectors)

debiased_target_vectors = np.dot(target_vectors, transformation_matrix)
debiased_attribute_vectors = np.dot(attribute_vectors, transformation_matrix)

# Convert words to embedding vectors

# Get the debiased embeddings of the target and attribute words
X_embed_debiased = np.dot(np.array([embeddings[word] for word in X if word in embeddings]), transformation_matrix)
Y_embed_debiased = np.dot(np.array([embeddings[word] for word in Y if word in embeddings]), transformation_matrix)
A_embed_debiased = np.dot(np.array([embeddings[word] for word in A if word in embeddings]), transformation_matrix)
B_embed_debiased = np.dot(np.array([embeddings[word] for word in B if word in embeddings]), transformation_matrix)

# Plot the debiased embeddings of the target and attribute words
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_embed_debiased[:, 0], X_embed_debiased[:, 1], color='red', label='Target: Religious vs. Non-Religious')
ax.scatter(Y_embed_debiased[:, 0], Y_embed_debiased[:, 1], color='blue', label='Target: Non-Religious vs. Religious')
ax.scatter(A_embed_debiased[:, 0], A_embed_debiased[:, 1], color='green', label='Attribute: Positive')
ax.scatter(B_embed_debiased[:, 0], B_embed_debiased[:, 1], color='orange', label='Attribute: Negative')

for i, word in enumerate(X):
    ax.annotate(word, (X_embed_debiased[i, 0], X_embed_debiased[i, 1]))
for i, word in enumerate(Y):
    ax.annotate(word, (Y_embed_debiased[i, 0], Y_embed_debiased[i, 1]))
for i, word in enumerate(A):
    ax.annotate(word, (A_embed_debiased[i, 0], A_embed_debiased[i, 1]))
for i, word in enumerate(B):
    ax.annotate(word, (B_embed_debiased[i, 0], B_embed_debiased[i, 1]))

# apply autosizing
fig.tight_layout()

ax.legend()
plt.show()

# Perform statistical significance test
n = 10000
sample1 = [weat_effect_size(Q.reshape(-1), W.reshape(-1), np.random.choice(E.reshape(-1), size=len(E.reshape(-1)), replace=False), np.random.choice(R.reshape(-1), size=len(R.reshape(-1)), replace=False), embeddings) for _ in range(n)]
sample2 = [weat_effect_size(np.random.choice(Q.reshape(-1), size=len(Q.reshape(-1)), replace=False), np.random.choice(W.reshape(-1), size=len(W.reshape(-1)), replace=False), np.random.choice(E.reshape(-1), size=len(E.reshape(-1)), replace=False), np.random.choice(R.reshape(-1), size=len(R.reshape(-1)), replace=False), embeddings) for _ in range(n)]
p_value = (np.sum(sample1 > effect_size) + np.sum(sample2 < effect_size)) / n

# Print results
print(f"WEAT effect size AFTER DEBIAS: {effect_size:.3f}")
print(f"p-value: {p_value:.3f}")
