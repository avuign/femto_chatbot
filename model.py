import math

import torch
import torch.nn as nn


# Model
class CharModel(nn.Module):
    def __init__(self, context_size, voc_size, embedding_dim, hidden_dims):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.E = nn.Embedding(voc_size, embedding_dim)

        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

        self.P = nn.Linear(embedding_dim, voc_size)

    def attention_scores_matrix(self, Q_vector, K_vector):
        n = len(Q_vector)
        dim = len(Q_vector[0])
        matrix = torch.zeros(shape=(n, n), dtype=torch.float32)

        for i in range(0, n):
            for j in range(0, i + 1):
                matrix[i, j] = torch.dot(Q_vector[i], K_vector[j])
            for j in range(i + 1, n):
                matrix[i, j] = -torch.inf
        matrix *= 1 / math.sqrt(self.embedding_dim)
        matrix = nn.functional.softmax(matrix, dim=-1)

        return matrix

    def forward(self, tokens):

        n = len(tokens)

        Q_vector = torch.zeros(shape=(n, self.embedding_dim), dtype=torch.float32)
        K_vector = torch.zeros(shape=(n, self.embedding_dim), dtype=torch.float32)
        V_vector = torch.zeros(shape=(n, self.embedding_dim), dtype=torch.float32)

        for i in range(0, n):
            x = tokens[i]
            x = self.E(x)
            Q_vector[i] = self.W_Q(x)
            K_vector[i] = self.W_K(x)
            V_vector[i] = self.W_V(x)

        attention_matrix = self.attention_scores_matrix(Q_vector, K_vector)
        x = torch.matmul(attention_matrix, V_vector)

        logits = self.P(x)

        return logits
