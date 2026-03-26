import math

import torch
import torch.nn as nn


# Model
class Femto_Chatbot(nn.Module):
    def __init__(self, voc_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.E = nn.Embedding(voc_size, embedding_dim)

        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

        self.P = nn.Linear(embedding_dim, voc_size)

    def attention_scores_matrix(self, Q_vector, K_vector):
        n = len(Q_vector)
        matrix = torch.zeros(n, n, dtype=torch.float32)

        for i in range(0, n):
            for j in range(0, i + 1):
                matrix[i, j] = torch.dot(Q_vector[i], K_vector[j])
            for j in range(i + 1, n):
                matrix[i, j] = -torch.inf
        matrix *= 1 / math.sqrt(self.embedding_dim)
        matrix = nn.functional.softmax(matrix, dim=-1)

        return matrix

    def forward_old(self, tokens):

        n = len(tokens)

        Q_vector = torch.zeros(n, self.embedding_dim, dtype=torch.float32)
        K_vector = torch.zeros(n, self.embedding_dim, dtype=torch.float32)
        V_vector = torch.zeros(n, self.embedding_dim, dtype=torch.float32)

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

    def forward(self, tokens):
        seq_len = tokens.shape[-1]
        x = self.E(tokens)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.embedding_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        x = weights @ V
        logits = self.P(x)
        return logits
