import torch


def attn_scores(input_seq):
    scores = input_seq @ input_seq.T
    return scores


def attn_weights(input_seq):
    scores = attn_scores(input_seq)
    weights = torch.softmax(scores, dim=-1)
    return weights


def context_vector(input_seq):
    weights = attn_weights(input_seq)
    context_vec = weights @ input_seq
    return context_vec
