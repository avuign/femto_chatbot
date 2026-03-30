import torch
import torch.nn as nn
from config import *
from data import Tokenizer, dic, text_to_words
from model import Femto_GPT


def generate_new_sentences(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


if __name__ == "__main__":
    with open("alice.txt") as file:
        text = file.read()
    tokenizer = Tokenizer(dic(text_to_words(text)))

    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model = Femto_GPT(
        tokenizer.voc_size,
        N_LAYERS,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        CONTEXT_SIZE,
        N_HEADS,
        DROP_RATE,
        QKV_BIAS,
    )
    model.eval()  # disable dropout

    out = generate_new_sentences(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=CONTEXT_SIZE,
    )

    print("Output:", out)
    print("Output length:", len(out[0]))
