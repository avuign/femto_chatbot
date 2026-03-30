from data import Tokenizer, dic, inputs_and_targets, text_to_words

with open("alice.txt") as file:
    text = file.read()

tokenizer = Tokenizer(dic(text_to_words(text)))

print(len(tokenizer.word_to_int))

code = tokenizer.encode("this moment the door of the house opened with AJ")
print(code)

text = tokenizer.decode(code)
print(text)

print(inputs_and_targets(text, 3, tokenizer))

import torch
from self_attention import attn_scores, attn_weights

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)
print(attn_scores(inputs))
print(attn_weights(inputs))
