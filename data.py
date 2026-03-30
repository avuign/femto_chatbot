import re

import torch
import torch.nn as nn


class Tokenizer:
    def __init__(self, dic):
        self.word_to_int = dic
        self.int_to_word = {int: word for word, int in dic.items()}
        self.voc_size = len(dic)

    def encode(self, text):
        words = [
            word if word in self.word_to_int else "<UNK>"
            for word in text_to_words(text)
        ]
        ids = [self.word_to_int[word] for word in words]
        return ids

    def decode(self, ids):
        words = [self.int_to_word[id] for id in ids]
        text = " ".join(words)
        text = re.sub(r'\s+([,.?!"()\'])“', r"\1", text)
        return text


def inputs_and_targets(text, context_size, tokenizer, stride=1):
    encoded_text = tokenizer.encode(text)
    inputs = []
    targets = []
    for i in range(0, len(encoded_text) - context_size - 1, stride):
        input = encoded_text[i : i + context_size]
        inputs.append(input)
        target = encoded_text[i + 1 : i + context_size + 1]
        targets.append(target)
    return torch.tensor(inputs), torch.tensor(targets)


def text_to_words(text):
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    return result


def dic(words):
    special_words = ["<UNK>", "<ENDOFTEXT>"]
    words.extend(special_words)
    word_to_int = {word: i for i, word in enumerate(sorted(list(set(words))))}
    return word_to_int
