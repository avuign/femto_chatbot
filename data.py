def text_to_input(words, context_size, dic):
    contexts = []
    targets = []
    for i in range(0, len(words) - context_size):
        window = words[i : i + context_size]
        target = words[i + context_size]

        context = []
        for word in window:
            context.append(dic[word])

        contexts.append(context)
        targets.append(dic[target])
    return contexts, targets


def load_data(filename, context_size):
    with open(filename) as text:
        words = text.read().split()

    dic = {word: i for i, word in enumerate(list(set(words)))}
    # decoding = {i: word for word, i in dic.items()}
    #
    input_context, targets = text_to_input(words, context_size, dic)
    return input_context, targets, dic
