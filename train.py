import torch
import torch.nn as nn
from config import *
from data import Tokenizer, dic, inputs_and_targets, text_to_words
from model import Femto_GPT


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def train(model, X, Y, voc_size, num_epochs, batch_size, lr):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            optimizer.zero_grad()
            batch_X = X[i : i + batch_size]
            batch_Y = Y[i : i + batch_size]

            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            logits = model(batch_X)

            loss = nn.CrossEntropyLoss()(
                logits.flatten(0, 1),
                batch_Y.flatten(),
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


if __name__ == "__main__":
    filename = FILENAME

    with open(filename) as file:
        text = file.read()

    tokenizer = Tokenizer(dic(text_to_words(text)))

    print("Loading data...")
    X, Y = inputs_and_targets(text, CONTEXT_SIZE, tokenizer, stride=CONTEXT_SIZE)
    print(f"Data loaded: {len(X)} examples, vocab size: {tokenizer.voc_size}")

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
    model.eval()

    print("Model created, starting training...")
    train(model, X, Y, tokenizer.voc_size, NUM_EPOCHS, BATCH_SIZE, LR)

    torch.save(model.state_dict(), "femto_gpt.pt")
