import torch
import torch.nn as nn
from data import load_data
from model import Femto_Chatbot

CONTEXT_SIZE = 5
EMBEDDING_DIM = 16
NUM_EPOCHS = 20
BATCH_SIZE = 64
LR = 0.01


def train_old(model, X, Y, num_epochs, batch_size, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            batch = {"words": X[i : i + batch_size], "next_word": Y[i : i + batch_size]}
            for training_example, target in zip(
                torch.tensor(batch["words"]), torch.tensor(batch["next_word"])
            ):
                logits = model(training_example)[-1]
                loss = nn.CrossEntropyLoss()(logits, target)
                loss /= batch_size
                epoch_loss += loss.item()
                print(
                    f"  Batch {i // batch_size}/{len(X) // batch_size}, Loss: {loss.item():.4f}"
                )
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


def train(model, X, Y, num_epochs, batch_size, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr)
    print("Converting to tensors...")
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            batch_Y = Y[i : i + batch_size]

            logits = model(batch_X)[:, -1, :]
            loss = nn.CrossEntropyLoss()(logits, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


if __name__ == "__main__":
    filename = "shakespeare.txt"

    print("Loading data...")
    X, Y, dic = load_data(filename, CONTEXT_SIZE)
    print(f"Data loaded: {len(X)} examples, vocab size: {len(dic)}")

    voc_size = len(dic)
    model = Femto_Chatbot(voc_size, EMBEDDING_DIM)

    print("Model created, starting training...")
    train(model, X, Y, NUM_EPOCHS, BATCH_SIZE, LR)

    torch.save(model.state_dict(), "femto_chatbot.pt")
