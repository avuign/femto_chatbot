import torch
import torch.nn as nn


def train(model, X, Y, num_epochs, batch_size, lr):

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
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
