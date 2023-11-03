import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_classes: int = 2, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, num_classes)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_batch) -> torch.Tensor:
        embeddings = self.embedding(input_batch)

        output, _ = self.lstm(embeddings)
        output = output.max(dim=1)[0]
        output = self.activation(output)
    
        output = self.linear_1(output)
        output = self.dropout(output)
        output = self.activation(output)

        output = self.linear_2(output)
        return output
