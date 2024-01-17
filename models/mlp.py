import torch
from torch import Tensor
from torch import nn
from torchinfo import summary

from models.positional_encoding import PositionalEncoding


class MLP(nn.Module):
    def __init__(self, seq_len: int, in_features: int, num_classes: int, 
                 device=None, dtype=None):
        """ Multi-layer Perceptron

        (batch_size, seq_len, in_features) => linearModel => (batch_size, num_classes)

        Args:
            seq_len (int): length of sequences
            in_features (int): number of features for each example
            num_classes (int): number of classes
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # create classification head
        self.classification_head = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features*seq_len, 512, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(512, 512, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Feed x through MLP

        Args:
            x (Tensor): Tensor with shape (batch, seq_len, feature) 

        Returns:
            Tensor: output with shape (batch, num_classes)
        """
        logits = self.classification_head(x)
        return logits


if __name__ == "__main__":
    # test transformer with dummy input
    batch, seq_len, features = 64, 128, 32
    num_classes = 2

    mlp = MLP(seq_len, features, num_classes)
    summary(mlp, input_size=(batch, seq_len, features))

    X = torch.rand((batch, seq_len, features))
    y = torch.randint(0, 2, (batch, )).long()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    pred = mlp(X)
    assert pred.size() == torch.Size([batch, num_classes])

    loss = loss_fn(pred, y)
    print("loss:", loss, loss.item())
