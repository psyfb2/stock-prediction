import torch
from torch import Tensor
from torch import nn
from torchinfo import summary


class MLP(nn.Module):
    def __init__(self, dropout: float, seq_len: int, in_features: int,
                 num_classes: int, hidden_layers=[512, 512], 
                 device=None, dtype=None):
        """ Multi-layer Perceptron

        (batch_size, seq_len, in_features) => linearModel => (batch_size, num_classes)

        Args:
            dropout (float): drop out value (between 0 and 1)
            seq_len (int): length of sequences
            in_features (int): number of features for each example
            num_classes (int): number of classes
            hidden_layers (List[int]): hidden layers for MLP
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        all_layers = [in_features * seq_len] + hidden_layers + [num_classes]
        layers = [nn.Flatten()]
        for k in range(len(all_layers) - 2):
            layers.append(nn.Linear(all_layers[k], all_layers[k + 1], **factory_kwargs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(all_layers[-2], all_layers[-1], **factory_kwargs))

        self.classification_head = nn.Sequential(*layers)

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
    batch, seq_len, features = 1024, 16, 64
    num_classes = 2
    hidden_layers = [512, 512]

    mlp = MLP(dropout=0.1, seq_len=seq_len, in_features=features, 
              num_classes=num_classes, hidden_layers=hidden_layers)
    summary(mlp, input_size=(batch, seq_len, features))

    X = torch.rand((batch, seq_len, features))
    y = torch.randint(0, 2, (batch, )).long()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    pred = mlp(X)
    assert pred.size() == torch.Size([batch, num_classes])

    loss = loss_fn(pred, y)
    print("loss:", loss, loss.item())
