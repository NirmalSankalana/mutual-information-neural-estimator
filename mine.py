import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """
    Creates sequential fully-connected layers FC_1->FC_2->...->FC_N.

    Parameters
    ----------
    fc_sizes : int
        Fully connected sequential layer sizes.
    """

    def __init__(self, *fc_sizes: int):
        super().__init__()
        fc_sizes = list(fc_sizes)
        n_classes = fc_sizes.pop()
        classifier = []
        for in_features, out_features in zip(fc_sizes[:-1], fc_sizes[1:]):
            classifier.append(nn.Linear(in_features, out_features))
            classifier.append(nn.ReLU(inplace=True))
        classifier.append(
            nn.Linear(in_features=fc_sizes[-1], out_features=n_classes))
        self.mlp = nn.Sequential(*classifier)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x


class MINE_Net(nn.Module):

    def __init__(self, x_size: int, y_size: int, hidden_units=(100, 50)):
        """
        A network to estimate the mutual information between X and Y, I(X; Y).

        Parameters
        ----------
        x_size, y_size : int
            Number of neurons in X and Y.
        hidden_units : int or tuple of int
            Hidden layer size(s).
        """
        super().__init__()  # Initialise the base class
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        self.fc_x = nn.Linear(x_size, hidden_units[0], bias=False)
        self.fc_y = nn.Linear(y_size, hidden_units[0], bias=False)
        self.xy_bias = nn.Parameter(torch.zeros(hidden_units[0]))
        # the output mutual info is a scalar; hence, the last dimension is 1
        self.fc_output = MLP(*hidden_units, 1)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x, y : torch.Tensor
            Data batches.

        Returns
        -------
        mi : torch.Tensor
            Kullback-Leibler lower-bound estimation of I(X; Y).
        """
        hidden = F.relu(self.fc_x(x) + self.fc_y(y) + self.xy_bias,
                        inplace=True)
        mi = self.fc_output(hidden)
        return mi


class MINE_Trainer:
    """
    Parameters
    ----------
    mine_model : MINE_Net
        A network to estimate mutual information.
    learning_rate : float
        Optimizer learning rate.
    smooth_filter_size : int
        Smoothing filter size. The larger the filter, the smoother but also
        more biased towards lower values of the resulting estimate.
    """

    log2_e = np.log2(np.e)

    def __init__(self, mine_model: nn.Module, learning_rate=1e-3,
                 smooth_filter_size=30):
        if torch.cuda.is_available():
            mine_model = mine_model.cuda()
        self.mine_model = mine_model
        self.optimizer = torch.optim.Adam(self.mine_model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=1e-5)
        self.smooth_filter_size = smooth_filter_size

        self.scheduler = None
        self.mi_history = None
        self.reset()

    def train(self, data_loader, num_epochs):
        """
        Train the MINE model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        num_epochs : int
            Number of training epochs.
        """
        self.mine_model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            for x_batch, y_batch in data_loader:
                if torch.cuda.is_available():
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                self.optimizer.zero_grad()

                mi_estimate = self.mine_model(x_batch, y_batch)
                loss = -mi_estimate.mean()  # Minimize the negative mutual information

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Compute the average loss over the epoch
            avg_loss = total_loss / len(data_loader)

            # Optionally, you can print the loss for each epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

            # Optionally, you can use a learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Optionally, you can store the MI history for later analysis
            self.mi_history.append(avg_loss)

    def reset(self):
        """
        Reset the trainer for re-training.
        """
        self.mi_history = []


def generate_data(num_samples, x_size, y_size):
    x = torch.randn(num_samples, x_size)
    y = torch.randn(num_samples, y_size)
    return x, y


def main():
    print("Executing main")
    x_size = 10
    y_size = 10
    hidden_units = (100, 50)
    mine_model = MINE_Net(x_size=x_size, y_size=y_size,
                          hidden_units=hidden_units)
    mine_trainer = MINE_Trainer(mine_model)

    # Generate synthetic data
    num_samples = 1000
    x, y = generate_data(num_samples, x_size, y_size)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model
    num_epochs = 50
    mine_trainer.train(data_loader, num_epochs)

    # Access the mutual information history
    mi_history = mine_trainer.mi_history

    # Print the last few mutual information estimates
    print("Last few MI estimates:", mi_history[-5:])


if __name__ == "__main__":
    main()
