import torch.nn as nn

class FraudDetectorModule(nn.Module):
    def __init__(self, num_features, num_hidden_first=60, dropout_rate=0.3):
        super(FraudDetectorModule, self).__init__()
        self.activation = nn.ReLU()

        self.layer1 = nn.Linear(num_features, num_hidden_first)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(num_hidden_first, num_hidden_first // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(num_hidden_first // 2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        return x