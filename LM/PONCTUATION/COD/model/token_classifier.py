import torch.nn as nn

class TokenClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation='relu', num_layers=1, dropout=0.0):
        super(TokenClassifier, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise ValueError("Activation function must be 'relu', 'sigmoid', or 'tanh'.")
        
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError("Activation function must be 'relu', 'sigmoid', or 'tanh'.")
            layers.append(nn.Dropout(dropout))
            
        layers.append(nn.Linear(hidden_size, num_classes))
        layers.append(nn.Sigmoid()) 
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)