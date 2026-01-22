from torch import nn



class WineQualityModel_0(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_layer = nn.Linear(in_features=input_dim, out_features=33) # we have 11 features and 1 label. We are taking 4 times the input features as hidden units 
        self.hidden_layer_1 = nn.Linear(in_features=33, out_features=33)
        self.hidden_layer_2 = nn.Linear(in_features=33, out_features=33)
        self.output_layer = nn.Linear(in_features=33, out_features=output_dim) # output layer will have 6 unit for the final 1 label.
        self.relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x