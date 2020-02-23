import torch.nn as nn
from DynamicalSystem.ODEModel import NNODEModel

class NeuralNetODEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        Build a neural network with two fully connected layers and the dynamical system layer
        :param input_dim (int): windows time
        :param hidden_dim (int): number of hidden units
        :param output_dim (int): windows time
        '''
        super(NeuralNetODEModel,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        #layers creation
        self.encoder = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.ode = NNODEModel(in_dim=self.hidden_dim, hid_dim=self.hidden_dim, out_dim=self.hidden_dim)
        self.decoder = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)

    def forward(self, x):
        '''
        Forward pass of the model
        :param x (torch.Tensor): data time point, tensor of dimension [batch size, time length]
        :return:
        out (torch.Tensor): predicted data, tensor of dimension [batch size, time length]
        '''
        out = self.encoder(x)
        out = self.ode(out)
        out = self.decoder(out)
        return out

class FullyLinearLayerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        Neural net with linear layer as latent layer
        :param input_dim (int): windows time
        :param hidden_dim (int): number of hidden units
        :param output_dim (int): windows time
        '''
        super(FullyLinearLayerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        #layers creation
        self.encoder_layer = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.latent_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.decoder_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)

    def forward(self, x):
        '''
        Forward pass of the model
        :param x (torch.Tensor): data time point, tensor of dimension [batch size, time length]
        :return:
        out (torch.Tensor): predicted data, tensor of dimension [batch size, time length]
        '''
        out = self.encoder_layer(x)
        out = self.latent_layer(out)
        out = self.decoder_layer(out)
        return out