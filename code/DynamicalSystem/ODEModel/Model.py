#Implement the neural net with the model
import torch
import torch.nn as nn

from ..ODELayers import LinearODELayer, NonLinearODELayer

#Linear model for the pendulum
class LinearODEModel(nn.Module):
    def __init__(self, in_feature, out_feature, weight=None, bias=False, reset=True):
        '''
        Build a neural net containing the linear dynamical system layer. We build a simple neural net with a linear
        fully connected layer implemented in the file ODELayers. In this case, the model is simply a linear model
        y = Wx where W is the 2-dimension matrix to learn.

        Parameters:
        -----------
        :param in_feature (int): input dimension
        :param out_feature (int): output dimension
        :param bias (bool): determine if bias is needed in the model. default = False
        :param weight (torch.Tensor): weight tensor of shape [in_feat, out_feat], default = None
        :param reset (bool): determine if the parameters is init randomly, default = True

        Examples:
        ---------
        >>> import torch
        >>> from ODEModel import LinearODEModel
        >>> in_dim = 2
        >>> x = torch.Tensor([[1., 0.]])
        >>> w = torch.Tensor([[1.,1.], [-0.01, 1.]])
        >>> lin = LinearODEModel(in_dim, in_dim, weight=w, bias=False, reset=False)
        >>> pred = lin(x)
        >>> print(pred)
        tensor([[1., 1.]], grad_fn=<MmBackward>)
        '''
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        self.weight = weight
        self.reset = reset
        self.ode = LinearODELayer(self.in_feature, self.out_feature, self.bias, self.weight)
        if self.reset:
            self.ode.reset_parameters()

    def forward(self, x):
        '''
        Forward pass
        :param x (torch.Tensor): input data. Tensor of dimension [batch size, in_feature]
        :return:
        out (torch.Tensor): output feature. Tensor of dimension [batch size, out_feature]
        '''
        out = self.ode(x)
        return out

#General model for the Lorenz attractor
class NNODEModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, weight = None, reset=True, bias=False):
        '''
        Build a neural net containing the linear dynamical system layer and the non-linear dynamical system layer. This is
        the final model to be implemented. In order to learn the function F in the dynamical system, we use the second order
        Taylor expansion of the form F(y) = Wy + tr(y)Hy where W is the 2-dimension tensor and H is the 3-dimension tensor
        to learn. Since the model is validate, we use this model to forecast a real time series..

        Parameters:
        ----------
        :param in_dim (int): Input dimension
        :param hid_dim (int): Hidden dimension
        :param out_dim (int): Output dimension
        :param bias (bool): determine if bias is needed in the model
        :param weight (dict): dictionary containing a 2-dimension weight (torch.Tensor) to implement the linear part
        and a 3-dimension weight (torch.Tensor) for the quadratic part of the system. default = None
        :param reset (bool): determine the initialization of the weight parameter. default = None

        Example:
        -------
        >>> import torch
        >>> from ODEModel import NNODEModel
        >>> in_dim = 3
        >>> x = torch.Tensor([[0., 1., 1.05]])
        >>> lin_weight = torch.Tensor([[0.9, 0.1, 0.], [0.3, 0.99, 0.], [0., 0., 0.9733]])
        >>> q1 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        >>> q2 = [[0., 0., -0.01], [0., 0., 0.], [0., 0. ,0.]]
        >>> q3 = [[0., 0., 0.], [0., 0.01, 0.], [0. ,0. ,0.]]
        >>> nl_weight = torch.Tensor([q1, q2, q3])
        >>> w = {'linear': lin_weight, 'quadratic': nl_weight}
        >>> ode = NNODEModel(in_dim, in_dim, in_dim, w, reset=False)
        >>> pred = ode(x)
        >>> print(pred)
        tensor([[0.3000, 0.9900, 1.0319649]], grad_fn=<AddBackward0>)
        '''
        super().__init__()
        #Dimension
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.bias = bias
        if weight is not None:
            self.lin_weight = weight['linear']
            self.nl_weight = weight['quadratic']
        else:
            self.lin_weight = None
            self.nl_weight = None
            
        #Layer
        self.lin_ode = LinearODELayer(self.in_dim, self.out_dim, self.bias, self.lin_weight)
        self.nl_ode = NonLinearODELayer(self.in_dim, self.hid_dim, self.out_dim, self.nl_weight)

        #weight init
        if reset:
            self.lin_ode.weight.data.uniform_(-0.1,0.1)
            if self.bias:
                self.lin_ode.bias.data.uniform_(-0.1,0.1)
            self.nl_ode.weight.data.uniform_(-0.001,0.001)
            


    def forward(self, x):
        '''
        Forward Pass
        :param x (torch.Tensor): input data. Tensor of dimension [batch size, in_feature]
        :return:
        out (torch.Tensor): output feature. Tensor of dimension [batch size, out_feature]
        '''
        out = self.lin_ode(x) + self.nl_ode(x)
        return out
