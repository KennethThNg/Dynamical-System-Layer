# Implement dynamical system layer
import torch
import torch.nn as nn
import math

#Implement Linear Part of the system
class LinearODELayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False, weight=None):
        '''
        The aim is to recreate a dynamical system as an inner layer. In the dynamical system is of the form
                dy(t) = F(y(t))dt
        where F is the functional we need to derive. In this module, we assume that the function is linear
        of the form F(y) = Wy where W is the  coefficient matrix to learn.

        Linear part of the dynamical system

        Parameters
        ----------
        :param in_features (int): input dimension
        :param out_features (int): output dimension
        :param bias (bool): determine if bias is added in the model, default = False
        :param weight (torch.Tensor): weight of shape [in_feature, out_feature] of the layer, default=None

        Examples
        ---------
        >>> import torch
        >>> import torch.nn as nn
        >>> from Layer import LinearODELayer
        >>> in_dim = 2
        >>> x = torch.Tensor([[1., 0.]])
        >>> w = torch.Tensor([[1.,1.], [-0.01, 1.]])
        >>> lin = LinearODELayer(in_dim, in_dim, bias=False, weight=w)
        >>> pred = lin(x)
        >>> print(pred)
        tensor([[1., 1.]], grad_fn=<MmBackward>)

        '''
        super(LinearODELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weight is None:
            self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        else:
            self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)


    # the forward step
    def forward(self, x):
        '''
        Compute the prediction of the model. In particular, we obtain the computation
                            y = Hx
        where H is the (in_dim x in_dim) tensor matrix.

        Parameters
        ----------
        :param x (torch.Tensor): input data. Tensor of dimension [batch_size, in_feature]

        Returns
        --------
        :return:
        output (torch.Tensor): output data. Tensor of dimension [batch_size, out_feature]
        '''
        output = x.mm(self.weight)
        if self.bias is not None:
            output += self.bias
        return output
    # the backward function is not necessary, sincewe have only used
    # functions from pytorch (in particular, .reshape and .mm)

    # this function is inherited from the pytorch class Linear
    def reset_parameters(self):
        '''
        reset the weights of the model
        :return: initialized weight
        '''
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

# Implement Non Linear part of the dynamical system
class NonLinearODELayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, weight=None):
        '''
        The aim is to recreate a dynamical system as an inner layer. In the dynamical system is of the form
                dy(t) = F(y(t))dt
        where F is the functional we need to derive. In this module, we assume that the function is linear
        of the form F(y) = tr(y)Wy where W is the coefficient tensor of three dimension to learn and tr() is
        the transpose function.

        Non Linear dynamical system layer, in particular, this is the quadratic part

        Parameters
        ----------
        :param in_dim (Int): Input dimension
        :param hid_dim (Int): hidden dimension
        :param out_dim (Int): output dimension
        :param weight (torch.Tensor): weight of shape [in_dim, hid_dim , out_dim] of the layer, default=None

        Examples
        --------
        >>> import torch
        >>> import torch.nn as nn
        >>> from Layer import NonLinearODELayer
        >>> in_dim = 3
        >>> x = torch.Tensor([[0., 1., 1.05]])
        >>> q1 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        >>> q2 = [[0., 0., -0.01], [0., 0., 0.], [0., 0. ,0.]]
        >>> q3 = [[0., 0., 0.], [0., 0.01, 0.], [0. ,0. ,0.]]
        >>> w = torch.Tensor([q1, q2, q3])
        >>> nl = NonLinearODELayer(in_dim, in_dim, in_dim, w)
        >>> pred = nl(x)
        >>> print(pred)
        tensor([[0.0000, 0.0000, 0.0100]], grad_fn=<TBackward>)
        '''
        super(NonLinearODELayer, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        if weight is None:
            self.weight = nn.Parameter(torch.Tensor(self.in_dim, self.hid_dim, self.out_dim))
        else:
            self.weight = nn.Parameter(weight)

    def forward(self, x):
        '''
        Compute the model prediction (forward pass)
        :param x (torch.Tensor): input data. Tensor of dimension [batch size, in_dim]
        :return:
        out (torch.Tensor): predicted values. Tensor of dimension [batch size, out_dim]
        '''
        xBx = x.matmul(self.weight).matmul(x.t()).permute(2,1,0)
        out = torch.diagonal(xBx).t()
        return out

    def reset_parameters(self):
        '''
        :return: initialized weight
        '''
        nn.init.kaiming_uniform_(self.weight, a=math.srqt(5))
