'''Test the implemented model'''
import torch
from numpy.testing import assert_almost_equal
from ODEModel import LinearODEModel, NNODEModel

import pytest

#Testing the model
def test_LinearODEModel():
    in_dim_lin = 2
    y_lin = torch.Tensor([[1., 1.]]).numpy()
    x_lin = torch.Tensor([[1., 0.]])
    w_lin = torch.Tensor([[1.,1.], [-0.01, 1.]])
    lin = LinearODEModel(in_dim_lin, in_dim_lin, weight=w_lin, bias=False, reset=False)
    pred_lin = lin(x_lin).detach().numpy()
    assert_almost_equal(pred_lin, y_lin)

def test_NNODEModel():
    in_dim_nl = 3
    y_nl = torch.Tensor([[0.3000, 0.9900, 1.0319649]]).numpy()
    x_nl = torch.Tensor([[0., 1., 1.05]])
    lin_weight = torch.Tensor([[0.9, 0.1, 0.], [0.3, 0.99, 0.], [0., 0., 0.9733]])
    q1 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    q2 = [[0., 0., -0.01], [0., 0., 0.], [0., 0. ,0.]]
    q3 = [[0., 0., 0.], [0., 0.01, 0.], [0. ,0. ,0.]]
    nl_weight = torch.Tensor([q1, q2, q3])
    w = {'linear': lin_weight, 'quadratic': nl_weight}
    ode = NNODEModel(in_dim_nl, in_dim_nl, in_dim_nl, w, reset=False)
    pred_ode = ode(x_nl).detach().numpy()
    assert_almost_equal(pred_ode, y_nl)

