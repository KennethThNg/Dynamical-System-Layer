""" Testing for the Layers of the dynamical system"""

import torch
from numpy.testing import assert_almost_equal
from ODELayers import LinearODELayer
from ODELayers import NonLinearODELayer
import pytest



#Testing the model
def test_LinearODELayer():
    in_dim_lin = 2
    w_lin = torch.Tensor([[1.,1.], [-0.01, 1.]])
    x_lin = torch.Tensor([[1., 0.]])
    lin = LinearODELayer(in_dim_lin, in_dim_lin, False, w_lin)
    y_lin = torch.Tensor([[1., 1.]]).numpy()
    pred_lin = lin(x_lin).detach().numpy()
    assert_almost_equal(pred_lin, y_lin)

def test_NonLinearODELayer():
    in_dim_nl = 3
    q1 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    q2 = [[0., 0., -0.01], [0., 0., 0.], [0., 0. ,0.]]
    q3 = [[0., 0., 0.], [0., 0.01, 0.], [0. ,0. ,0.]]
    w_nl = torch.Tensor([q1, q2, q3])
    x_nl = torch.Tensor([[0., 1., 1.05]])
    nl = NonLinearODELayer(in_dim_nl, in_dim_nl, in_dim_nl, w_nl)
    y_nl = torch.Tensor([[0.0000, 0.0000, 0.0100]]).numpy()
    pred_nl = nl(x_nl).detach().numpy()
    assert_almost_equal(pred_nl, y_nl)

