import torch

def create_matrix_time(tensor, N):
    '''
    Create multiple time series of length N from one time serie.
    :param tensor (torch.Tensor): tensor of shape [time length, 1]
    :param N (int): time length of each time series
    :return:
    new_tensor (torch.Tensor): tensor containing all time series of length N.
    '''
    new_tensor = torch.zeros(N, tensor.size(0) - N + 1)
    for k in range(new_tensor.size(1)):
        new_tensor[:,k] = tensor[k:(k + N)].squeeze()
    return new_tensor

def input_target_split(feature, delta, N):
    '''
    Split the dataset into input and target set
    :param feature (torch.Tensor): tensor to split of shape [time length, dimnension]
    :param delta (int): gap between the target and the input
    :param N (int): number of time series in the matrix
    :return:
    matrix_input (torch.Tensor) input set of shape [time length - delta +, N, dimnesion]
    matrix_target (torch.Tensor) target set of shape [time length - delta +, N, dimnesion]
    '''
    input_ = torch.FloatTensor(feature[:-delta])
    target = torch.FloatTensor(feature[delta:])
    matrix_input = create_matrix_time(input_, N).permute(1,0)
    matrix_target = create_matrix_time(target, N).permute(1,0)
    return matrix_input, matrix_target