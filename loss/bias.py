import torch
from torch.autograd.functional import jacobian

class Laplacian():
    def __init__(self, model):
        self.model = model
    
    def get_sum_of_gradients_log_p(self, x):
        log_p = self.model.minus_forward(x)
        log_p_gradient = torch.autograd.grad(
                outputs=log_p, inputs=x,
                grad_outputs=torch.ones_like(log_p),
                create_graph=True, only_inputs=True
            )[0]
        
        return log_p_gradient.sum(0)

    def get_laplacian(self, x):
        return jacobian(self.get_sum_of_gradients_log_p, x).swapaxes(0, 1).diagonal(dim1=-2, dim2=-1).sum(-1)