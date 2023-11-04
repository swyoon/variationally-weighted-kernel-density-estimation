import torch

def sliced_VR_score_matching(energy_net, samples, noise=None, detach=False, noise_type='gaussian'):
    """ Sliced score matching loss from:
        https://github.com/ermongroup/sliced_score_matching/
    """
    samples.requires_grad_(True)
    if noise is None:
        vectors = torch.randn_like(samples)
        if noise_type == 'radermacher':
            vectors = vectors.sign()
        elif noise_type == 'gaussian':
            pass
        else:
            raise ValueError("Noise type not implemented")
    else:
        vectors = noise

    logp = -energy_net(samples).sum()
    grad1 = torch.autograd.grad(logp, samples, create_graph=True)[0]
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.norm(grad1, dim=-1) ** 2 * 0.5
    if detach:
        loss1 = loss1.detach()
    grad2 = torch.autograd.grad(gradv, samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)
    if detach:
        loss2 = loss2.detach()

    loss = (loss1 + loss2).mean()
    return loss

def ssm_loss(model, x, v):
        """SSM loss from
        Sliced Score Matching: A Scalable Approach to Density and Score Estimation
        The loss is computed as
        s = -dE(x)/dx
        loss = vT*(ds/dx)*v + 1/2*(vT*s)^2
        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises
        Returns:
            SSM loss
        """
        x = x.unsqueeze(0).expand(1, *x.shape) # (n_slices, b, ...)
        x = x.contiguous().view(-1, *x.shape[2:]) # (n_slices*b, ...)
        x = x.requires_grad_()
        score = model.score(x) # (n_slices*b, ...)
        sv    = torch.sum(score * v) # ()
        loss1 = torch.sum(score * v, dim=-1) ** 2 * 0.5 # (n_slices*b,)
        gsv   = torch.autograd.grad(sv, x, create_graph=True)[0] # (n_slices*b, ...)
        loss2 = torch.sum(v * gsv, dim=-1) # (n_slices*b,)
        loss = (loss1 + loss2).mean() # ()
        return loss

def get_random_noise(x, std=1, device='cuda'):
        """Sampling random noises
        Args:
            x (torch.Tensor): input samples
            n_slices (int, optional): number of slices. Defaults to None.
        Returns:
            torch.Tensor: sampled noises
        """
        v = torch.randn((1,)+x.shape, dtype=x.dtype, device=device)
        v = v.view(-1, *v.shape[2:])*std # (n_slices*b, 2)
            
        return v