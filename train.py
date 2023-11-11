import torch
import argparse
import numpy as  np

from KDE import find_optimal_bandwidth
from ratio import KernelRatioNaive, KernelRatioAlpha, KernelRatioGaussian
from model.energy import Score_network, Weight_network, Energy
from loss.bias import Laplacian
from loss.sliced_score_matching import sliced_VR_score_matching

from scipy.spatial.distance import pdist

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model', type=str, default='KDE')
parser.add_argument('--dim', type=int, default=20)
parser.add_argument('--score_epoch', type=int, default=500)
parser.add_argument('--weight_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_data', type=int, default=1024)
args = parser.parse_args()

if args.device == 'cuda':
    gpu=True
else:
    gpu=False


mean1 = np.concatenate([np.array([0]), np.zeros((args.dim-1,))])
Cov1 = np.eye(args.dim)*np.concatenate([np.array([1.]), np.ones((args.dim-1,))])
mean2 = np.concatenate([np.sqrt([2]), np.zeros((args.dim-1,))])
Cov2 = np.eye(args.dim)*np.concatenate([np.array([1.]), np.ones((args.dim-1,))])

L = torch.linalg.cholesky(torch.tensor(Cov1.astype(np.float32)))
data1 = torch.randn(args.num_data, args.dim) @ L.T + mean1.astype(np.float32)
L = torch.linalg.cholesky(torch.tensor(Cov2.astype(np.float32)))
data2 = torch.randn(args.num_data, args.dim) @ L.T + mean2.astype(np.float32)

TKL = (np.trace(np.linalg.inv(Cov2) @ Cov1) + (mean2-mean1).T @ np.linalg.inv(Cov2) @ (mean2-mean1) - args.dim + np.log(np.linalg.det(Cov2)/np.linalg.det(Cov1)))/2

print(f"True KL divergence: {TKL}")

data1_set = torch.utils.data.TensorDataset(data1)
data2_set = torch.utils.data.TensorDataset(data2)
total_set = torch.utils.data.TensorDataset(torch.cat([data1, data2]))
data1_loader = torch.utils.data.DataLoader(data1_set, batch_size=args.batch_size, shuffle=True)
data2_loader = torch.utils.data.DataLoader(data2_set, batch_size=args.batch_size, shuffle=True)
total_loader = torch.utils.data.DataLoader(total_set, batch_size=args.batch_size, shuffle=True)

l_h = np.linspace(0.2, 1., 20)

if args.model == "KDE":
    opt_h1 = find_optimal_bandwidth(data1, l_h, lik=False, gpu=gpu)
    opt_h2 = find_optimal_bandwidth(data2, l_h, lik=False, gpu=gpu)
    opt_h = (opt_h1 + opt_h2) / 2
    model = KernelRatioNaive(h=opt_h, gpu=gpu)
    model.eps = 1e-100
    model.fit(data1, data2)
    print(f"KL divergence calculated by KDE: {model.kl()}")

elif args.model == "based":
    opt_h1 = find_optimal_bandwidth(data1[:int(len(data1)/4)], l_h, gpu=gpu)
    opt_h2 = find_optimal_bandwidth(data2[:int(len(data2)/4)], l_h, gpu=gpu)
    opt_h = (opt_h1 + opt_h2) / 2
    med_dist1 = np.median(pdist(data1))
    med_dist2 = np.median(pdist(data2))
    med_dist = (med_dist1 + med_dist2) / 2
    model = KernelRatioGaussian(grid_sample=3000, solver='para', para_h=med_dist, para_l=0.1, h=opt_h, gpu=gpu, kmeans=False, einsum_batch=100, reg=0.1, stabilize=True, online=True)
    model.eps = 1e-100
    model.fit(data1, data2)
    print(f"KL divergence calculated by VWKDE model based: {model.kl()}")
    
elif args.model == "free":
    score_model_p1 = Energy(net=Score_network(input_dim=args.dim, units=[300,300], dropout=True)).to(args.device)
    score_model_p2 = Energy(net=Score_network(input_dim=args.dim, units=[300,300], dropout=True)).to(args.device)
    optimizer_sp1 = torch.optim.Adam(score_model_p1.parameters(), lr=1e-4)
    optimizer_sp2 = torch.optim.Adam(score_model_p2.parameters(), lr=1e-4)

    print("Train score models for p1 and p2")

    for epoch in range(args.score_epoch):
        loss1 = 0
        loss2 = 0

        for x in data1_loader:
            x = x[0].to(args.device)
            loss = sliced_VR_score_matching(score_model_p1, x)
            optimizer_sp1.zero_grad()
            loss.backward()
            optimizer_sp1.step()
            loss1 += loss.item() / len(data1_loader)

        for x in data2_loader:
            x = x[0].to(args.device)
            loss = sliced_VR_score_matching(score_model_p2, x)
            optimizer_sp2.zero_grad()
            loss.backward()
            optimizer_sp2.step()
            loss2 += loss.item() / len(data2_loader)

        if epoch % 100 == 99:
            print(f"Epoch: {epoch+1} | Loss1: {loss1}")
            print(f"Epoch: {epoch+1} | Loss2: {loss2}")

    score_model_p1.eval()
    score_model_p2.eval()

    p1_laplacian = Laplacian(score_model_p1)
    p2_laplacian = Laplacian(score_model_p2)

    weight_model = Energy(net=Weight_network(input_dim=args.dim, units=[128,128,128,64], dropout=False)).to(args.device)
    optimizer_w = torch.optim.Adam(weight_model.parameters(), lr=1e-3)

    print("Train a weight model")

    for epoch in range(args.weight_epoch):
        total_loss = 0
        for x in total_loader:
            x = x[0].to(args.device).requires_grad_()
            output = weight_model(x)
            output_gradient = torch.autograd.grad(
                outputs=output, inputs=x,
                grad_outputs=torch.ones_like(output),
                create_graph=True, only_inputs=True
            )[0]
            log_p1 = score_model_p1.minus_forward(x)
            grad_logp1 = torch.autograd.grad(
                outputs=log_p1.view(-1, 1), inputs=x,
                grad_outputs=torch.ones_like(output),
                create_graph=True, only_inputs=True
            )[0]
            log_p2 = score_model_p2.minus_forward(x)
            grad_logp2 = torch.autograd.grad(
                outputs=log_p2.view(-1, 1), inputs=x,
                grad_outputs=torch.ones_like(output),
                create_graph=True, only_inputs=True
            )[0]
            lp_p1 = p1_laplacian.get_laplacian(x) - (grad_logp1**2).sum(1)
            lp_p2 = p2_laplacian.get_laplacian(x) - (grad_logp2**2).sum(1)
            loss = ((((output_gradient)*(-grad_logp1+grad_logp2)).sum(1) + 0.5*(lp_p1-lp_p2))**2).mean()
            optimizer_w.zero_grad()
            loss.backward()
            optimizer_w.step()
            total_loss += loss.item() / len(total_loader)
    
        if epoch % 100 == 99:
            print(f"Epoch: {epoch+1} | Loss: {total_loss}")

    weight_model.eval()
    opt_h1 = find_optimal_bandwidth(data1[:int(len(data1)/4)], l_h, gpu=gpu)
    opt_h2 = find_optimal_bandwidth(data2[:int(len(data2)/4)], l_h, gpu=gpu)
    opt_h = (opt_h1 + opt_h2) / 2
    logWeights1 = weight_model(data1.to(args.device)).detach().cpu().numpy()
    logWeights2 = weight_model(data2.to(args.device)).detach().cpu().numpy()
    kl_model = KernelRatioAlpha(opt_h, gpu=gpu)
    kl_model.fit(data1, data2, np.exp(logWeights1), np.exp(logWeights2))
    print(f"KL divergence calculated by VWKDE model free: {kl_model.kl()}")
