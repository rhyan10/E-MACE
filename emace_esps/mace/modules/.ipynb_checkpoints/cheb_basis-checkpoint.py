import torch
import torch
from torch.linalg import eigvals

def chebcompanion(c):
    one = torch.tensor([1], device=c.device)
    c = torch.cat((c, one))
    n = len(c) - 1
    mat = torch.zeros((n, n), device=c.device, dtype=c.dtype)
    scl = torch.tensor([1.] + [torch.sqrt(torch.tensor(0.5))]*(n-1), device=c.device)
    top = mat.view(-1)[1::n+1]
    bot = mat.view(-1)[n::n+1]
    top[0] = torch.sqrt(torch.tensor(0.5, device=c.device))
    top[1:] = 1/2
    bot[...] = top
    mat[:, -1] -= (c[:-1]/c[-1])*(scl/scl[-1])*0.5
    return mat

def calc_roots(c):
    comb = torch.tensor([[-4, 0, -2], [0, 4, 0], [0, 0, -2]], device = c.device, dtype=c.dtype) 
    c = torch.matmul(comb, c) + torch.tensor([0,3,0], device = c.device, dtype=c.dtype)
    a = chebcompanion(c).flip(0).flip(1)
    L = eigvals(a).real 
    sorted_L, _ = torch.sort(L)
    # sorted_L = sorted_L * 10 + torch.tensor([-2573.80789629 -2568.39723458 -2566.77376865], device=c.device)
    return sorted_L