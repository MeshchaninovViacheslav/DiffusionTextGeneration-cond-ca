import torch
from tqdm import tqdm

n = 10000
A = torch.randn(n, n).cuda()

for i in tqdm(range(n ** 2)):
    A = A @ A