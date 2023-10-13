import torch
from tqdm import tqdm

n = 1000
A = torch.randn(n, n).cuda()

for i in tqdm(range(n ** 3)):
    A = A @ A
    if i % n == 0:
        A = torch.random.randn(n, n).cuda()