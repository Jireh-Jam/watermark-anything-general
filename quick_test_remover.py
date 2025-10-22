import torch
from watermark_anything.attacks.learned_remover import LearnedRemover
m = LearnedRemover(in_channels=3, base_ch=32, depth=4, tanh_scale=0.05)
x = torch.randn(4,3,256,256)
y = m(x)
print('out', y.shape, 'max_abs', float(y.abs().max()))