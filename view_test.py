import torch

# a = torch.range(1, 16)
# b = torch.range(17, 32)
# print('Shape ', a.shape, " Tensor : ", a)
# print('Shape ', b.shape, " Tensor : ", b)
# c = torch.stack((a, b),)
# print(c)
#
# print('Shape ', a.shape, " Tensor : ", a)
#
# c = c.permute(1, 0)
# print(c)

random_t = torch.rand(1, 64, 10)
print(random_t.shape)
random_t = random_t.view(-1, 10)
print(random_t.shape)