# import numpy as np
# # 函数向量化
# def sign_func(x):
#     if x>0:
#         return 1
#     elif x<0:
#         return -1
#     else:
#         return 0
#
# n = -4
# res_ori = sign_func(n)
# sign_vector = np.vectorize(sign_func)
# res_vec = sign_vector(n)
# print(res_ori, res_vec)

import torch
import numpy as np


def func(a, b):
    if a > b:
        return a + b
    else:
        return a - b


x = torch.rand((1,3))
y = torch.rand((1,3))
print(x)
print(y)
# print(func(x, y))
print(np.vectorize(func)(x, y))

