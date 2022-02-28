import torch
from utils.chamfer_distance.chamfer_distance import ChamferDistance
from utils.chamfer.dist_chamfer import chamferDist
from torch.autograd import Variable
from models.loss import PointLoss_test
import time
from torch.autograd import gradcheck

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# p1 = torch.rand([32, 2048, 3])
# p2 = torch.rand([32, 2048, 3], device=device)
points1 = Variable(torch.rand([32, 2048, 3], device=device), requires_grad=True)
points2 = Variable(torch.rand([32, 2048, 3], device=device), requires_grad=True)

chamfer_dist = ChamferDistance()
# time_start = time.time()
dist1, dist2 = chamfer_dist(points1, points2)
loss = (torch.mean(dist1)) + (torch.mean(dist2))
# time_end = time.time()
# print(time_end - time_start)
# print(loss)
loss.backward()
print(points1.grad)
print('***************')

# points1.zero_grad()
points1.grad.data.zero_()

chamfer_dist = chamferDist()
# time_start = time.time()
cost = chamfer_dist(points1, points2)
loss = (torch.mean(cost[0])) + (torch.mean(cost[1]))
# time_end = time.time()
# print(time_end - time_start)
# print(loss)
loss.backward()
print(points1.grad)
print('***************')

# loss.backward()
# print(points1.grad, points2.grad)
# time_start = time.time()
loss, _, _ = PointLoss_test(points1, points2)
# loss = torch.sum(points1 + points2)
# time_end = time.time()
# print(time_end - time_start)
# print(loss)
loss.backward()
print(points1.grad)



# # import numpy as np
# #
# # chamfer_dist = ChamferDistance()
# #
# # def test_interpolation_grad():
# #     p1 = torch.rand([3, 100, 3], requires_grad=True).cuda().float()
# #     p2 = torch.rand([3, 100, 3], requires_grad=True).cuda().float()
# #     inputss = (p1, p2)
# #
# #     test = gradcheck(chamfer_dist, inputss, eps=1e-1)
# #     print(test)
# #
# #
# # test_interpolation_grad()
# #


# import torch
# from torch.autograd import Variable as V
#
# m = V(torch.FloatTensor([[2, 3]]), requires_grad=True)   # 注意这里有两层括号，非标量
# n = V(torch.zeros(1, 2))
# n[0, 0] = m[0, 0] ** 2
# n[0, 1] = m[0, 1] ** 3
# n.backward(torch.Tensor([[1,1]]), retain_graph=True)
# print(m.grad)
