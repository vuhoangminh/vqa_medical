import torch
import torch.nn as nn
from copy import deepcopy

a=torch.rand((1,1,4,4))
b=nn.Conv2d(1,1,3,padding=1,bias=False)
d=nn.Conv2d(1,1,3,padding=1,bias=False)
loss1=torch.rand((1,1,4,4))
loss2=torch.rand((1,1,4,4))

c=b(a)
c2=c.detach()
e=d(c)
e2=d(c2)
e2.backward(loss)
print('detach c:', b.weight.grad, d.weight.grad)
d.weight.grad.zero_()

e.backward(loss)
print('first bp:', b.weight.grad, d.weight.grad)
b.weight.grad.zero_()
d.weight.grad.zero_()

c=b(a)
e=d(c)
e.backward(loss)
print('second bp:', b.weight.grad, d.weight.grad)