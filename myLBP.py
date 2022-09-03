import math

from torch import nn
from torch.autograd import Function
import torch

from torch.utils.cpp_extension import load

LBP_cuda = load(
    "LBP_cuda",
    ["LBP_cuda.cpp", "mycuda.cu"],
    verbose=True,
    build_directory="./",
)
# help(LBP_cuda)

import LBP_cuda

torch.manual_seed(1)


class LBPfucntion(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        kernels,
        projection_map,
        Output0,
        Output,
        gradIN,
        gradKernels,
        Image_grad_X,
        Image_grad_Y,
        padwh,
        ALPHA,
        LR,
        h_filter,
    ):
        gradIN[...] = 0
        gradKernels[...] = 0
        Image_grad_X[...] = 0
        Image_grad_Y[...] = 0
        Output0[...] = 0
        Output[...] = 0
        
        
        Output, input_padded = LBP_cuda.forward(
            input.contiguous(),
            padwh,
            kernels.contiguous(),
            projection_map.contiguous(),
            Output.contiguous(),
            Output0.contiguous(),
            Image_grad_Y.contiguous(),
            Image_grad_X.contiguous(),
        )

        Image_grad_Y[:, :, 1:, :] = (
            input_padded[:, :, 1:, :] - input_padded[:, :, :-1, :]
        )
        Image_grad_Y[:, :, 0, :] = 0
        Image_grad_X[:, :, :, 1:] = (
            input_padded[:, :, :, 1:] - input_padded[:, :, :, :-1]
        )
        Image_grad_X[:, :, :, 0] = 0

        ctx.save_for_backward(
            input,
            kernels,
            projection_map,
            Output0,
            Output,
            gradIN,
            gradKernels,
            Image_grad_X,
            Image_grad_Y,
        )
        ctx.in1 =   (padwh,
        input_padded,
        ALPHA,
        LR,
        h_filter,)
        return Output

    @staticmethod
    def backward(ctx, grad_out):
        # print("grad_out.mean = ", grad_out.mean())
        
        (
            input,
            kernels,
            projection_map,
            Output0,
            Output,
            gradIN,
            gradKernels,
            Image_grad_X,
            Image_grad_Y,
        ) = ctx.saved_tensors
        (padwh,
        input_padded,
        ALPHA,
        LR,
        h_filter) = ctx.in1
        gradIN, gradKernels = LBP_cuda.backward(
            grad_out.contiguous(),
            Output0.contiguous(),
            Image_grad_X.contiguous(),
            Image_grad_Y.contiguous(),
            kernels.contiguous(),
            projection_map.contiguous(),
            gradIN.contiguous(),
            gradKernels.contiguous(),
            input.size(0),
            input.size(1),
            input.size(2),
            input.size(3),
            projection_map.size(0),
            projection_map.size(1),
            ALPHA,
        )
        grad_input = gradIN[:, :, padwh:-padwh, padwh:-padwh]
        # print("gradKernels = ", gradKernels.mean())
        
        # print("mean = ", abs(torch.round(LR * gradKernels)).mean())
        # print(kernels)
        
        kernels -= torch.round(LR * gradKernels).long()
        torch.clip_(kernels, 0, h_filter-1)
        
        # print(kernels)
        return (
            grad_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LBP(nn.Module):
    def __init__(self, Xshape, Wshape, number_of_points=4, lr=10, alpha=0.1):
        super(LBP, self).__init__()

        self.LR = lr
        self.ALPHA = alpha
        if len(Xshape) != 4:
            raise Exception("Invalid Xshape dimension!")
        if len(Wshape) != 4:
            raise Exception("Invalid Wshape dimension!")
        self.Wshape = Wshape
        self.Xshape = Xshape
        self.number_of_points = number_of_points
        self.n_filters, self.d_filter, self.h_filter, self.w_filter = self.Wshape
        self.n_x, self.d_x, self.h_x, self.w_x = self.Xshape

        if self.d_filter != self.d_x:
            raise Exception("Invalid channel size")
        if self.h_filter != self.w_filter:
            raise Exception("Invalid Wshape: H!= W ")

        self.padw = self.w_filter // 2

        self.kernels = nn.Parameter(
            torch.randint(
                0,
                self.h_filter,
                (self.n_filters, self.number_of_points, 2),
                dtype=torch.int32,
            ),
            requires_grad=False,
        )  # [0==>W, 1 ==> H]

        self.projection_map = nn.Parameter(
            torch.randint(
                0,
                self.d_filter,
                (self.n_filters, self.number_of_points),
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        self.Output0 = nn.Parameter(torch.zeros(
            (self.n_x, self.n_filters, self.h_x, self.w_x, self.number_of_points),
            dtype=torch.float32),
            requires_grad=False
        )

        self.Output = nn.Parameter(torch.zeros(
            (self.n_x, self.n_filters, self.h_x, self.w_x),
            dtype=torch.float32),
            requires_grad=False
        )

        self.gradIN = nn.Parameter(torch.zeros(
            (self.n_x, self.d_x, self.h_x + 2 * self.padw, self.w_x + 2 * self.padw),
            dtype=torch.float32),requires_grad=True)
        
        self.gradKernels = nn.Parameter(torch.zeros(
            self.kernels.shape, dtype=torch.float32), requires_grad=False
        )

        self.Image_grad_X = nn.Parameter(torch.zeros(
            (self.n_x, self.d_x, self.h_x + 2 * self.padw, self.w_x + 2 * self.padw),
            dtype=torch.float32),
            requires_grad=False
        )
        self.Image_grad_Y = nn.Parameter(torch.zeros(
            (self.n_x, self.d_x, self.h_x + 2 * self.padw, self.w_x + 2 * self.padw),
            dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, input):
        return LBPfucntion.apply(
            input,
            self.kernels,
            self.projection_map,
            self.Output0,
            self.Output,
            self.gradIN,
            self.gradKernels,
            self.Image_grad_X,
            self.Image_grad_Y,
            self.padw,
            self.ALPHA,
            self.LR,
            self.h_filter,
        )

    def extra_repr(self):
        return 'input_size={}, output_size={}, filter_features={}/{} points'.format(
            self.Xshape, (self.n_x, self.n_filters, self.h_x, self.w_x),(self.h_filter, self.w_filter) , self.number_of_points
        )

from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor

train_data = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=ToTensor())
validation_data = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=ToTensor())


batch_size = 100

train_dataloader = DataLoader(train_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self,batch=1):
        super(NeuralNetwork, self).__init__()
        self.lbp_1 = LBP(
            (batch, 1, 28, 28), (10, 1, 3, 3), number_of_points=2, lr=1000, alpha=1
        )
        # self.lbp_2 = LBP(
        #     (batch, 10, 28, 28), (20, 10, 3, 3), number_of_points=4, lr=100, alpha=1
        # )

        self.conv1 = nn.Conv2d(1,10,(3,3),padding=(1,1)) 
        self.flatten = nn.Flatten()
        self.Lin1 = nn.Linear(10*28*28, 10)
        
    def forward(self, x):
        # x1 = self.lbp_1(x) 
        # x2 = self.lbp_2(x1) 

        x1 = self.conv1(x)
        
        x3 = self.flatten(x1)
        out = self.Lin1(x3)
        return out


model = NeuralNetwork(batch=batch_size).cuda()

print(">>>>>>>>>>>>>>>>>  STARTED  <<<<<<<<<<<<<<<<<<<")
print(model)

learning_rate = 1e-3

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



for ep in range(50):
    for batch, (X, y) in enumerate(train_dataloader):

        pred = model(X.cuda())
        loss = loss_fn(pred, y.cuda())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # break;
    
    # if ep > 3:
    #     model.Lin1.weight.required_grad=False
    #     model.Lin1.bias.required_grad=False
    # else:
    #     model.Lin1.weight.required_grad=True
    #     model.Lin1.bias.required_grad=True
    correct = 0
    with torch.no_grad():
        for X, y in validation_dataloader:
            pred = model(X.cuda())
            correct += (pred.argmax(1) == y.cuda()).type(torch.float).sum().item()
            
    # print(model.lbp_1.kernels.data)
    print(f"[{ep}]\t loss = {loss.item():.6f} \ttest Accuracy = {correct / 100} %")
    
    
print(">>>>>>>>>>>>>>>>> FINISHED <<<<<<<<<<<<<<<<<<<<")
