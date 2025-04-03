import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

def relu(x):
    return torch.relu(x)

def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)
def zero_padding(x, pad_top , pad_bottom , pad_left, pad_right):
    batch_size , channels , height , width = x.shape
    new_height = height + pad_top + pad_bottom #for top and bottom
    new_width = width + pad_left + pad_right #for left and right
    padded_x = torch.zeros((batch_size , channels , new_height , new_width) , dtype=x.dtype)

    padded_x[: , : , pad_top:pad_top+height , pad_left:pad_left+width] = x
    return padded_x


def convolve2d(x , kernel , stride = 1, padding =0):
    x_padded = zero_padding(x , padding ,padding , padding , padding)
    batch_size , in_channels , in_height , in_width = x_padded.shape
    out_channels , _ , kernel_height , kernel_width = kernel.shape
    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1
    out = torch.zeros((batch_size , out_channels , out_height , out_width))
    for i in range(out_height):
        for j in range(out_width):
            x_slice = x_padded[: , : , i*stride:i*stride+kernel_height , j*stride : j*stride+kernel_width]
            for k in range(out_channels):
                out[: , k , i , j] = torch.sum(x_slice * kernel[k , : , : , : ] , dim=(1,2,3))
    return out

def max_pool2d(x , size = 2 , stride = 2):
    batch_size , channels , height , width = x.shape
    out_height = (height - size) // stride + 1
    out_width = (width - size) // stride  + 1
    out = torch.zeros((batch_size , channels , out_height , out_width)) 
    for i in range(out_height):
        for j in range(out_width):
            x_slice = x[: , : , i*stride:i*stride+size , j*stride:j*stride+size]
            out[: , : , i , j] = torch.amax(x_slice , dim=(2,3))
    return out

class LeNet:
    def __init__(self) -> None:
        self.conv1_weight = torch.randn(6,1,5,5 , requires_grad=True)
        self.conv2_weight = torch.randn(16,6,5,5, requires_grad=True)
        self.fc1_weight = torch.randn(120 , 16*5*5, requires_grad=True)
        self.fc2_weight = torch.randn(84 , 120, requires_grad=True)
        self.fc3_weight = torch.randn(10 , 84, requires_grad=True)
    def forward(self , x):
        x = convolve2d(x , self.conv1_weight , padding=2)
        x = relu(x)
        x = max_pool2d(x)
        x = convolve2d(x , self.conv2_weight)
        x = relu(x)
        x = max_pool2d(x)
        x = x.reshape(x.shape[0] , -1)
        x = relu(torch.matmul(x , self.fc1_weight.T))
        x = relu(torch.matmul(x , self.fc2_weight.T))
        x = softmax(torch.matmul(x , self.fc3_weight.T))
        return x

    def parameters(self):
        return [self.conv1_weight , self.conv2_weight , self.fc1_weight , self.fc2_weight , self.fc3_weight]

# class FashionMNIST:
#     def __init__(self,batch_size = 64 , resize=(28,28)) -> None:
#         trans = torchvision.transforms.Compose([transforms.Resize(resize) ,
#                                      transforms.ToTensor()])
#         self.batch_size = batch_size
#         self.train = torchvision.datasets.FashionMNIST(
#             root = 'root' , train = True , transform = trans , download = True
#         )
#         self.val = torchvision.datasets.FashionMNIST(
#             root = 'root' , train = False , download = True , transform = trans
#         )
    
#     def text_labels(self , indices):
#         labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#         return [labels[int(i)] for i in indices]
    
#     def get_dataloader(self , train):
#         data = self.train if train else self.val
#         return torch.utils.data.DataLoader(data , self.batch_size , shuffle = train )

# batch_size = 128
# resize = (28,28)
# data = FashionMNIST(batch_size , resize)
# train_loader = data.get_dataloader(train=True)
# test_loader = data.get_dataloader(train=False)


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize MNIST dataset
    ])
    
    train_data = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

train_loader , test_loader = get_mnist_loaders(batch_size=8)


model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

c = 0
for X , y in train_loader:
    # print("X shape" , X.shape)
    # print("y shape " , y.shape)
    c += 1
print("C " , c)

num_epochs = 10
for epoch in range(num_epochs):
    for X , y in train_loader:
        optimizer.zero_grad()
        output = model.forward(X)
        # print(output)
        # print('------------')
        # print(y)
        loss = criterion(output , y)
        loss.backward()
        optimizer.step()
        print("Loss" , loss)
    if epoch%2: 
        print(f"Loss : {loss} after num of {epoch + 1} ")
# for X , y in train_loader:
#     optimizer.zero_grad()
#     output = model.forward(X)
#         # print(output)
#         # print('------------')
#         # print(y)
#     loss = criterion(output , y)
#     loss.backward()
#     optimizer.step()
#     print("Loss" , loss)