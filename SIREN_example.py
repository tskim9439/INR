#%%
import os
from typing import OrderedDict
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time

# Image Data related functions
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
# %%
# Define Model
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
    def forward_with_intermediate(self, x):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        net = []
        net += [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for i in range(hidden_layers):
            net += [SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0)]
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            net += [final_linear]
        else:
            net += [SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)]
        
        self.net = nn.Sequential(*net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivatives
        output = self.net(coords)
        return output, coords
    
    def forward_with_activations(self, coords, retain_grad=False):
        """
        Returns not only model output, but also intermediate activations,
        Used for visualizing
        """
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations['_'.join(str(layer.__class__), "%d" % activation_count)] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations['_'.join(str(layer.__class__), "%d" % activation_count)]
            activation_count += 1
        return activations

# Train related functions
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def divergence(grad, x):
    div = 0.
    for i in range(grad.shape[-1]):
        div += torch.autograd.grad(grad[..., i], x, torch.ones_like(grad[...,i]), create_graph=True)[0][..., i:i+1]
    return div

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

#%%
sidelength = 256
cameramen_dataset = ImageFitting(sidelength)
dataloader = DataLoader(cameramen_dataset,
                        batch_size=1,
                        pin_memory=True,
                        num_workers=0)
model = Siren(in_features=2, hidden_features=256, hidden_layers=3, out_features=1, outermost_linear=True)
model.cuda()

total_steps = 500
steps_til_summary = 100
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

for step in range(total_steps):
    model_output, coords = model(model_input)
    loss = ((model_output - ground_truth)**2).mean()
    
    if not step % steps_til_summary:
        print(f'Step [{step:3d}/{total_steps:3d}], Loss {loss.item():.4f}')
        img_grad = gradient(model_output, coords)
        img_laplacian = laplace(model_output, coords)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(model_output.cpu().view(sidelength, sidelength).detach().numpy())
        axes[1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength, sidelength).detach().numpy())
        axes[2].imshow(img_laplacian.cpu().view(sidelength, sidelength).detach().numpy())
        plt.show()
    optim.zero_grad()
    loss.backward()
    optim.step()
# %%
model.eval()
SR_sidelength = 512
SR_coords = get_mgrid(SR_sidelength)
SR_model_output, _ = model(SR_coords.cuda())

SR_sidelength2 = 1024
SR_coords2 = get_mgrid(SR_sidelength2)
SR_model_output2, _ = model(SR_coords2.cuda())

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(ground_truth.cpu().view(sidelength, sidelength).detach().numpy())
axes[1].imshow(SR_model_output.cpu().view(SR_sidelength, SR_sidelength).detach().numpy())
axes[2].imshow(SR_model_output2.cpu().view(SR_sidelength2, SR_sidelength2).detach().numpy())
plt.show()
# %%
plt.imshow(SR_model_output2.cpu().view(SR_sidelength2, SR_sidelength2).detach().numpy())
plt.savefig('cameramen_1024.png', dpi=1024)
# %%
