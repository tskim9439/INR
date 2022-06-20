#%%
import os
from typing import OrderedDict
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import torchaudio
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time
import math

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


# %%
import torchaudio
import scipy.io.wavfile as wavfile
import io
from IPython.display import Audio

def one_hot_encoder(x, encode_rate=256):
    print(x)
    return torch.eye(encode_rate)[x]

def decode(x, decoder):
    x = F.softmax(x, dim=-1)
    x = torch.argmax(x, dim=-1)
    x = decoder(x.long())
    return x

class AudioFile(torch.utils.data.Dataset):
    def __init__(self, filename, encoder):
        self.rate, self.data = wavfile.read(filename)
        print(self.rate)
        self.data = torchaudio.transforms.Resample(self.rate, 16000)(torch.Tensor(self.data).float())
        self.rate = 16000
        print(self.rate)
        self.data = self.data#.astype(np.float32)[:len(self.data)//4]
        self.timepoints = get_mgrid(len(self.data), 1)
        self.encoder = encoder

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        amplitude = self.data #torch.Tensor(self.data).float()
        scale = torch.max(torch.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = self.encoder(torch.Tensor(amplitude))
        amplitude = one_hot_encoder(amplitude, 256)
        amplitude = torch.Tensor(amplitude).view(-1, 256)
        return self.timepoints, amplitude

mulaw_encoder = torchaudio.transforms.MuLawEncoding(256)
mulaw_decoder = torchaudio.transforms.MuLawDecoding(256)

audio = AudioFile('/home/gtts/INR/LJ001-0001.wav', mulaw_encoder) #'/home/gtts/INR/LJ001-0001.wav', encoder=mulaw_encoder)
dataloader = DataLoader(audio,
                        batch_size=1,
                        pin_memory=True,
                        num_workers=0)

# %%
import copy
import torchinfo
total_steps = 5000
steps_til_summary = 1000
model = Siren(in_features=1,
              hidden_features=512,
              hidden_layers=3,
              out_features=256,
              outermost_linear=True,
              first_omega_0=3000)
model.cuda()
print(torchinfo.summary(model, input_size=(1,1)))
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
model_input, ground_truth = next(iter(dataloader))
model_input = model_input.view(-1, 1)
ground_truth = ground_truth.view(-1, 256)
print(ground_truth.shape)
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
criterion = nn.CrossEntropyLoss()
best_loss = 99999.
for step in range(total_steps):
    model_output, coords = model(model_input)
    
    loss = criterion(model_output, ground_truth) #((model_output - ground_truth)**2).mean()
    
    if not step % steps_til_summary:
        print(f'Step [{step:3d}/{total_steps:3d}], Loss {loss.item():.6f}')
        #img_grad = gradient(model_output, coords)
        #img_laplacian = laplace(model_output, coords)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].plot(decode(model_output.detach().cpu(), mulaw_decoder))
        axes[1].plot(decode(ground_truth.detach().cpu(), mulaw_decoder))
        axes[2].plot(decode(ground_truth.detach().cpu(), mulaw_decoder) - decode(model_output.detach().cpu(), mulaw_decoder))
        plt.show()
    optim.zero_grad()
    loss.backward()
    optim.step()
    if loss.item() < best_loss:
        print(f'{best_loss} -> {loss.item()}')
        best_model = copy.deepcopy(model)
        best_loss = loss.item()
# %%
with torch.no_grad():
    model_output, coords = best_model(model_input)
    loss = criterion(model_output, ground_truth)
print(f"Final Loss : {loss.item()}")
# %%
# 2.8270906113903038e-05
# 4.532434104476124e-05 -> * 50
# 1.3868239875591826e-05 -> * 25
#%%
rate, data = wavfile.read('/home/gtts/INR/LJ001-0001.wav') #'/home/gtts/INR/LJ001-0001.wav')
data = torchaudio.transforms.Resample(rate, 16000)(torch.Tensor(data).float())
scale = torch.max(torch.abs(data))
data = data / scale #.astype(np.float32)[:len(data)//4]
plt.plot(data)
plt.show()
output = decode(model_output.detach().cpu(), mulaw_decoder)
#%%
plt.plot(output)
plt.show()
plt.plot(data - output)
plt.show()
# %%
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from metric import PESQ, PSNR
pesq = PESQ(16000, 16000)
psnr = PSNR()
#pesq_wide = PerceptualEvaluationSpeechQuality(16000, 'wb')
#pesq_narrow = PerceptualEvaluationSpeechQuality(16000, 'nb')

#downsample_output = torchaudio.transforms.Resample(44100, 16000)(output)
#downsample_gt = torchaudio.transforms.Resample(44100, 16000)(torch.Tensor(data).float())
# 16000 : (tensor(2.7185), tensor(3.3257))
# 8000 : tensor(3.6670)
#pesq_wide(output, data), pesq_narrow(output, data)
pesq(output, data), psnr(output, data) # (3.9287679195404053, tensor(55.8476))

# %%
