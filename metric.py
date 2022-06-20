import torch
import torch.nn.functional as F
from asteroid.metrics import get_metrics
from pytorch_msssim import ms_ssim, ssim
from torchaudio.functional import resample


class PSNR:
    def __call__(self, inputs, targets):
        return -10 * torch.log10(F.mse_loss(inputs.detach(), targets.detach()))


class PESQ:
    def __init__(self, input_s_rate, target_s_rate=16000):
        self.input_s_rate = input_s_rate # sample rate
        self.target_s_rate = target_s_rate

    def __call__(self, inputs, targets):
        inputs = resample(inputs, self.input_s_rate, self.target_s_rate)
        inputs = inputs.reshape(-1).cpu().numpy()
        targets = resample(targets, self.input_s_rate, self.target_s_rate)
        targets = targets.reshape(-1).cpu().numpy()

        return get_metrics(targets, targets, inputs,
                           sample_rate=self.target_s_rate,
                           metrics_list=['pesq'])['pesq']