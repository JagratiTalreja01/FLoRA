# /home/jagrati/Desktop/Jagrati/Flora_package/losses/hydro.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# ----- basic -----
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps))

def ssim3x3(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    s_x  = F.avg_pool2d(x*x, 3,1,1) - mu_x*mu_x
    s_y  = F.avg_pool2d(y*y, 3,1,1) - mu_y*mu_y
    s_xy = F.avg_pool2d(x*y, 3,1,1) - mu_x*mu_y
    n = (2*mu_x*mu_y + C1) * (2*s_xy + C2)
    d = (mu_x*mu_x + mu_y*mu_y + C1) * (s_x + s_y + C2)
    ssim_map = n / (d + 1e-6)
    return torch.clamp(ssim_map, 0.0, 1.0).mean()

def sobel_mag(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)

def edge_loss(pred_rgb, gt_rgb):
    y_pred = pred_rgb.mean(1, keepdim=True)
    y_gt   = gt_rgb.mean(1, keepdim=True)
    return F.l1_loss(sobel_mag(y_pred), sobel_mag(y_gt))

def fft_mag_loss(pred_rgb, gt_rgb, eps=1e-6):
    """
    Compare log-magnitude spectra in FP32 (FFT doesn't support fp16).
    Safe under AMP.
    """
    def fft_mag(x):
        with autocast(enabled=False):  # force fp32 ops here
            x32 = x.mean(1, keepdim=True).to(torch.float32)
            f = torch.fft.rfft2(x32, norm='ortho')
            mag = torch.log(torch.abs(f) + torch.tensor(eps, dtype=torch.float32, device=x32.device))
            return mag
    return F.l1_loss(fft_mag(pred_rgb), fft_mag(gt_rgb))

# ----- segmentation -----
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = logits.clamp(1e-6, 1-1e-6)
        targets = targets.float()
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum() + self.eps
        dice = 2 * inter / denom
        return 1 - dice

def bce_loss(logit, target):
    """
    Safe under AMP when mask head outputs probabilities (after sigmoid).
    Forces BCE to run in fp32.
    """
    with autocast(enabled=False):
        x = logit.to(torch.float32).clamp(1e-6, 1 - 1e-6)
        t = target.to(torch.float32)
        return F.binary_cross_entropy(x, t)

# ----- hydrology-aware edge alignment -----
def hydro_edge_loss(pred_rgb, sar, mask_gt, band=3):
    with torch.no_grad():
        pooled = F.max_pool2d(mask_gt, kernel_size=band, stride=1, padding=band//2)
        boundary = (pooled - mask_gt).abs() > 0.5  # boolean band
    y_gray = pred_rgb.mean(1, keepdim=True)
    g_y = sobel_mag(y_gray)
    g_s = sobel_mag(sar.mean(1, keepdim=True))
    cos = F.cosine_similarity(g_y, g_s, dim=1, eps=1e-6).unsqueeze(1)
    return (1 - cos)[boundary].mean() if boundary.any() else (1 - cos).mean()

# ----- distillation between pyramids -----
def distill_pyramids(f_sar_list, f_opt_list):
    loss = 0.0
    for fs, fo in zip(f_sar_list, f_opt_list):
        loss = loss + F.l1_loss(fs, fo.detach())
    return loss
