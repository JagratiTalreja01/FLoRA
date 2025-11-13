# train_flora.py
# Supervised FLoRA (teacher prior enabled) + decoupled heads + LPIPS device fix + optional seg warm-up

import os, time, yaml, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

def set_seed(s: int):
    import random
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dynamic_import(module_path: str, cls_or_fn: str):
    import importlib
    m = importlib.import_module(module_path); return getattr(m, cls_or_fn)

def psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse < eps: return torch.tensor(99.0, device=pred.device)
    return 10.0 * torch.log10(1.0 / (mse + eps))

def ssim_metric(x, y, eps=1e-6):
    C1 = 0.01 ** 2; C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1); mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + eps)
    return ssim_map.mean()

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction='mean'): super().__init__(); self.eps=eps; self.reduction=reduction
    def forward(self, x, y):
        loss = torch.sqrt((x-y)**2 + self.eps**2)
        return loss.mean() if self.reduction=='mean' else (loss.sum() if self.reduction=='sum' else loss)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, pred, target):
        B = pred.shape[0]; pred = pred.view(B,-1); target = target.view(B,-1)
        inter = (pred*target).sum(1); denom = pred.sum(1)+target.sum(1)
        dice = (2*inter + self.eps)/(denom + self.eps)
        return 1 - dice.mean()

class LPIPSLoss(nn.Module):
    def __init__(self, net='alex'):
        super().__init__()
        try:
            import lpips
            self.lpips = lpips.LPIPS(net=net); self.enabled = True
        except Exception:
            self.lpips = None; self.enabled = False
            print("[warn] LPIPS not available; continuing without it.")
    def forward(self, x, y):
        if not self.enabled: return torch.zeros((), device=x.device)
        dev = x.device
        if next(self.lpips.parameters(), torch.tensor([], device=dev)).device != dev:
            self.lpips = self.lpips.to(dev)
        self.lpips.eval()
        x2 = x*2-1; y2 = y*2-1
        return self.lpips(x2, y2).mean()

def try_import_hydro_losses():
    EdgeLoss = FFTMagLoss = HydroBoundaryLoss = None
    try:
        from hydro import EdgeLoss as _E, FFTMagLoss as _F, HydroBoundaryLoss as _H
        EdgeLoss, FFTMagLoss, HydroBoundaryLoss = _E, _F, _H
        print("[info] Loaded Edge/Freq/Hydro losses from hydro.py")
    except Exception:
        print("[warn] hydro.py extras not found; skipping edge/freq/hydro losses.")
    return EdgeLoss, FFTMagLoss, HydroBoundaryLoss

def build_model(cfg):
    from models.flora import FLoRA, SimpleEncoder, SimplePyramid, PriorPyramid, RGBDecoder, MaskDecoder
    base = cfg.get('base_channels', 64); in_ch_sar = cfg.get('in_ch_sar', 2); prior_ch = cfg.get('prior_channels', 1)
    sar_enc = SimpleEncoder(in_ch=in_ch_sar, base=base)
    opt_teacher = SimplePyramid(in_ch=prior_ch, base=base)
    prior_pred  = PriorPyramid(in_ch=in_ch_sar, base=base, widen=2)
    dec_rgb = RGBDecoder(base=base, out_ch=3); dec_mask = MaskDecoder(base=base, out_ch=1)
    # >>> change here: removed unsupported decouple kwarg
    return FLoRA(
        sar_enc, opt_teacher, dec_rgb, dec_mask,
        prior_pred=prior_pred,
        base=base, prior_channels=prior_ch
    )

def build_dataloaders(cfg):
    DatasetClass = dynamic_import(cfg['dataset']['module'], cfg['dataset']['class'])
    tr_kwargs = cfg['dataset']['train_kwargs']; va_kwargs = cfg['dataset']['val_kwargs']
    train_ds = DatasetClass(split='train', **tr_kwargs)
    val_ds   = DatasetClass(split='val',   **va_kwargs)
    bs = cfg['train'].get('batch_size', 2); nw = cfg['train'].get('num_workers', 4)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl

def compute_losses(pred_rgb, pred_msk, gt_rgb, gt_msk, loss_objs, w):
    L = {}
    L['rec']   = loss_objs['charb'](pred_rgb, gt_rgb) * w.get('rec',1.0)
    ssim_v     = ssim_metric(pred_rgb, gt_rgb); L['ssim']  = (1.0-ssim_v) * w.get('ssim',0.0)
    L['lpips'] = loss_objs['lpips'](pred_rgb, gt_rgb) * w.get('lpips',0.0)
    if loss_objs.get('edge') is not None: L['edge'] = loss_objs['edge'](pred_rgb, gt_rgb) * w.get('edge', 0.0)
    if loss_objs.get('freq') is not None: L['freq'] = loss_objs['freq'](pred_rgb, gt_rgb) * w.get('freq', 0.0)
    if loss_objs.get('hyd')  is not None: L['hyd']  = loss_objs['hyd'](pred_rgb, gt_rgb, pred_msk) * w.get('hyd', 0.0)

    has_pos = (gt_msk.sum(dim=[1,2,3]) > 0).float().view(-1,1,1,1)
    seg_enable = (has_pos.mean() > 0).item()
    if seg_enable:
        L['dice'] = loss_objs['dice'](pred_msk, gt_msk) * w.get('seg',1.0)
        L['bce']  = F.binary_cross_entropy(pred_msk.clamp(1e-6,1-1e-6), gt_msk) * w.get('seg_bce',0.0)
    else:
        dev = pred_msk.device; L['dice']=torch.zeros((),device=dev); L['bce']=torch.zeros((),device=dev)

    total = sum(L.values()); return total, L, seg_enable

def train_one_epoch(model, optimizer, train_dl, device, loss_objs, wcfg, grad_clip=None, epoch=1, freeze_seg_until=0):
    model.train()
    if epoch==1 and freeze_seg_until>0:
        for p in model.dec_mask.parameters(): p.requires_grad=False
    if epoch==freeze_seg_until+1:
        for p in model.dec_mask.parameters(): p.requires_grad=True

    t0 = time.time(); loss_avg=0.0; count=0; seg_batches=0
    for batch_idx, batch in enumerate(train_dl):
        sar  = batch['sar'].to(device, non_blocking=True)
        opt  = batch['opt'].to(device, non_blocking=True)
        prior = batch.get('prior', None)
        prior = prior.to(device, non_blocking=True) if prior is not None else None
        gt_msk = batch.get('mask', None)
        gt_msk = (gt_msk.to(device, non_blocking=True) if gt_msk is not None
                  else torch.zeros((sar.size(0),1,sar.size(2),sar.size(3)), device=device))

        # TEACHER ON during training
        pred_rgb, pred_msk, _, _ = model(sar, prior=prior, use_teacher=True)

        loss, Ldict, seg_enable = compute_losses(pred_rgb, pred_msk, opt, gt_msk, loss_objs, wcfg)

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss at batch {batch_idx}")
        loss.backward()
        if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_avg += float(loss.detach().cpu()); count += 1
        if seg_enable: seg_batches += 1

    dt = time.time()-t0
    return loss_avg/max(count,1), seg_batches, dt

@torch.no_grad()
def validate(model, val_dl, device):
    model.eval(); psnr_sum=0.0; ssim_sum=0.0; n=0
    for batch in val_dl:
        sar = batch['sar'].to(device, non_blocking=True)
        opt = batch['opt'].to(device, non_blocking=True)
        # Inference without teacher (deploy setting)
        pred_rgb, pred_msk, _, _ = model(sar, prior=None, use_teacher=False)
        psnr_sum += float(psnr(pred_rgb, opt).cpu())
        ssim_sum += float(ssim_metric(pred_rgb, opt).cpu()); n += 1
    return psnr_sum/max(n,1), ssim_sum/max(n,1)

def save_ckpt(path, model, optimizer, epoch, best_val, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                'epoch': epoch, 'best': best_val, 'cfg': cfg}, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args(); cfg = yaml.safe_load(open(args.config,'r'))

    set_seed(cfg.get('seed',42))
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'): torch.cuda.set_device(args.gpu)

    train_dl, val_dl = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    EdgeLoss, FFTMagLoss, HydroBoundaryLoss = try_import_hydro_losses()
    loss_objs = {
        'charb': CharbonnierLoss(1e-3), 'lpips': LPIPSLoss('alex'),
        'dice': DiceLoss(),
        'edge': EdgeLoss() if EdgeLoss else None,
        'freq': FFTMagLoss() if FFTMagLoss else None,
        'hyd' : HydroBoundaryLoss() if HydroBoundaryLoss else None,
    }
    wcfg = cfg.get('loss_weights', {'rec':0.8,'ssim':0.15,'lpips':0.2,'seg':1.0,'seg_bce':0.2,'edge':0.2,'freq':0.1,'hyd':0.05})

    lr = cfg['train'].get('lr',1e-4); wd = cfg['train'].get('weight_decay',1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    out_dir = Path(cfg['train'].get('out_dir','./checkpoints_flora')); out_dir.mkdir(parents=True, exist_ok=True)
    best_psnr = -1.0; epochs = cfg['train'].get('epochs',50); grad_clip = cfg.get('grad_clip',1.0)
    freeze_seg_until = cfg['train'].get('freeze_seg_warmup_epochs',0)

    print(f"✅ Loaded config from {args.config}")
    print(f"→ device: {device} | epochs: {epochs} | batch_size: {cfg['train'].get('batch_size',2)}")

    for ep in range(1, epochs+1):
        tr_loss, seg_batches, dt = train_one_epoch(model, optimizer, train_dl, device, loss_objs, wcfg,
                                                   grad_clip=grad_clip, epoch=ep, freeze_seg_until=freeze_seg_until)
        val_psnr, val_ssim = validate(model, val_dl, device)
        print(f"[{ep:03d}/{epochs}] TrainLoss: {tr_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | Seg batches: {seg_batches} | {dt:.1f}s")

        save_ckpt(str(out_dir/'flora_last.pt'), model, optimizer, ep, best_psnr, cfg)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_ckpt(str(out_dir/'flora_best.pt'), model, optimizer, ep, best_psnr, cfg)
            print(f"💾 New best (PSNR {best_psnr:.2f}) → flora_best.pt")

if __name__ == "__main__":
    main()
