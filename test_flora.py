# test_flora.py — evaluate & visualize with honest SAR rendering

import os, yaml, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from train_flora import (build_model, dynamic_import, set_seed, psnr, ssim_metric, LPIPSLoss)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_dir', default='preview_flora')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        torch.cuda.set_device(args.gpu)

    cfg = yaml.safe_load(open(args.config, 'r'))
    set_seed(cfg.get('seed', 42))

    DatasetClass = dynamic_import(cfg['dataset']['module'], cfg['dataset']['class'])
    val_ds = DatasetClass(split='val', **cfg['dataset']['val_kwargs'])
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=True)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    lpips = LPIPSLoss(net='alex').to(device)
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for i, batch in enumerate(val_dl):
        sar = batch['sar'].to(device)
        opt = batch['opt'].to(device)
        # We evaluate like deploy: no teacher at inference
        # (Even if a 'prior' exists in the dataset, we set prior=None here for fairness)
        pred_rgb, pred_msk, _, _ = model(sar, prior=None, use_teacher=False)

        # Metrics
        psnr_vals.append(psnr(pred_rgb, opt).item())
        ssim_vals.append(ssim_metric(pred_rgb, opt).item())
        lpips_vals.append(lpips(pred_rgb, opt).item())

        # Honest SAR visualization: R=VV, G=VH, B=0.5*(VV+VH)
        sar_np = sar[0].detach().cpu().numpy()
        sar_vv, sar_vh = sar_np[0], sar_np[1]
        mix = 0.5 * (sar_vv + sar_vh)
        vv01 = (sar_vv - sar_vv.min()) / (sar_vv.max() - sar_vv.min() + 1e-8)
        vh01 = (sar_vh - sar_vh.min()) / (sar_vh.max() - sar_vh.min() + 1e-8)
        mx01 = (mix    - mix.min())    / (mix.max()    - mix.min()    + 1e-8)
        sar_img = np.stack([vv01, vh01, mx01], axis=-1)

        pred_rgb_np  = pred_rgb[0].detach().cpu().numpy().transpose(1, 2, 0)
        gt_rgb_np    = opt[0].detach().cpu().numpy().transpose(1, 2, 0)
        pred_mask_np = pred_msk[0,0].detach().cpu().numpy()

        fig = plt.figure(figsize=(12, 4), dpi=150)
        ax1 = plt.subplot(1,4,1); ax1.imshow(sar_img); ax1.set_title("Input SAR"); ax1.axis('off')
        ax2 = plt.subplot(1,4,2); ax2.imshow(np.clip(pred_rgb_np,0,1)); ax2.set_title("Pred RGB"); ax2.axis('off')
        ax3 = plt.subplot(1,4,3); ax3.imshow(np.clip(pred_mask_np,0,1), cmap='gray', vmin=0, vmax=1); ax3.set_title("Pred Mask"); ax3.axis('off')
        ax4 = plt.subplot(1,4,4); ax4.imshow(np.clip(gt_rgb_np,0,1)); ax4.set_title("GT RGB"); ax4.axis('off')
        plt.tight_layout(); fig.savefig(os.path.join(args.out_dir, f"sample_{i:03d}.png")); plt.close(fig)

    print(f"✅ Saved panels in {args.out_dir}")
    print(f"📊 Mean PSNR: {np.mean(psnr_vals):.2f} | Mean SSIM: {np.mean(ssim_vals):.4f} | Mean LPIPS: {np.mean(lpips_vals):.4f}")

if __name__ == "__main__":
    main()
