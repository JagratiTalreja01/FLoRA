# tools/vis_flora.py
# Quick n samples from VAL set; same as test but only images.

import os, yaml, argparse, numpy as np, matplotlib.pyplot as plt, torch
from torch.utils.data import DataLoader
from train_flora import build_model, dynamic_import, set_seed

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--n', type=int, default=8)
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'): torch.cuda.set_device(args.gpu)

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get('seed',42))
    DatasetClass = dynamic_import(cfg['dataset']['module'], cfg['dataset']['class'])
    ds = DatasetClass(split='val', **cfg['dataset']['val_kwargs'])
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()

    os.makedirs('viz', exist_ok=True)
    for i, batch in enumerate(dl):
        if i >= args.n: break
        sar = batch['sar'].to(device); opt = batch['opt'].to(device)
        pr_rgb, pr_msk, _, _ = model(sar, prior=None, use_teacher=False)

        sar_np = sar[0].detach().cpu().numpy(); vv, vh = sar_np[0], sar_np[1]
        sar_img = np.stack([(vv-vv.min())/(vv.max()-vv.min()+1e-8),
                            (vh-vh.min())/(vh.max()-vh.min()+1e-8),
                            0.5*np.ones_like(vv)], axis=-1)
        pr_rgb_np = pr_rgb[0].detach().cpu().numpy().transpose(1,2,0)
        gt_rgb_np = opt[0].detach().cpu().numpy().transpose(1,2,0)
        mask_np   = pr_msk[0,0].detach().cpu().numpy()

        fig = plt.figure(figsize=(12,4), dpi=150)
        ax1=plt.subplot(1,4,1); ax1.imshow(sar_img); ax1.set_title('Input SAR'); ax1.axis('off')
        ax2=plt.subplot(1,4,2); ax2.imshow(pr_rgb_np); ax2.set_title('Pred RGB'); ax2.axis('off')
        ax3=plt.subplot(1,4,3); ax3.imshow(mask_np, vmin=0, vmax=1, cmap='gray'); ax3.set_title('Pred Mask'); ax3.axis('off')
        ax4=plt.subplot(1,4,4); ax4.imshow(gt_rgb_np); ax4.set_title('GT RGB'); ax4.axis('off')
        plt.tight_layout(); plt.savefig(f'viz/val_{i:03d}.png'); plt.close(fig)

if __name__ == "__main__":
    main()
