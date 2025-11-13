# models/flora.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_act(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
        nn.SiLU(inplace=True)
    )

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(cin, cout), conv_bn_act(cout, cout))
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.body(x)
        return self.pool(x), x

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(conv_bn_act(in_ch + skip_ch, out_ch), conv_bn_act(out_ch, out_ch))
    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, (dx//2, dx - dx//2, dy//2, dy - dy//2))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class FiLM(nn.Module):
    def __init__(self, c_feat, c_cond):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Conv2d(c_cond, c_feat, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c_feat, 2*c_feat, 3, 1, 1)
        )
    def forward(self, x, cond):
        gamma, beta = self.affine(cond).chunk(2, dim=1)
        return x * (1 + torch.tanh(gamma)) + beta

class WindowedCrossAttn2D(nn.Module):
    def __init__(self, c, nhead=4, window=8):
        super().__init__()
        self.q = nn.Conv2d(c, c, 1)
        self.k = nn.Conv2d(c, c, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.attn = nn.MultiheadAttention(c, nhead, batch_first=True)
        self.proj = nn.Conv2d(c, c, 1)
        self.window = window
    def forward(self, q2d, kv2d):
        B,C,H,W = q2d.shape
        w = self.window
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        if pad_h or pad_w:
            q2d = F.pad(q2d, (0, pad_w, 0, pad_h))
            kv2d = F.pad(kv2d, (0, pad_w, 0, pad_h))
            H2, W2 = H + pad_h, W + pad_w
        else:
            H2, W2 = H, W
        q = self.q(q2d); k = self.k(kv2d); v = self.v(kv2d)
        q_unf = F.unfold(q, kernel_size=w, stride=w)
        k_unf = F.unfold(k, kernel_size=w, stride=w)
        v_unf = F.unfold(v, kernel_size=w, stride=w)
        L = q_unf.shape[-1]
        def to_seq(t): return t.view(B, C, w*w, L).permute(0,3,2,1).contiguous().view(B*L, w*w, C)
        q_seq, k_seq, v_seq = to_seq(q_unf), to_seq(k_unf), to_seq(v_unf)
        out_seq, _ = self.attn(q_seq.to(torch.float32), k_seq.to(torch.float32), v_seq.to(torch.float32))
        out = out_seq.view(B, L, w*w, C).permute(0,3,2,1).contiguous().view(B, C*w*w, L)
        out = F.fold(out, output_size=(H2, W2), kernel_size=w, stride=w)
        out = self.proj(out)
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out + q2d[:, :, :H, :W]

class FusionBlock(nn.Module):
    def __init__(self, c_feat, c_prior):
        super().__init__()
        self.xattn = WindowedCrossAttn2D(c_feat, nhead=4, window=8)
        self.film  = FiLM(c_feat, c_prior)
        self.gate  = nn.Sequential(nn.Conv2d(c_feat, c_feat, 1), nn.Sigmoid())
    def forward(self, fsar, fprior):
        z = self.xattn(fsar, fprior)
        z = self.film(z, fprior)
        g = self.gate(fsar)
        return fsar + g * (z - fsar)

class SimpleEncoder(nn.Module):
    def __init__(self, in_ch=2, base=64):
        super().__init__()
        self.stem = conv_bn_act(in_ch, base)
        self.d1 = Down(base, base)
        self.d2 = Down(base, base*2)
        self.d3 = Down(base*2, base*4)
        self.mid = nn.Sequential(conv_bn_act(base*4, base*8), conv_bn_act(base*8, base*8))
    def forward(self, x):
        x0 = self.stem(x)
        x1, s1 = self.d1(x0)
        x2, s2 = self.d2(x1)
        x3, s3 = self.d3(x2)
        xm = self.mid(x3)
        return [s1, s2, s3, xm]

class SimplePyramid(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        self.stem = conv_bn_act(in_ch, base)
        self.l1   = conv_bn_act(base, base)
        self.down1= nn.MaxPool2d(2)
        self.l2   = conv_bn_act(base, base*2)
        self.down2= nn.MaxPool2d(2)
        self.l3   = conv_bn_act(base*2, base*4)
        self.down3= nn.MaxPool2d(2)
        self.l4   = conv_bn_act(base*4, base*8)
    def forward(self, x):
        p1 = self.l1(self.stem(x))
        p2 = self.l2(self.down1(p1))
        p3 = self.l3(self.down2(p2))
        p4 = self.l4(self.down3(p3))
        return [p1, p2, p3, p4]

class PriorPyramid(nn.Module):
    def __init__(self, in_ch=2, base=64, widen=2):
        super().__init__()
        wb = base * widen
        self.stem = conv_bn_act(in_ch, wb)
        self.l1   = conv_bn_act(wb, wb)
        self.down1= nn.MaxPool2d(2)
        self.l2   = conv_bn_act(wb, wb*2)
        self.down2= nn.MaxPool2d(2)
        self.l3   = conv_bn_act(wb*2, wb*4)
        self.down3= nn.MaxPool2d(2)
        self.l4   = conv_bn_act(wb*4, wb*8)
        self.p1 = nn.Conv2d(wb,   base,   1)
        self.p2 = nn.Conv2d(wb*2, base*2, 1)
        self.p3 = nn.Conv2d(wb*4, base*4, 1)
        self.p4 = nn.Conv2d(wb*8, base*8, 1)
    def forward(self, x):
        t1 = self.l1(self.stem(x))
        t2 = self.l2(self.down1(t1))
        t3 = self.l3(self.down2(t2))
        t4 = self.l4(self.down3(t3))
        return [self.p1(t1), self.p2(t2), self.p3(t3), self.p4(t4)]

class RGBDecoder(nn.Module):
    def __init__(self, base=64, out_ch=3):
        super().__init__()
        self.up3 = Up(in_ch=base*8, skip_ch=base*4, out_ch=base*4)
        self.up2 = Up(in_ch=base*4, skip_ch=base*2, out_ch=base*2)
        self.up1 = Up(in_ch=base*2, skip_ch=base,   out_ch=base)
        self.head= nn.Sequential(
            nn.Conv2d(base + base, base, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, out_ch, 1),
        )
    def forward(self, feats):
        s1, s2, s3, xm = feats
        x = self.up3(xm, s3); x = self.up2(x, s2); x = self.up1(x, s1)
        x = torch.cat([x, s1], dim=1)
        return torch.sigmoid(self.head(x))

class MaskDecoder(nn.Module):
    def __init__(self, base=64, out_ch=1):
        super().__init__()
        self.up3 = Up(in_ch=base*8, skip_ch=base*4, out_ch=base*4)
        self.up2 = Up(in_ch=base*4, skip_ch=base*2, out_ch=base*2)
        self.up1 = Up(in_ch=base*2, skip_ch=base,   out_ch=base)
        self.head= nn.Sequential(
            nn.Conv2d(base + base, base, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, out_ch, 1),
        )
    def forward(self, feats):
        s1, s2, s3, xm = feats
        x = self.up3(xm, s3); x = self.up2(x, s2); x = self.up1(x, s1)
        x = torch.cat([x, s1], dim=1)
        return torch.sigmoid(self.head(x))

class _Adapter(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c, c, 1), nn.SiLU(inplace=True), nn.Conv2d(c, c, 3, 1, 1))
    def forward(self, x): return self.proj(x)

class FLoRA(nn.Module):
    """
    decouple: dict
      - seg_from_rgb: bool (stop seg grads into shared)
      - rgb_from_seg: bool (stop rgb grads into shared)
    """
    def __init__(self, sar_enc=None, opt_teacher=None, dec_rgb=None, dec_mask=None,
                 prior_pred=None, base=64, prior_channels=1, decouple=None):
        super().__init__()
        self.base = base
        self.decouple = decouple or {"seg_from_rgb": True, "rgb_from_seg": False}
        self.sar_enc = sar_enc if sar_enc is not None else SimpleEncoder(in_ch=2, base=base)
        self.opt_teacher = opt_teacher if opt_teacher is not None else SimplePyramid(in_ch=prior_channels, base=base)
        self.prior_pred = prior_pred if prior_pred is not None else PriorPyramid(in_ch=2, base=base, widen=2)
        self.fuse1 = FusionBlock(base,   base)
        self.fuse2 = FusionBlock(base*2, base*2)
        self.fuse3 = FusionBlock(base*4, base*4)
        self.fuse4 = FusionBlock(base*8, base*8)
        self.rgb_a1 = _Adapter(base);   self.rgb_a2 = _Adapter(base*2)
        self.rgb_a3 = _Adapter(base*4); self.rgb_a4 = _Adapter(base*8)
        self.seg_a1 = _Adapter(base);   self.seg_a2 = _Adapter(base*2)
        self.seg_a3 = _Adapter(base*4); self.seg_a4 = _Adapter(base*8)
        self.dec_rgb  = dec_rgb  if dec_rgb  is not None else RGBDecoder(base=base, out_ch=3)
        self.dec_mask = dec_mask if dec_mask is not None else MaskDecoder(base=base, out_ch=1)

    def forward(self, sar, prior=None, use_teacher=True):
        s1, s2, s3, xm = self.sar_enc(sar)
        if use_teacher and (prior is not None):
            p1, p2, p3, p4 = self.opt_teacher(prior)
        else:
            p1, p2, p3, p4 = self.prior_pred(sar)

        f1 = self.fuse1(s1, p1); f2 = self.fuse2(s2, p2)
        f3 = self.fuse3(s3, p3); f4 = self.fuse4(xm, p4)

        f_for_rgb = [f1, f2, f3, f4]
        f_for_seg = [fi.detach() for fi in [f1, f2, f3, f4]] if self.decouple.get("seg_from_rgb", True) else [f1, f2, f3, f4]
        if self.decouple.get("rgb_from_seg", False):
            f_for_rgb = [fi.detach() for fi in f_for_rgb]

        fr1, fr2, fr3, fr4 = self.rgb_a1(f_for_rgb[0]), self.rgb_a2(f_for_rgb[1]), self.rgb_a3(f_for_rgb[2]), self.rgb_a4(f_for_rgb[3])
        fs1, fs2, fs3, fs4 = self.seg_a1(f_for_seg[0]), self.seg_a2(f_for_seg[1]), self.seg_a3(f_for_seg[2]), self.seg_a4(f_for_seg[3])

        y_rgb  = self.dec_rgb([fr1, fr2, fr3, fr4])
        y_mask = self.dec_mask([fs1, fs2, fs3, fs4])
        return y_rgb, y_mask, (f1, f2, f3, f4), (p1, p2, p3, p4)
