import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import split_feature, merge_splits
from .position import PositionEmbeddingSine


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out


def generate_shift_window_attn_mask(input_resolution, window_size_h, window_size_w,
                                    shift_size_h, shift_size_w, device=torch.device('cuda')):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    if torch.compiler.is_compiling():
        # export path (SMV): the same region ids 0..8 from arange
        # comparisons (rows [0, h - wh) / [h - wh, h - sh) / [h - sh, h), columns alike), so a
        # size-free export has no Python slice loop over the size; exact
        rr = ((torch.arange(h, device=device) >= h - window_size_h).long()
              + (torch.arange(h, device=device) >= h - shift_size_h).long())
        cr = ((torch.arange(w, device=device) >= w - window_size_w).long()
              + (torch.arange(w, device=device) >= w - shift_size_w).long())
        img_mask = (rr[:, None] * 3 + cr[None, :]).float().view(1, h, w, 1)
    else:
        img_mask = torch.zeros((1, h, w, 1), device=device)  # 1 H W 1; CPU-alloc + .to() breaks CUDA graph capture
        h_slices = (slice(0, -window_size_h),
                    slice(-window_size_h, -shift_size_h),
                    slice(-shift_size_h, None))
        w_slices = (slice(0, -window_size_w),
                    slice(-window_size_w, -shift_size_w),
                    slice(-shift_size_w, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

    mask_windows = split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def _roll_hw(x, sh, sw, left):
    """torch.roll over dims (1, 2) by (sh, sw), leftward (negative shifts) when `left`. On the
    export path (SMV) as explicit slices: onnxscript's roll takes only
    constant shifts, a symbolic `% n` makes the size unbacked and a symbolic sign test would
    guard, so the direction is a flag and 0 < shift < n is written out (exact data movement).
    Eager keeps torch.roll."""
    if not torch.compiler.is_compiling():
        return torch.roll(x, shifts=(-sh, -sw) if left else (sh, sw), dims=(1, 2))
    for s, d in ((sh, 1), (sw, 2)):
        n = x.shape[d]
        if left:
            x = torch.cat((x.narrow(d, s, n - s), x.narrow(d, 0, s)), d)
        else:
            x = torch.cat((x.narrow(d, n - s, s), x.narrow(d, 0, n - s)), d)
    return x


def _window_layout():
    """export path (SMV): a transformer keeps its tokens in window order (or token-outer, EXPORT_TOKEN_OUTER) from its
    first block to its last (every op but the attention works per token, and a block's self and cross attention share
    one shift), so an attention is a reshape instead of a partition, a merge and rolls; eager keeps the original
    layout"""
    return torch.compiler.is_compiling()


# export path (SMV): the quarter-scale transformer (attn_num_splits 8) runs each block on this many groups of whole
# windows, so a group's tensors stay near the GPU's L2 (its FFN ran at half the rate on 117,760 tokens as on 29,440);
# onnx_export.py exports gmflow_bidir_a once per count and the host picks one by the token count
EXPORT_WINDOW_CHUNKS = 1

# export path (SMV): a shifted block's attention per mask region instead of with the mask (onnx_export.py's
# gmflow_bidir_a_g8r, the host's pick from kGmRegionTokens up): after the roll by half a window only the last window row
# and column hold two regions each (the corner four) and a token attends only inside its own (the mask's -100 is a 0
# weight), so every region is plain attention and the masked kernel (1.5x the plain one) goes; small windows lose more
# to the extra calls than they save
EXPORT_REGION_ATTENTION = False

# export path (SMV): the scales (attn_num_splits) whose transformer keeps its tokens token-outer (_to_groups,
# FeatureTransformer._forward_windowed) instead of in window order: an attention reads its group's slice in the fused
# kernel's own layout instead of 3 copies into it and 1 out; onnx_export.py sets them per variant as measured faster
# (GMFLOW_TOKEN_OUTER)
EXPORT_TOKEN_OUTER = ()

# export path (SMV): the token-outer scales that add gmflow's position encoding after the layout change instead of
# before it (FeatureTransformer.takes_position): with window splits the encoding is one window's table added to every
# window of both frames, so token-outer it is a broadcast add over the outer position axis and the entry stays a
# permute (fused with the encoding it ran at half the bandwidth); onnx_export.py sets them per variant
# (GMFLOW_TOKEN_POSITION)
EXPORT_TOKEN_POSITION = ()


def fold_projections(module):
    """export path (SMV): every TransformerLayer's attention is single-head and linear in its projections, so
    q k^T = s Wq^T Wk t^T and merge(P v) = P t (Wm Wv)^T; the two products (formed in fp64, stored fp32) become buffers
    the export uses instead of the k projection and the merge (two of the four 128 x 128 GEMMs of every attention).
    Called by onnx_export.py after the weights load; eager keeps the four projections"""
    for m in module.modules():
        if isinstance(m, TransformerLayer):
            with torch.no_grad():
                wq, wk, wv, wm = (x.weight.double() for x in (m.q_proj, m.k_proj, m.v_proj, m.merge))
                m.register_buffer("fold_q", (wk.t() @ wq).float(), persistent=False)
                m.register_buffer("fold_v", (wm @ wv).float(), persistent=False)


def _to_windows(x, h, w, k):
    """[B, H*W, C] in raster order -> the same tokens in split_feature's order (K x K windows, each H/K x W/K)"""
    b, _, c = x.shape
    return split_feature(x.view(b, h, w, c), num_splits=k, channel_last=True).reshape(b, h * w, c)


def _from_windows(x, h, w, k):
    """the inverse of _to_windows"""
    b, _, c = x.shape
    return merge_splits(x.reshape(b * k * k, h // k, w // k, c), num_splits=k, channel_last=True).reshape(b, h * w, c)


def _shift_windows(x, h, w, k, into):
    """window-ordered tokens -> the shifted windows' order (into) or back: merge, roll by half a window, split"""
    b, _, c = x.shape
    t = _roll_hw(_from_windows(x, h, w, k).view(b, h, w, c), (h // k) // 2, (w // k) // 2, into)
    return _to_windows(t.reshape(b, h * w, c), h, w, k)


def _region_attention(q, k, v, h, w, num_splits, group, groups):
    """the shifted windows' attention of group `group` of `groups` (whole window rows, window-ordered tokens) as plain
    attention per mask region (EXPORT_REGION_ATTENTION): the interior windows in one batch, the last column's windows
    split into left / right, the last row's into top / bottom, the corner into four; [B, n * L, C] in and out"""
    b, _, c = q.size()
    wh, ww = h // num_splits, w // num_splits
    sh, sw = wh // 2, ww // 2
    th, lw = wh - sh, ww - sw  # a split window's top rows / left columns (the mask's region bounds)
    rows = num_splits // groups
    last = group == groups - 1
    inner = rows - 1 if last else rows  # window rows with interior windows and a last-column window
    qkv = [t.view(b, rows, num_splits, wh, ww, c) for t in (q, k, v)]

    def attend(pick, tokens):
        q_, k_, v_ = (pick(t).reshape(-1, tokens, c) for t in qkv)
        scores = torch.matmul(q_, k_.permute(0, 2, 1)) / (c ** 0.5)
        return torch.matmul(torch.softmax(scores, dim=-1), v_)

    parts = []
    if inner > 0:
        mid = attend(lambda t: t[:, :inner, :num_splits - 1], wh * ww).view(b, inner, num_splits - 1, wh, ww, c)
        left = attend(lambda t: t[:, :inner, num_splits - 1, :, :lw], wh * lw).view(b, inner, wh, lw, c)
        right = attend(lambda t: t[:, :inner, num_splits - 1, :, lw:], wh * sw).view(b, inner, wh, sw, c)
        parts.append(torch.cat((mid, torch.cat((left, right), dim=3).unsqueeze(2)), dim=2))
    if last:
        r = rows - 1
        top = attend(lambda t: t[:, r, :num_splits - 1, :th], th * ww).view(b, num_splits - 1, th, ww, c)
        bottom = attend(lambda t: t[:, r, :num_splits - 1, th:], sh * ww).view(b, num_splits - 1, sh, ww, c)
        tl = attend(lambda t: t[:, r, num_splits - 1, :th, :lw], th * lw).view(b, th, lw, c)
        tr = attend(lambda t: t[:, r, num_splits - 1, :th, lw:], th * sw).view(b, th, sw, c)
        bl = attend(lambda t: t[:, r, num_splits - 1, th:, :lw], sh * lw).view(b, sh, lw, c)
        br = attend(lambda t: t[:, r, num_splits - 1, th:, lw:], sh * sw).view(b, sh, sw, c)
        corner = torch.cat((torch.cat((tl, tr), dim=2), torch.cat((bl, br), dim=2)), dim=1)
        parts.append(torch.cat((torch.cat((top, bottom), dim=2), corner.unsqueeze(1)), dim=1).unsqueeze(1))
    out = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    return out.reshape(b, -1, c)


def _to_groups(x, h, w, k, g):
    """[B, H, W, C] raster -> [g, L, W_g * B, C]: g groups of whole window rows (K x K windows, each L = H/K x W/K
    tokens), token-outer: the position in the window, then the window, then the frame; a group's [L, W_g * B, C] is
    the fused attention kernel's own layout, so an attention needs no copy into it or out of it"""
    b, _, _, c = x.shape
    wh, ww, r = h // k, w // k, k // g
    x = x.reshape(b, g, r, wh, k, ww, c).permute(1, 3, 5, 2, 4, 0, 6)
    return x.reshape(g, wh * ww, b * r * k, c)


def _from_groups(x, b, h, w, k, g):
    """the inverse of _to_groups"""
    c = x.shape[-1]
    wh, ww, r = h // k, w // k, k // g
    return x.reshape(g, wh, ww, r, k, b, c).permute(5, 0, 3, 1, 4, 2, 6).reshape(b, h, w, c)


def _swap_frames(x, b):
    """the cross attention's partner: the frame batch's halves swapped (frame-innermost tokens)"""
    g, n, wb, c = x.shape
    x = x.view(g, n, wb // b, b, c)
    return torch.cat((x[:, :, :, b // 2:], x[:, :, :, :b // 2]), dim=3).view(g, n, wb, c)


def _region_attention_tokens(q, k, v, h, w, num_splits, group, groups, nb):
    """_region_attention on _to_groups' token-outer [L, W * nb, C]: a region = a slice of the position axes, its
    windows x frames the attention batch"""
    n, wb, c = q.shape
    wh, ww = h // num_splits, w // num_splits
    sh, sw = wh // 2, ww // 2
    th, lw = wh - sh, ww - sw  # a split window's top rows / left columns (the mask's region bounds)
    rows = num_splits // groups
    last = group == groups - 1
    inner = rows - 1 if last else rows  # window rows with interior windows and a last-column window
    qkv = [t.view(wh, ww, rows, num_splits, nb, c) for t in (q, k, v)]

    def attend(pick):
        qs, ks, vs = (pick(t) for t in qkv)
        ti, tj, tr_, tx = qs.shape[:4]
        qh, kh, vh = (t.reshape(ti * tj, tr_ * tx * nb, c) for t in (qs, ks, vs))
        s = torch.matmul(qh.permute(1, 0, 2), kh.permute(1, 2, 0)) / (c ** 0.5)
        o = torch.matmul(torch.softmax(s, dim=-1), vh.permute(1, 0, 2)).permute(1, 0, 2)
        return o.reshape(ti, tj, tr_, tx, nb, c)

    parts = []
    if inner > 0:
        mid = attend(lambda t: t[:, :, :inner, :num_splits - 1])
        left = attend(lambda t: t[:, :lw, :inner, num_splits - 1:])
        right = attend(lambda t: t[:, lw:, :inner, num_splits - 1:])
        parts.append(torch.cat((mid, torch.cat((left, right), dim=1)), dim=3))
    if last:
        r = rows - 1
        top = attend(lambda t: t[:th, :, r:, :num_splits - 1])
        bottom = attend(lambda t: t[th:, :, r:, :num_splits - 1])
        tl = attend(lambda t: t[:th, :lw, r:, num_splits - 1:])
        tr = attend(lambda t: t[:th, lw:, r:, num_splits - 1:])
        bl = attend(lambda t: t[th:, :lw, r:, num_splits - 1:])
        br = attend(lambda t: t[th:, lw:, r:, num_splits - 1:])
        corner = torch.cat((torch.cat((tl, tr), dim=1), torch.cat((bl, br), dim=1)), dim=0)
        parts.append(torch.cat((torch.cat((top, bottom), dim=0), corner), dim=3))
    out = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
    return out.reshape(n, wb, c)


def single_head_split_window_attention(q, k, v,
                                       num_splits=1,
                                       with_shift=False,
                                       h=None,
                                       w=None,
                                       attn_mask=None,
                                       windowed=False,
                                       windows=None,
                                       ):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C]; windows: a window-ordered group of that many whole windows (EXPORT_WINDOW_CHUNKS), attn_mask
    # its rows; windowed "tokens": the group as [L, W * B, C] (_to_groups)
    assert q.dim() == k.dim() == v.dim() == 3

    assert h is not None and w is not None

    if windowed == "tokens":
        # the tokens already sit in the (shifted) windows' token-outer order (FeatureTransformer._forward_windowed):
        # the attention batch is the group's windows x frames, the mask repeated per frame; a (group, groups) pair in
        # the mask's place = EXPORT_REGION_ATTENTION
        n, wb, c = q.size()
        nb = wb // windows
        if with_shift and isinstance(attn_mask, tuple):
            return _region_attention_tokens(q, k, v, h, w, num_splits, *attn_mask, nb)
        scores = torch.matmul(q.permute(1, 0, 2), k.permute(1, 2, 0)) / (c ** 0.5)  # [W * B, L, L]
        if with_shift:
            scores += attn_mask.repeat_interleave(nb, dim=0)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v.permute(1, 0, 2)).permute(1, 0, 2)  # [L, W * B, C]

    assert windows is not None or q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * (windows or num_splits * num_splits)

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    scale_factor = c ** 0.5

    if windowed:
        # the tokens already sit in the (shifted) windows' order (FeatureTransformer, _window_layout): each window's
        # L tokens are contiguous, so [B, H*W, C] is [B*K*K, L, C] as it is; a (group, groups) pair in the mask's place
        # = EXPORT_REGION_ATTENTION
        if with_shift and isinstance(attn_mask, tuple):
            return _region_attention(q, k, v, h, w, num_splits, *attn_mask)
        scores =torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)) / scale_factor
        if with_shift:
            scores += attn_mask.repeat(b, 1, 1)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v.view(b_new, -1, c)).view(b, -1, c)

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = _roll_hw(q, shift_size_h, shift_size_w, True)
        k = _roll_hw(k, shift_size_h, shift_size_w, True)
        v = _roll_hw(v, shift_size_h, shift_size_w, True)

    q = split_feature(q, num_splits=num_splits, channel_last=True)  # [B*K*K, H/K, W/K, C]
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)

    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
                          ) / scale_factor  # [B*K*K, H/K*W/K, H/K*W/K]

    if with_shift:
        scores += attn_mask.repeat(b, 1, 1)

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c),
                       num_splits=num_splits, channel_last=True)  # [B, H, W, C]

    # shift back
    if with_shift:
        out = _roll_hw(out, shift_size_h, shift_size_w, False)

    out = out.view(b, -1, c)

    return out


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=1,
                 attention_type='swin',
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 with_shift=False,
                 **kwargs,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn

        self.with_shift = with_shift

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                attn_num_splits=None,
                windowed=False,
                windows=None,
                **kwargs,
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        folded = torch.compiler.is_compiling() and hasattr(self, "fold_q")
        if folded:
            # export path (SMV): fold_projections' matrices; a window group's slice is viewed by the windowed attention
            query = F.linear(query, self.fold_q)
            key = key.contiguous()
            value = F.linear(value, self.fold_v)
        else:
            # single-head attention
            query = self.q_proj(query)  # [B, L, C]
            key = self.k_proj(key)  # [B, L, C]
            value = self.v_proj(value)  # [B, L, C]

        if self.attention_type == 'swin' and attn_num_splits > 1:
            if self.nhead > 1:
                # we observe that multihead attention slows down the speed and increases the memory consumption
                # without bringing obvious performance gains and thus the implementation is removed
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value,
                                                             num_splits=attn_num_splits,
                                                             with_shift=self.with_shift,
                                                             h=height,
                                                             w=width,
                                                             attn_mask=shifted_window_attn_mask,
                                                             windowed=windowed,
                                                             windows=windows,
                                                             )
        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        if not folded:
            message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=256,
                 nhead=1,
                 attention_type='swin',
                 ffn_dim_expansion=4,
                 with_shift=False,
                 **kwargs,
                 ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model,
                                          nhead=nhead,
                                          attention_type=attention_type,
                                          no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion,
                                          with_shift=with_shift,
                                          )

        self.cross_attn_ffn = TransformerLayer(d_model=d_model,
                                               nhead=nhead,
                                               attention_type=attention_type,
                                               ffn_dim_expansion=ffn_dim_expansion,
                                               with_shift=with_shift,
                                               )

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                attn_num_splits=None,
                windowed=False,
                windows=None,
                **kwargs,
                ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                shifted_window_attn_mask=shifted_window_attn_mask,
                                attn_num_splits=attn_num_splits,
                                windowed=windowed,
                                windows=windows,
                                )

        # cross attention and ffn
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     shifted_window_attn_mask=shifted_window_attn_mask,
                                     attn_num_splits=attn_num_splits,
                                     windowed=windowed,
                                     windows=windows,
                                     )

        return source


class FeatureTransformer(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=128,
                 nhead=1,
                 attention_type='swin',
                 ffn_dim_expansion=4,
                 **kwargs,
                 ):
        super(FeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             attention_type=attention_type,
                             ffn_dim_expansion=ffn_dim_expansion,
                             with_shift=True if attention_type == 'swin' and i % 2 == 1 else False,
                             )
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def token_outer(self, attn_num_splits):
        """export path (SMV): this scale runs _forward_windowed (EXPORT_TOKEN_OUTER)"""
        return (self.attention_type == 'swin' and attn_num_splits > 1 and _window_layout()
                and attn_num_splits in EXPORT_TOKEN_OUTER)

    def takes_position(self, attn_num_splits):
        """export path (SMV): this scale adds the position encoding itself, after _to_groups (EXPORT_TOKEN_POSITION),
        so gmflow skips feature_add_position for it"""
        return self.token_outer(attn_num_splits) and attn_num_splits in EXPORT_TOKEN_POSITION

    def forward(self, feature0, feature1,
                attn_num_splits=None,
                **kwargs,
                ):

        b, c, h, w = feature0.shape
        assert self.d_model == c

        if self.token_outer(attn_num_splits):
            return self._forward_windowed(feature0, feature1, attn_num_splits, self.takes_position(attn_num_splits))

        feature0 = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        if self.attention_type == 'swin' and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # concat feature0 and feature1 in batch dimension to compute in parallel
        concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, H*W, C]
        concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, H*W, C]

        win = self.attention_type == 'swin' and attn_num_splits > 1 and _window_layout()
        if win:
            # every block works on window-ordered tokens: the order changes once in, once out, and a shifted block's
            # rolled windows once around the block (its self and cross attention share them)
            concat0 = _to_windows(concat0, h, w, attn_num_splits)
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        # a block works per token and per window, so it runs on groups of whole windows (window order makes each group
        # a contiguous token range, its cross attention the same range of the other frame); the shifts see every token
        g = EXPORT_WINDOW_CHUNKS if win and attn_num_splits == 8 else 1
        kk = attn_num_splits * attn_num_splits
        assert kk % g == 0
        for layer in self.layers:
            shifted = win and layer.self_attn.with_shift
            regions = shifted and EXPORT_REGION_ATTENTION  # the attention takes its (group, groups), not the mask
            if shifted:
                concat0 = _shift_windows(concat0, h, w, attn_num_splits, True)
                concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)
            if g == 1:
                concat0 = layer(concat0, concat1,
                                height=h,
                                width=w,
                                shifted_window_attn_mask=(0, 1) if regions else shifted_window_attn_mask,
                                attn_num_splits=attn_num_splits,
                                windowed=win,
                                )
            else:
                n = h * w // g
                concat0 = torch.cat([layer(concat0[:, i * n:(i + 1) * n], concat1[:, i * n:(i + 1) * n],
                                           height=h,
                                           width=w,
                                           shifted_window_attn_mask=(i, g) if regions else
                                           shifted_window_attn_mask[i * kk // g:(i + 1) * kk // g],
                                           attn_num_splits=attn_num_splits,
                                           windowed=win,
                                           windows=kk // g,
                                           ) for i in range(g)], dim=1)
            if shifted:
                concat0 = _shift_windows(concat0, h, w, attn_num_splits, False)

            # update feature1
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        if win:
            concat0 = _from_windows(concat0, h, w, attn_num_splits)
        feature0, feature1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

        # reshape back
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return feature0, feature1

    def _forward_windowed(self, feature0, feature1, k, with_position):
        """export path (SMV, _window_layout + EXPORT_TOKEN_OUTER): every block works on one token-outer stream
        (_to_groups), so an attention reads its group's [L, W * B, C] slice as it is (window-ordered tokens cost 3 copies
        into the fused kernel's layout and 1 out per attention); the layout changes once in, once out, and a shifted
        block's rolled windows once around the block (its self and cross attention share them); a block works per token
        and per window, so at the quarter scale it runs on EXPORT_WINDOW_CHUNKS groups of whole window rows (a group's
        tensors stay near the GPU's L2); with_position = add feature_add_position's per-window table here, once over
        the position axis (takes_position)"""
        b, c, h, w = feature0.shape
        wsh, wsw = h // k, w // k
        mask = generate_shift_window_attn_mask(input_resolution=(h, w), window_size_h=wsh, window_size_w=wsw,
                                               shift_size_h=wsh // 2, shift_size_w=wsw // 2, device=feature0.device)
        g = EXPORT_WINDOW_CHUNKS if k == 8 else 1
        kk = k * k
        assert k % g == 0
        bb = 2 * b
        x0 = _to_groups(torch.cat((feature0, feature1), dim=0).permute(0, 2, 3, 1), h, w, k, g)
        if with_position:
            pos = PositionEmbeddingSine(num_pos_feats=c // 2)(feature0[:1, :, :wsh, :wsw])  # [1, C, H/K, W/K]
            x0 = x0 + pos[0].flatten(1).t()[None, :, None, :]
        x1 = _swap_frames(x0, bb)
        sh, sw = wsh // 2, wsw // 2
        for layer in self.layers:
            shifted = layer.self_attn.with_shift
            regions = shifted and EXPORT_REGION_ATTENTION  # the attention takes its (group, groups), not the mask
            if shifted:
                x0 = _to_groups(_roll_hw(_from_groups(x0, bb, h, w, k, g), sh, sw, True), h, w, k, g)
                x1 = _swap_frames(x0, bb)
            x0 = torch.stack([layer(x0[i], x1[i], height=h, width=w,
                                    shifted_window_attn_mask=(i, g) if regions else mask[i * kk // g:(i + 1) * kk // g],
                                    attn_num_splits=k, windowed="tokens", windows=kk // g) for i in range(g)], dim=0)
            if shifted:
                x0 = _to_groups(_roll_hw(_from_groups(x0, bb, h, w, k, g), sh, sw, False), h, w, k, g)
            x1 = _swap_frames(x0, bb)
        out = _from_groups(x0, bb, h, w, k, g).permute(0, 3, 1, 2).contiguous()  # [2B, C, H, W]
        feature0, feature1 = out.chunk(chunks=2, dim=0)
        return feature0, feature1


class FeatureFlowAttention(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(FeatureFlowAttention, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                local_window_attn=False,
                local_window_radius=1,
                **kwargs,
                ):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow,
                                                  local_window_radius=local_window_radius)

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(self, feature0, flow,
                                  local_window_radius=1,
                                  ):
        # per-pixel batched gemms chunked below 65535 batches: same TRT-RTX gridDim.z
        # launch-cap bug as matching.local_correlation_softmax (see the note there)
        from .matching import GEMM_BATCH_CHUNK
        assert flow.size(1) == 2
        assert local_window_radius > 0

        b, c, h, w = feature0.size()
        if torch.compiler.is_compiling():
            # export path (SMV): the chunk loop below unrolls by H * W,
            # which a size-free export cannot hold. The two per-pixel products (1 x C by C x k*k,
            # then 1 x k*k by k*k x 2) as multiply + sum over one axis: no batched gemm, so no
            # launch cap and nothing per size. A different summation order: equivalent, measured
            # closer to eager fp32 than the chunked engine
            ks = 2 * local_window_radius + 1
            q = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1)  # [B, C, H*W]
            kp = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)
            kw = F.unfold(kp, kernel_size=ks, padding=local_window_radius).view(b, c, ks * ks, h * w)
            scores = (q.unsqueeze(2) * kw).sum(1) / (c ** 0.5)  # [B, k*k, H*W]
            prob = torch.softmax(scores, dim=1)
            fw = F.unfold(flow, kernel_size=ks, padding=local_window_radius).view(b, 2, ks * ks, h * w)
            return (prob.unsqueeze(1) * fw).sum(2).view(b, 2, h, w).contiguous()  # [B, 2, H, W]

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b, h * w, 1, c)  # [B, H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]

        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b, h * w, c, kernel_size ** 2)  # [B, H*W, C, (2R+1)^2]

        flow_window = F.unfold(flow, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 2*(2R+1)^2), H*W]

        flow_window = flow_window.view(b, 2, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b, h * w, kernel_size ** 2, 2)  # [B, H*W, (2R+1)^2, 2]

        hw = h * w
        step = max(1, GEMM_BATCH_CHUNK // b)
        outs = []
        for i in range(0, hw, step):
            scores = torch.matmul(feature0_reshape[:, i:i + step],
                                  feature0_window[:, i:i + step]) / (c ** 0.5)  # [B, chunk, 1, (2R+1)^2]
            prob = torch.softmax(scores, dim=-1)
            outs.append(torch.matmul(prob, flow_window[:, i:i + step]).squeeze(-2))  # [B, chunk, 2]

        out = torch.cat(outs, dim=1).view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out
