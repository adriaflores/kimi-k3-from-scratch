"""Causal suffix-invariance checks for the public inference entry point.

The original M9 frozen protocol is archived in the pre-cleanup-m9 Git tag.
"""

import torch

from kimi_k3.inference import generate_tokens, next_token_logits
from kimi_k3.model import KimiK3, KimiK3Config
from kimi_k3.utils import load_config


def test_actual_tiny_kimi_reference_suffix_is_causal():
    torch.manual_seed(781)
    cfg = load_config('configs/model/kimi_k3_tiny.yaml', KimiK3Config)
    model = KimiK3(cfg).eval()
    prefix = [7, 13, 29]
    with torch.inference_mode():
        logits_a, _ = model(torch.tensor([[7, 13, 29, 0]]))
        logits_b, _ = model(torch.tensor([[7, 13, 29, 47]]))
    # Replacing strictly future tokens cannot influence prefix logits.
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], rtol=1e-5, atol=1e-5)
    public = next_token_logits(model, prefix, device=torch.device('cpu'), pad_id=0)
    torch.testing.assert_close(public, logits_a[0, 2].float(), rtol=1e-5, atol=1e-5)


def test_public_generation_same_seed_same_token_ids():
    torch.manual_seed(781)
    cfg = load_config('configs/model/kimi_k3_tiny.yaml', KimiK3Config)
    model = KimiK3(cfg).eval()
    options = dict(device=torch.device('cpu'), mode='sample', seed=42,
                   max_new_tokens=4, top_k=20, top_p=0.92, eos_id=0)
    a = generate_tokens(model, [7, 13, 29], **options)
    b = generate_tokens(model, [7, 13, 29], **options)
    assert a == b and 1 <= len(a) <= 4
