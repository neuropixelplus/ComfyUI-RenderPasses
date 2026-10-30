# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Node: a KSampler whose live preview shows EVERY image of the batch at once, tiled in a grid (2 → 1×2, 4 → 2×2, 9 → 3×3…).
# ComfyUI's own preview decodes only the first latent of the batch (x0[0]), so with 4 plates you watch one. Here each step decodes all
# of them with the model's Latent2RGB factors (or TAESD when that is the preview method and its decoder is installed) and sends the grid
# through the normal progress bar: it shows on the node in the graph and to any page listening to ComfyUI's websocket (the Create panel).
# Sampling itself is exactly ComfyUI's KSampler (same noise, same result for the same seed).
import math

import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

CAT = "RenderPasses"
LADO_MAX = 1024          # long side of the preview grid


def _rejilla(imgs):
    """[PIL images of the same size] → one PIL grid, as square as possible, at most LADO_MAX on its long side."""
    from PIL import Image
    n = len(imgs)
    cols = math.ceil(math.sqrt(n))
    filas = math.ceil(n / cols)
    w, h = imgs[0].size
    hueco = max(2, w // 64)
    G = Image.new("RGB", (cols * w + (cols - 1) * hueco, filas * h + (filas - 1) * hueco), (12, 12, 12))
    for i, im in enumerate(imgs):
        G.paste(im, ((i % cols) * (w + hueco), (i // cols) * (h + hueco)))
    k = LADO_MAX / max(G.size)
    return G.resize((max(1, round(G.size[0] * k)), max(1, round(G.size[1] * k)))) if k < 1 else G


def callback_lote(model, steps):
    """Like latent_preview.prepare_callback, but the preview is the grid of the whole batch."""
    previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        vista = None
        if previewer is not None:
            if getattr(x0, "is_nested", False):
                x0 = x0.tensors[0]
            try:
                imgs = [previewer.decode_latent_to_preview(x0[i:i + 1]) for i in range(min(x0.shape[0], 16))]
                vista = ("JPEG", _rejilla(imgs), LADO_MAX)
            except Exception:
                vista = previewer.decode_latent_to_preview_image("JPEG", x0)
        pbar.update_absolute(step + 1, total_steps, vista)
    return callback


class RenderPassesBatchPreviewKSampler:
    """KSampler with a live preview of every image in the batch (a grid), instead of only the first one."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CAT
    DESCRIPTION = ("ComfyUI's KSampler, but the live preview shows every image of the batch at once in a grid (ComfyUI's own preview "
                   "shows only the first one). Same noise and same result as KSampler for the same seed.")

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        latente = latent_image
        x = comfy.sample.fix_empty_latent_channels(model, latente["samples"], latente.get("downscale_ratio_spacial", None),
                                                   latente.get("downscale_ratio_temporal", None))
        ruido = comfy.sample.prepare_noise(x, seed, latente.get("batch_index"))
        muestras = comfy.sample.sample(model, ruido, steps, cfg, sampler_name, scheduler, positive, negative, x, denoise=denoise,
                                       noise_mask=latente.get("noise_mask"), callback=callback_lote(model, steps),
                                       disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        out = latente.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = muestras
        return (out,)


NODE_CLASS_MAPPINGS = {"RenderPassesBatchPreviewKSampler": RenderPassesBatchPreviewKSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"RenderPassesBatchPreviewKSampler": "RenderPasses · KSampler (batch preview)"}
