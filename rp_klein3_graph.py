# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# FLUX.3 Klein graph built from the nodes of ComfyUI's official templates (image_flux3_klein_*_image_edit / _t2i):
# UNETLoader · CLIPLoader "flux3" · VAELoader · CLIPTextEncode · [ReferenceMultiLatent] · EmptyFlux2LatentImage · KSampler · VAEDecode · SaveImage.

MOTORES = {"klein3": {"modelos": [("diffusion_models", "flux3_image_klein_sd_int8_convrot.safetensors"),
                                  ("text_encoders", "qwen3vl_4b_int8_convrot.safetensors"),
                                  ("vae", "flux2-vae.safetensors")],
                      "licencia": "FLUX Kommunity: non-commercial; outputs may be used commercially under US$5M/year revenue"}}


def grafo_klein3(prompt, imagenes, prefijo, tam, semilla, pasos=16, modelo=None, sampler="euler", cfg=1, mejorar=False, lienzo=None):
    """With images = edit (the first one is edited; up to 10 references); without images = text to image.
    Official recipes: edit distilled = euler / 16 / cfg 1; edit Base = euler / 25 / cfg 3.5."""
    if mejorar:
        raise ValueError("Prompt enhancement is not included in RenderPasses")
    M = MOTORES["klein3"]["modelos"]
    P = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": modelo or M[0][1], "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": M[1][1], "type": "flux3", "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": M[2][1]}},
         "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
         "44": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": tam[0], "height": tam[1], "batch_size": 1}}}
    imagenes = list(imagenes or [])[:10]
    if imagenes:
        P["32"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
        mp = round(min(4.0, max(0.5, tam[0] * tam[1] / 1e6)), 2) if len(imagenes) == 1 else 1.0
        P["17"] = {"class_type": "ReferenceMultiLatent", "inputs": {"resolution": "resize", "resolution.megapixels": mp, "positive": ["4", 0], "negative": ["32", 0], "vae": ["3", 0]}}
        for k, img in enumerate(imagenes):
            P["img%d" % k] = {"class_type": "LoadImage", "inputs": {"image": img}}
            P["17"]["inputs"]["images.image%d" % k] = ["img%d" % k, 0]
        if len(imagenes) == 1:
            # A single reference goes in at the EXACT output size ("keep resolution"): ReferenceMultiLatent counts megapixels as
            # 1024×1024, so a w·h/1e6 reference came out ~2.4 % larger than the canvas and Klein's output drifted by a few pixels.
            P["ref0"] = {"class_type": "ImageScale", "inputs": {"image": ["img0", 0], "upscale_method": "lanczos", "width": tam[0], "height": tam[1], "crop": "disabled"}}
            P["17"]["inputs"] = {"resolution": "keep resolution", "positive": ["4", 0], "negative": ["32", 0], "vae": ["3", 0], "images.image0": ["ref0", 0]}
        pos, neg = ["17", 0], ["17", 1]
    else:
        P["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}}
        pos, neg = ["4", 0], ["5", 0]
    P["7"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": pos, "negative": neg, "latent_image": ["44", 0], "seed": semilla, "steps": pasos, "cfg": cfg,
                                                   "sampler_name": sampler, "scheduler": "simple", "denoise": 1}}
    P["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    P["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefijo}}
    return P
