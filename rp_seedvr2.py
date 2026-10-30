# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Upscale an image to 2K / 4K with SeedVR2. Fibo 2.0 was trained at ~1 MP (at 4K it repeats objects), so images are generated
# at 1–2K and upscaled; works the same for Klein and Krea.
# Recipe = the official ComfyUI template "utility_seedvr2_3b_int8_upscale_image" (subgraph "Upscale Image (SeedVR2 3B Int8)"):
#   LoadImage → JoinImageWithAlpha → ResizeImageMaskNode (scale by multiplier, lanczos) → SeedVR2Preprocess → VAEEncodeTiled (512/128/4096/8)
#   → SeedVR2Conditioning → KSampler (1 step, cfg 1, euler, simple, denoise 1) → VAEDecodeTiled → SeedVR2PostProcessing (color "lab") → SaveImage.
#   The template uses color correction "none", which lets SeedVR2 boost contrast and saturation; "lab" (the node's default)
#   transfers the original's colour in CIELAB and keeps the new detail.
# Only the multiplier (4K = long side 3840) and the output prefix differ from the template.
import os
import random
import re
import shutil
import sys

from aiohttp import web
from server import PromptServer

from . import rp_comun as comun

try:
    import folder_paths
except Exception:
    folder_paths = None

R = PromptServer.instance.routes
UNET = "seedvr2_3b_int8_convrot.safetensors"
VAE = "seedvr2_ema_vae_fp16.safetensors"
LADO = {"4k": 3840, "2k": 2560}
MAX_LADO = 8192


def _dirs():
    if folder_paths:
        return folder_paths.get_input_directory(), folder_paths.get_output_directory()
    return os.path.abspath("input"), os.path.abspath("output")


def _hay(carpeta, nombre):
    return comun.hay_modelo(carpeta, nombre)     # same relative path with any separator, or same file name in a subfolder


def _origen(d):
    """The image in input/ (LoadImage only reads from there): if it comes from output/ it is copied as np_<name>."""
    inp, out = _dirs()
    if d.get("archivo"):
        ruta = comun.resolver(inp, d["archivo"])
        if not ruta or not os.path.isfile(ruta):
            raise ValueError("image not found")
        return os.path.relpath(ruta, inp).replace("\\", "/"), ruta
    ruta = comun.resolver(out, d.get("rel"))
    if not ruta or not os.path.isfile(ruta) or os.path.splitext(ruta)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError("image not found")
    nombre = "np_" + os.path.basename(ruta)
    shutil.copyfile(ruta, os.path.join(inp, nombre))
    return nombre, ruta


def multiplicador(w, h, destino):
    if destino in ("x2", "x4"):
        return float(destino[1:])
    m = re.fullmatch(r"l(\d{3,4})", str(destino))   # "l1984" = up to that long side (used by Fibo's large output with several images)
    lado = int(m.group(1)) if m else LADO.get(destino)
    if not lado:
        raise ValueError("unknown target (4k, 2k, x2 or x4)")
    m = lado / float(max(w, h))
    if m < 1.05:
        raise ValueError("the image is already %d px on its long side: no need to upscale it to %s" % (max(w, h), destino.upper()))
    return round(min(m, 4.0), 4)


def grafo(nombre, mult, semilla, prefijo):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": nombre}},
        "50": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["1", 0], "alpha": ["1", 1]}},
        "51": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "52": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "57": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["50", 0], "resize_type": "scale by multiplier", "resize_type.multiplier": mult, "scale_method": "lanczos"}},
        "58": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["57", 0]}},
        "48": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["58", 0], "vae": ["51", 0], "tile_size": 512, "overlap": 128, "temporal_size": 4096, "temporal_overlap": 8}},
        "61": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["52", 0], "vae_conditioning": ["48", 0]}},
        "54": {"class_type": "KSampler", "inputs": {"model": ["52", 0], "positive": ["61", 0], "negative": ["61", 1], "latent_image": ["48", 0], "seed": semilla,
                                                     "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "55": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["54", 0], "vae": ["51", 0], "tile_size": 512, "overlap": 128, "temporal_size": 4096, "temporal_overlap": 8}},
        "59": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["55", 0], "original_resized_images": ["57", 0], "color_correction_method": "lab"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["59", 0], "filename_prefix": prefijo}},
    }


def encadenar(G, imagen, ancho, alto, destino, semilla, salida="8", prefijo=None):
    """Append the SeedVR2 nodes (same recipe as the official template) AFTER another graph, in the same job: the model's image
    (e.g. Fibo at ~1 MP, its native size) → ×mult → the SaveImage node `salida`. No JoinImageWithAlpha (the image comes straight
    from the VAE, without alpha). Returns (W, H, mult). Node ids are prefixed with "sv" to avoid collisions."""
    mult = multiplicador(ancho, alto, destino)
    g = grafo("_", mult, semilla, prefijo or "x")
    del g["1"], g["50"], g["9"]
    g["57"]["inputs"]["input"] = imagen
    for k, n in g.items():
        for kk, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and v[0] in g:
                n["inputs"][kk] = ["sv" + v[0], v[1]]
        G["sv" + k] = n
    G["sv57"]["_meta"] = {"title": "RenderPasses upscale · SeedVR2 ×%g" % mult}
    G[salida]["inputs"]["images"] = ["sv59", 0]
    if prefijo:
        G[salida]["inputs"]["filename_prefix"] = prefijo
    return int(round(ancho * mult)), int(round(alto * mult)), mult


def faltan_modelos():
    return [n for c, n in (("diffusion_models", UNET), ("vae", VAE)) if not _hay(c, n)]

