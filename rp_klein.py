# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Klein 3 (FLUX.2 Klein) graph builder. The server builds the API graph with the same nodes as the official ComfyUI template
# (UNETLoader · CLIPLoader flux3 · VAELoader · ReferenceMultiLatent · KSampler…) through an optional host module that provides
# grafo_klein3(); if it is not loaded, armar() raises a clear error. This module only BUILDS the graph; it never queues anything.
#   armar({prompt (English), formato?, mp?, refs?: [names in input/], base?: bool, semilla?})
#        → {grafo, semilla, tam, prefijo, modo: crear|editar, version, avisos}
import os, random, re, sys

from aiohttp import web
from server import PromptServer

from . import rp_comun as comun

try:
    import folder_paths
    INPUT = folder_paths.get_input_directory()
except Exception:
    INPUT = os.path.abspath(".")

R = PromptServer.instance.routes
FORMATOS = ("1:1", "4:5", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9", "21:9")
BASE_UNET = "flux3_image_klein_int8_convrot.safetensors"        # the undistilled Base model from the official templates
_RX_NOM = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_ .()\-]{0,150}\.(png|jpe?g|webp)$", re.I)


def _em():
    """The module that provides grafo_klein3() / MOTORES (bundled: rp_klein3_graph.py)."""
    from . import rp_klein3_graph
    return rp_klein3_graph


def tam_formato(txt, mp):
    """"16:9" and megapixels → (width, height) in multiples of 16 (Klein up to 4 MP)."""
    a, b = (int(x) for x in txt.split(":"))
    k = (mp * 1e6 / (a * b)) ** 0.5
    return int(round(a * k / 16)) * 16, int(round(b * k / 16)) * 16


def tam_imagen(nombre):
    """Size of an input image, or None."""
    try:
        from PIL import Image
        with Image.open(os.path.join(INPUT, nombre)) as im:
            return im.size
    except Exception:
        return None


def armar(d):
    em = _em()
    if em is None or not hasattr(em, "grafo_klein3"):
        raise ValueError("the Klein 3 engine is not available in this installation (try restarting ComfyUI)")
    prompt = str(d.get("prompt") or "").strip()
    refs = [str(x) for x in (d.get("refs") or [])][:10]
    for n in refs:
        if not _RX_NOM.match(n) or "/" in n or "\\" in n or not os.path.isfile(os.path.join(INPUT, n)):
            raise ValueError("reference image \"%s\" not found in ComfyUI's input folder" % n[:60])
    if not prompt:
        raise ValueError("describe what you want to see or what you want to change")
    base = bool(d.get("base"))
    avisos = []
    # Prompt enhancers often write "cinematic lightning" (a bolt) for "lighting", which adds storms and rain.
    # Unless the prompt talks about storms or bolts, fix it and warn.
    if re.search(r"\blightning\b", prompt, re.I) and not re.search(r"\b(storm|thunder|bolt|strike|rayo|tormenta|relámpago)\b", prompt, re.I):
        prompt = re.sub(r"\blightning\b", "lighting", prompt, flags=re.I)
        avisos.append("Replaced \"lightning\" with \"lighting\". If you really wanted a lightning bolt, write \"lightning bolt\" or \"storm\".")
    if d.get("motor") == "krea" and not refs:            # text-to-image can also use Krea 2 (optional host module); editing is Klein 3 only
        fs = sys.modules.get((__package__ or "") + ".fotograma_srv")
        if fs is None or not hasattr(fs, "armar"):
            raise ValueError("Krea 2 text-to-image is not available in this installation (try restarting ComfyUI)")
        f = str(d.get("formato") or "16:9")
        if f not in FORMATOS:
            raise ValueError("invalid aspect ratio")
        K = fs.armar({"prompt": prompt, "motor": "krea", "destino": "libre", "tam": list(tam_formato(f, min(2.6, max(0.25, float(d.get("mp") or 1.0))))), "semilla": d.get("semilla")})
        return {"grafo": K["grafo"], "semilla": K["semilla"], "tam": K["tam"], "prefijo": K["prefijo"], "modo": "crear", "version": "Krea 2", "avisos": K.get("avisos") or []}
    mejorar = bool(d.get("mejorar", not refs and d.get("prefijo") != "t2i"))   # prompt enhancer: on by default for text-to-image, caller decides otherwise
    conservar = bool(d.get("conservar", True))            # "keep the rest" when editing a single image
    if refs and mejorar and len(refs) > 1:
        mejorar = False; avisos.append("The prompt enhancer is skipped with several images: it would remove the tags that say which image is which.")
    if refs:
        # "image 2" / "picture 2" → <ref_image_1>, which is how the official template cites references (the first is <ref_image_0>).
        # Without these tags Klein cannot tell the images apart and may return one of them unchanged.
        def _etq(m):
            n = int(m.group(1))
            return "<ref_image_%d>" % (n - 1) if 1 <= n <= len(refs) else m.group(0)
        prompt = re.sub(r"\b(?:the\s+)?(?:image|picture|reference|photo)\s+(\d{1,2})\b", _etq, prompt, flags=re.I)
        if len(refs) == 1 and conservar and not mejorar and not re.search(r"exactly the same|keep everything|keep the rest", prompt, re.I):
            prompt = prompt.rstrip() + " Keep everything else exactly the same: the subject, the pose, the composition, the lighting and the background."
        elif len(refs) > 1 and "<ref_image_" not in prompt:
            prompt = "Edit <ref_image_0> using the other images as references. " + prompt
            avisos.append("Say which image you mean (\"image 2\", \"photo 1\") so Klein knows which one to edit and which to use as reference.")
    try:
        semilla = int(d.get("semilla")) if d.get("semilla") not in (None, "") else random.randint(0, 2 ** 48)
    except Exception:
        semilla = random.randint(0, 2 ** 48)
    if refs:                                             # edit: output takes the shape of the first image (up to 4 MP, Klein's single-reference limit)
        t0 = tam_imagen(refs[0])
        if not t0:
            raise ValueError("could not read the size of the image to edit")
        k = min(1.0, (4e6 / (t0[0] * t0[1])) ** 0.5)
        tam = (max(256, int(round(t0[0] * k / 16)) * 16), max(256, int(round(t0[1] * k / 16)) * 16))
        if k < 1:
            avisos.append("The image is larger than 4 MP: it is reduced to %d×%d so Klein actually applies the change." % tam)
        if not base:
            avisos.append("The Distilled model can warm the colour of the whole image; to preserve the rest of the scene use \"Base\".")
    else:
        f = str(d.get("formato") or "16:9")
        if f not in FORMATOS:
            raise ValueError("invalid aspect ratio")
        mp = min(4.0, max(0.25, float(d.get("mp") or 1.0)))
        tam = tam_formato(f, mp)
        t = d.get("tam")             # exact size chosen by the user
        if isinstance(t, (list, tuple)) and len(t) == 2:
            try:
                w, h = int(t[0]), int(t[1])
            except (TypeError, ValueError):
                raise ValueError("invalid size")
            if not (256 <= w <= 4096 and 256 <= h <= 4096 and w * h <= 8.6e6):
                raise ValueError("size out of range (256–4096 px per side, up to ~8 MP)")
            tam = (int(round(w / 16)) * 16, int(round(h / 16)) * 16)
    prefijo = "np_klein/t2i" if d.get("prefijo") == "t2i" else "np_klein/movil_" + ("edit" if refs else "t2i")
    # Sampler recipes: create Distilled = euler · simple · 15 steps · cfg 1; create Base = euler · 50 · cfg 3.5;
    # edit Distilled = euler · 16 · cfg 1; edit Base = euler · 25 (or 50) · cfg 3.5.
    # (The official template has er_sde in the inner KSampler, but the subgraph value that wins is euler.)
    if refs:
        pasos_base = 50 if str(d.get("pasos") or "") == "50" else 25   # edit with Base: 25 (fast) or 50 (the template suggests 50 for best quality)
        receta = dict(pasos=pasos_base, sampler="euler", cfg=3.5) if base else dict(pasos=16, sampler="euler", cfg=1)
    else:
        receta = dict(pasos=50, sampler="euler", cfg=3.5) if base else dict(pasos=15, sampler="euler", cfg=1)
    unet = BASE_UNET if base else None
    falta = None
    try:                                                 # clear warning if the model is missing (instead of "Prompt outputs failed validation")
        import folder_paths                              # noqa: F401  (warning only: the graph is still built)
        nombre_unet = unet or em.MOTORES["klein3"]["modelos"][0][1]
        if not comun.hay_modelo("diffusion_models", nombre_unet):
            falta = nombre_unet
            avisos.insert(0, "⚠ Model \"%s\" is missing (models/diffusion_models): ComfyUI will reject the job. Download it or use the other quality." % nombre_unet)
    except Exception:
        pass
    lienzo = None if refs else str(d.get("formato") or "16:9")
    G = em.grafo_klein3(prompt, refs, prefijo, tam, semilla, modelo=unet, mejorar=mejorar, lienzo=lienzo, **receta)
    return {"grafo": G, "semilla": semilla, "tam": list(tam), "prefijo": prefijo, "modo": "editar" if refs else "crear", "version": "Base" if base else "Distilled",
            "mejorar": mejorar, "conservar": bool(refs) and len(refs) == 1 and conservar and not mejorar, "prompt_en": prompt, "pasos": receta["pasos"], "falta": falta, "avisos": avisos}
