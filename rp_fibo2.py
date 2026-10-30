# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Fibo 2.0 (Bria) graph builder. One 8B model that CREATES (Dev, 30 steps, cfg 5) and EDITS (Edit Turbo, 4 steps, cfg 1) from
# structured JSON captions (VGL). Native in ComfyUI (TextEncodeFibo2 node + CLIPLoader "fibo2"). Weights: Comfy-Org/Fibo-2.0.
# Model license: CC BY-NC 4.0 — NON-COMMERCIAL USE ONLY (commercial use requires an agreement with Bria).
# Supports plain-text edits, up to 4 references (<image_1>…), object removal, mask edits and outpainting by painting the area
# 50 % grey (the same trick bria-vgl uses: grey out the region and ask to replace the grey area), and text→image with JSON from a local LLM.
# Trained at ~1 MP: 2 MP is still fine; 4 MP repeats objects (warned in the UI).
# This module only BUILDS graphs; the web page queues them. The local VGL bridge (LLM) is not a render.
#   armar({modo: crear|editar, prompt (text or VGL JSON), formato?, tam?, refs?: [input], cajas?: [{tipo, bbox, desc}], ext?: {l,r,u,d, desc}, semilla?})
#        → {grafo, semilla, tam, prefijo, avisos, modo, imagen_extra?}
#   vgl(prompt, semilla?) → (json, seconds)   (short prompt → VGL JSON with a local Qwen3.8-27B)
import asyncio, functools, hashlib, importlib, json, math, os, random, re, sys, time

from aiohttp import web
from server import PromptServer

from . import rp_comun as comun

try:
    import folder_paths
    INPUT = folder_paths.get_input_directory()
except Exception:
    folder_paths = None
    INPUT = os.path.abspath(".")

R = PromptServer.instance.routes
AQUI = os.path.dirname(os.path.abspath(__file__))
UNET_DEV = ("fibo2_dev_int8_convrot.safetensors", "fibo2_dev_bf16.safetensors")
UNET_EDIT = ("fibo2_edit_turbo_int8_convrot.safetensors", "fibo2_edit_turbo_bf16.safetensors")
CLIP_CREAR = ("qwen3vl_4b_int8_convrot.safetensors", "qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_4b_bf16.safetensors")
CLIP_EDITAR = ("qwen3vl_4b_bf16.safetensors", "qwen3vl_4b_fp8_scaled.safetensors")     # int8 fails with images (ComfyUI issue 17248)
VAE = "flux2-vae.safetensors"
QWEN38 = "qwen3.8_27b_w4a8.safetensors"
FORMATOS = ("1:1", "4:5", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9", "21:9")
# model-card negative for text→image (pushes away from smooth, textureless and artefacts); empty for edits (the null caption it was trained with)
NEGATIVO_T2I = ('{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast","overexposed highlights","lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, lacking fine texture"},"style_medium":"","context":"","artistic_style":""}')
_RX_NOM = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_ .()\-]{0,150}\.(png|jpe?g|webp)$", re.I)
CLAVES_VGL = ("short_description", "objects", "background_setting", "lighting", "aesthetics", "photographic_characteristics", "style_medium")


def _hay(carpeta, nombre):
    return comun.hay_modelo(carpeta, nombre)     # same relative path with any separator, or same file name in a subfolder


def _primero(carpeta, lista):
    return next((n for n in lista if _hay(carpeta, n)), None)


def _nodo_listo():
    try:
        import nodes
        return "TextEncodeFibo2" in nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return False


def _nodos_cargados():
    try:
        import nodes
        return bool(nodes.NODE_CLASS_MAPPINGS)
    except Exception:
        return False


def tam_formato(txt, mp):
    """"16:9" and megapixels → (width, height) in multiples of 16."""
    a, b = (int(x) for x in txt.split(":"))
    k = (mp * 1e6 / (a * b)) ** 0.5
    return int(round(a * k / 16)) * 16, int(round(b * k / 16)) * 16


def _imagen_input(nombre):
    if not _RX_NOM.match(nombre) or "/" in nombre or "\\" in nombre or not os.path.isfile(os.path.join(INPUT, nombre)):
        raise ValueError("image \"%s\" not found in ComfyUI's input folder" % nombre[:60])
    return os.path.join(INPUT, nombre)


def _guarda_png(im, semilla_txt):
    """Derived image (grey area / grey-padded canvas) → input/ with a content-hash name, so LoadImage can read it."""
    h = hashlib.sha1((semilla_txt + str(im.size)).encode("utf-8") + im.tobytes()[:4096]).hexdigest()[:12]
    nombre = "np_fibo_%s.png" % h
    ruta = os.path.join(INPUT, nombre)
    if not os.path.isfile(ruta):
        im.save(ruta)
    return nombre


def _gris_cajas(ruta, cajas):
    """Box mask: paint each box [y0, x0, y1, x1] (0–1000) 50 % grey and return (name, phrase). The bria-vgl trick to edit only one region."""
    from PIL import Image
    im = Image.open(ruta).convert("RGB")
    w, h = im.size
    frases = []
    for c in cajas:
        y0, x0, y1, x1 = [max(0, min(1000, float(v))) for v in c["bbox"]]
        if y1 - y0 < 8 or x1 - x0 < 8:
            raise ValueError("a box is too small")
        im.paste((128, 128, 128), (int(x0 / 1000 * w), int(y0 / 1000 * h), int(x1 / 1000 * w), int(y1 / 1000 * h)))
        d = str(c.get("desc") or "").strip()
        if c.get("tipo") == "quita":
            frases.append("remove what is in the gray area and fill it with the surrounding background, continuing it seamlessly")
        elif d:
            frases.append("replace the gray area with %s" % d.rstrip("."))
    if not frases:
        raise ValueError("describe what goes in each marked area")
    if len(frases) > 1:     # several areas: numbered by position
        frases = ["in the %s gray area %s" % (["first", "second", "third", "fourth", "fifth"][min(i, 4)], f[0].lower() + f[1:]) for i, f in enumerate(frases)]
    return _guarda_png(im, "cajas" + json.dumps(cajas, sort_keys=True)), "; ".join(frases) + ". Keep everything outside the gray area exactly the same."


def _gris_mascara(ruta, ruta_mascara, tipo, desc):
    """SAM3 mask (white = what changes) → that area 50 % grey + the phrase. Feathered edge to avoid a hard outline."""
    from PIL import Image, ImageFilter
    im = Image.open(ruta).convert("RGB")
    m = Image.open(ruta_mascara).convert("L")
    if m.size != im.size:
        m = m.resize(im.size, Image.BILINEAR)
    if not m.point(lambda v: 255 if v > 127 else 0).getbbox():
        raise ValueError("the mask is empty: select again")
    im.paste(Image.new("RGB", im.size, (128, 128, 128)), mask=m.filter(ImageFilter.GaussianBlur(1.2)))
    d = str(desc or "").strip().rstrip(".")
    if tipo == "quita":
        txt = "Remove what is in the gray area and fill it with the surrounding background, continuing it seamlessly. Keep everything outside the gray area exactly the same."
    elif d:
        txt = "Replace the gray area with %s. Keep everything outside the gray area exactly the same." % d
    else:
        raise ValueError("describe what goes in the selected area")
    return _guarda_png(im, "mask" + os.path.basename(ruta_mascara)), txt


def _zona(ruta_mascara):
    """Where the selection is, in words ("in the lower left of the image"): with the SAM3 label Fibo finds it better than with a grey area."""
    from PIL import Image
    m = Image.open(ruta_mascara).convert("L").point(lambda v: 255 if v > 127 else 0)
    bb = m.getbbox()
    if not bb:
        raise ValueError("the mask is empty: select again")
    cx, cy = (bb[0] + bb[2]) / 2 / m.size[0], (bb[1] + bb[3]) / 2 / m.size[1]
    h = "left" if cx < 0.36 else "right" if cx > 0.64 else "center"
    v = "upper" if cy < 0.36 else "lower" if cy > 0.64 else "middle"
    return "center of" if (h == "center" and v == "middle") else ("%s %s of" % (v, h) if h != "center" else "%s part of" % v)


def _texto_quita(ruta_mascara, etq):
    """REMOVE: words only, naming the selection and its area. Painting grey does NOT work for removal: Fibo leaves a grey object."""
    etq = re.sub(r"[^\w\s\-',]", "", str(etq or "")).strip()[:60] or "selected object"
    return "Remove the %s located in the %s the image and fill the area with the surrounding background, continuing it seamlessly. Keep everything else exactly the same." % (etq, _zona(ruta_mascara))


def _outpaint(ruta, ext, desc):
    """Outpainting: the image on a larger canvas with the new border 50 % grey; the model fills it ("extend the scene"). Up to +300 % per side."""
    from PIL import Image
    im = Image.open(ruta).convert("RGB")
    w, h = im.size
    l, r, u, d = [max(0.0, min(3.0, float(ext.get(k) or 0) / 100.0)) for k in ("l", "r", "u", "d")]   # up to +300 % (a 16:9 → 9:16 reframe needs +216 % height)
    if not (l or r or u or d):
        raise ValueError("choose which side to extend")
    W, H = int(round(w * (1 + l + r))), int(round(h * (1 + u + d)))
    lienzo = Image.new("RGB", (W, H), (128, 128, 128))
    lienzo.paste(im, (int(round(w * l)), int(round(h * u))))
    lados = " and ".join(n for n, v in (("left", l), ("right", r), ("top", u), ("bottom", d)) if v)
    txt = ("Extend the scene to the %s to fill the gray area: continue the same place seamlessly, matching perspective, lighting, color grade, scale and style. " % lados) + (str(desc).strip().rstrip(".") + ". " if desc else "") + "Keep the original photo exactly as it is."
    return _guarda_png(lienzo, "ext" + json.dumps(ext, sort_keys=True)), txt, (W, H)


@functools.lru_cache(maxsize=64)
def _tam_nativo(ancho, alto):
    """Same as trained_size() in comfy_extras/nodes_fibo2.py (Bria's size for an aspect): multiples of 16 within ±10 % of 1 MP, least crop first, then closest to 1 MP."""
    ar, area = ancho / alto, 1024 * 1024
    mejor = None
    for w in range(64, 2 * area // 64, 16):
        for h in range(max(64, 16 * math.ceil(0.9 * area / w / 16)), 16 * math.floor(1.1 * area / w / 16) + 1, 16):
            k = (round(1 - min(w / h / ar, ar * h / w), 9), abs(w * h - area), -w)
            if mejor is None or k < mejor[0]:
                mejor = (k, w, h)
    return mejor[1], mejor[2]


def _refs_alineadas(G, imagenes, tam):
    """Multi-image edit at a chosen size. With several images the node encodes each reference at ITS native size (~1 MP) while the output
    uses another size, so they don't line up and the subject shifts. Here the node no longer adds the references (no VAE: only text and what
    Qwen sees); we add them with ReferenceLatent in the same order: the 1st image scaled (centre crop) to the output size, the rest at native size."""
    from PIL import Image
    del G["5"]["inputs"]["vae"]
    pos, neg = ["5", 0], ["5", 1]
    for k, n in enumerate(imagenes, 1):
        if k == 1:
            wh = tam
        else:
            with Image.open(os.path.join(INPUT, n)) as im:
                wh = _tam_nativo(*im.size)
        G["esc%d" % k] = {"class_type": "ImageScale", "inputs": {"image": ["img%d" % k, 0], "upscale_method": "lanczos", "width": wh[0], "height": wh[1], "crop": "center"}}
        G["vae%d" % k] = {"class_type": "VAEEncode", "inputs": {"pixels": ["esc%d" % k, 0], "vae": ["3", 0]}}
        G["rp%d" % k] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": ["vae%d" % k, 0]}}
        G["rn%d" % k] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": ["vae%d" % k, 0]}}
        pos, neg = ["rp%d" % k, 0], ["rn%d" % k, 0]
    G["6"]["inputs"]["positive"], G["6"]["inputs"]["negative"] = pos, neg


def armar(d):
    modo = "editar" if d.get("modo") == "editar" else "crear"
    prompt = str(d.get("prompt") or "").strip()
    avisos = []
    try:
        semilla = int(d["semilla"]) if d.get("semilla") not in (None, "") else random.randint(0, 2 ** 48)
    except (TypeError, ValueError):
        semilla = random.randint(0, 2 ** 48)
    if _nodos_cargados() and not _nodo_listo():
        raise ValueError("this ComfyUI does not support Fibo 2.0 yet (the TextEncodeFibo2 node is missing): update ComfyUI")
    if modo == "crear":
        unet = _primero("diffusion_models", UNET_DEV)
        clip = _primero("text_encoders", CLIP_CREAR)
        if folder_paths is not None and not unet:
            raise ValueError("Fibo 2.0 Dev model (%s) is missing from diffusion_models" % UNET_DEV[0])
        unet, clip = unet or UNET_DEV[0], clip or CLIP_CREAR[0]
        if not prompt:
            raise ValueError("describe what you want to see")
        f = str(d.get("formato") or "16:9")
        if f not in FORMATOS:
            raise ValueError("invalid aspect ratio")
        t = d.get("tam")
        if isinstance(t, (list, tuple)) and len(t) == 2:
            try:
                w, h = int(t[0]), int(t[1])
            except (TypeError, ValueError):
                raise ValueError("invalid size")
            if not (256 <= w <= 4096 and 256 <= h <= 4096):
                raise ValueError("size out of range (256–4096 px per side)")
            tam = (int(round(w / 16)) * 16, int(round(h / 16)) * 16)
        else:
            tam = tam_formato(f, min(8.5, max(0.25, float(d.get("mp") or 1.0))))
        mp = tam[0] * tam[1] / 1e6
        if mp > 2.3:
            avisos.append("Fibo was trained at ~1 MP: at %.1f MP the scene gets wider and repeats objects. Better generate at 1–2K and upscale." % mp)
        imagenes, steps, cfg = [], 30, 5.0
        neg = NEGATIVO_T2I if d.get("negativo", True) else ""
        prefijo = "np_klein/fibo_crear"
        wh = tam
    else:
        unet = _primero("diffusion_models", UNET_EDIT)
        clip = _primero("text_encoders", CLIP_EDITAR)
        if folder_paths is not None and not unet:
            raise ValueError("Fibo 2.0 Edit Turbo model (%s) is missing from diffusion_models" % UNET_EDIT[0])
        if folder_paths is not None and not clip:
            raise ValueError("editing needs the qwen3vl_4b text encoder in bf16 or fp8 (int8 fails with images)")
        unet, clip = unet or UNET_EDIT[0], clip or CLIP_EDITAR[0]
        refs = [str(x) for x in (d.get("refs") or [])][:4]
        if not refs:
            raise ValueError("choose the image you want to edit")
        for n in refs:
            _imagen_input(n)
        imagenes = list(refs)
        extra = None
        if d.get("ext"):
            extra, txt, (W, H) = _outpaint(_imagen_input(refs[0]), d["ext"], str(d.get("ext_desc") or d["ext"].get("desc") or ""))
            imagenes[0] = extra
            prompt = (prompt + " " + txt).strip() if prompt else txt
            if W / H > 2.4 or H / W > 2.4:
                avisos.append("The canvas is very elongated (more than 2.4:1): results may look odd.")
        elif d.get("mascara"):
            if d.get("mascara_tipo") == "quita":
                txt = _texto_quita(_imagen_input(str(d["mascara"])), d.get("mascara_etq"))     # no grey: the original image as is
            else:
                extra, txt = _gris_mascara(_imagen_input(refs[0]), _imagen_input(str(d["mascara"])), d.get("mascara_tipo"), d.get("mascara_desc"))
                imagenes[0] = extra
            prompt = (prompt + " " + txt).strip() if prompt else txt
        elif d.get("cajas"):
            extra, txt = _gris_cajas(_imagen_input(refs[0]), d["cajas"])
            imagenes[0] = extra
            prompt = (prompt + " " + txt).strip() if prompt else txt
        if not prompt:
            raise ValueError("describe what you want to change")
        steps, cfg, neg = 4, 1.0, ""
        prefijo = "np_klein/fibo_editar"
        wh = (0, 0)           # 0 = the trained size (~1 MP, with the 1st image's aspect)
        tam = None
        t = d.get("tam")      # Original · 1K · 2K; the node scales the image to that size (single image)
        if isinstance(t, (list, tuple)) and len(t) == 2 and not d.get("ext"):
            try:
                w, h = int(t[0]), int(t[1])
            except (TypeError, ValueError):
                raise ValueError("invalid size")
            if not (256 <= w <= 4096 and 256 <= h <= 4096):
                raise ValueError("size out of range (256–4096 px per side)")
            tam = wh = (int(round(w / 16)) * 16, int(round(h / 16)) * 16)
            if tam[0] * tam[1] / 1e6 > 2.3:
                avisos.append("Fibo was trained at ~1 MP: editing at %.1f MP may repeat details. If it looks odd, try 2K and then upscale to 4K." % (tam[0] * tam[1] / 1e6))
        if tam and len(imagenes) > 1:
            # As in Bria's pipeline (see comfy_extras/nodes_fibo2.py), several images are each seen at their trained ~1 MP size, so the framing
            # shifts if the output size differs. Fibo works at its trained size and SeedVR2 upscales to the chosen size in the SAME job.
            from PIL import Image
            with Image.open(os.path.join(INPUT, imagenes[0])) as im:
                nat = _tam_nativo(*im.size)
            if tuple(tam) != tuple(nat):
                objetivo = tam
                tam = wh = nat
                if d.get("salida") not in ("2k", "4k") and max(objetivo) > max(nat):
                    d["salida"] = "l%d" % max(objetivo)
                avisos.append("With several images Fibo works at its trained size (%d×%d), as in Bria's pipeline, so the framing does not shift." % nat)
        if len(refs) > 1 and "<image_" not in prompt:
            avisos.append("With several images, refer to each one as <image_1>, <image_2>… and say what to take from each.")
    G = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "fibo2", "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
         "5": {"class_type": "TextEncodeFibo2", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": neg, "vae": ["3", 0], "width": wh[0], "height": wh[1], "batch_size": 1}},
         "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "seed": semilla, "steps": steps, "cfg": cfg,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
         "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
         "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefijo}}}
    for k, n in enumerate(imagenes, 1):
        G["img%d" % k] = {"class_type": "LoadImage", "inputs": {"image": n}}
        G["5"]["inputs"]["images.image_%d" % k] = ["img%d" % k, 0]
    if modo == "editar" and tam and len(imagenes) > 1:
        _refs_alineadas(G, imagenes, tam)
    G["8"]["_meta"] = {"title": "RenderPasses output · Fibo 2.0"}
    grande = None
    if (d.get("salida") in ("2k", "4k") or re.fullmatch(r"l\d{3,4}", str(d.get("salida") or ""))) and tam:   # large output: Fibo generates at its native size and SeedVR2 upscales it in the SAME job
        try:
            ES = importlib.import_module(".rp_seedvr2", __package__)
        except ImportError:
            ES = None
        if ES is None:
            avisos.append("Could not add the large output: the upscaler is not available (restart ComfyUI). Output is %d×%d." % tuple(tam))
        elif ES.faltan_modelos():
            avisos.append("The large output needs SeedVR2 (%s): install the missing models. This time the output is %d×%d." % (", ".join(ES.faltan_modelos()), tam[0], tam[1]))
        else:
            try:
                W, H, m = ES.encadenar(G, ["7", 0], tam[0], tam[1], d["salida"], semilla, "8", prefijo + "_" + ("grande" if d["salida"].startswith("l") else d["salida"]))
                grande = [W, H]
                avisos.append("Fibo generates at %d×%d and SeedVR2 upscales to %d×%d (×%s) in the same job." % (tam[0], tam[1], W, H, ("%.2f" % m).rstrip("0").rstrip(".")))
            except ValueError as e:
                avisos.append("Large output: %s. Output is %d×%d." % (e, tam[0], tam[1]))
    return {"grafo": G, "semilla": semilla, "tam": grande or (list(tam) if tam else None), "tam_modelo": list(tam) if tam else None, "prefijo": prefijo, "avisos": avisos, "modo": modo, "pasos": steps,
            "modelo": "Fibo 2.0 Dev" if modo == "crear" else "Fibo 2.0 Edit Turbo", "imagen_extra": (imagenes[0] if modo == "editar" and imagenes[0] != refs[0] else None)}


# ───────── local VGL bridge: short prompt → structured JSON (Bria's prompts, MIT, with a local Qwen3.8-27B via TextGenerate)
def _vgl_textos(editar=False):
    """Create: Bria's "generate" prompts. Edit: "edit_image", the local model SEES the image and describes the FINAL scene (only the request changes)."""
    def leer(n):
        with open(os.path.join(AQUI, "fibo2_vgl", n), encoding="utf-8") as f:
            return f.read()
    pre = "edit" if editar else "generate"
    sistema = leer(pre + "_system.txt").replace("{style_preset_block}", "").replace("{llm_input_schema}", leer(pre + "_schema.txt"))
    sistema += "\nAll string values in the JSON must be written in English, even if the request is in another language. Output only the JSON object."
    return sistema, leer(pre + "_final.txt").strip()


def grafo_vgl(prompt, semilla=1, imagen=None):
    """imagen (name in input/) = bridge for EDITING: Qwen3.8 looks at it (describes the real scene and keeps pose and lighting)."""
    sistema, final = _vgl_textos(editar=bool(imagen))
    G = {"c": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN38, "type": "stable_diffusion", "device": "default"}},
         "g": {"class_type": "TextGenerate", "inputs": {"clip": ["c", 0], "prompt": final.replace("{user_prompt}", prompt), "max_length": 3200 if imagen else 2800, "sampling_mode": "on",
                                                        "sampling_mode.temperature": 0.4, "sampling_mode.top_k": 40, "sampling_mode.top_p": 0.9, "sampling_mode.min_p": 0.05,
                                                        "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": int(semilla), "thinking": False, "use_default_template": True,
                                                        "mtp": "auto", "system_prompt": sistema}},
         "p": {"class_type": "PreviewAny", "inputs": {"source": ["g", 0]}}}
    if imagen:
        G["i"] = {"class_type": "LoadImage", "inputs": {"image": imagen}}
        G["g"]["inputs"]["image"] = ["i", 0]
    return G


def extrae_json(txt):
    """First JSON object in the LLM text; must contain the VGL keys. Raises ValueError otherwise."""
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        raise ValueError("the local model did not return JSON")
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError("the local model returned malformed JSON (%s)" % e)
    if not isinstance(j, dict) or any(k not in j for k in CLAVES_VGL):
        raise ValueError("the local model's JSON is missing some VGL sections")
    return j


async def vgl(prompt, semilla=1, espera=420, imagen=None):
    import aiohttp
    if not _hay("text_encoders", QWEN38) and folder_paths is not None:
        raise ValueError("the VGL bridge needs the %s model in text_encoders" % QWEN38)
    if imagen:
        _imagen_input(imagen)   # ValueError if not in input/ or contains ../
    puerto = getattr(PromptServer.instance, "port", None) or 8000
    base = "http://127.0.0.1:%s" % puerto
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        async with s.post(base + "/prompt", json={"prompt": comun.modelos_os(grafo_vgl(prompt, semilla, imagen))}) as r:
            res = await r.json(content_type=None)
        pid = (res or {}).get("prompt_id")
        if not pid:
            raise ValueError("ComfyUI rejected the VGL bridge job: %s" % str((res or {}).get("error") or res)[:200])
        while time.time() - t0 < espera:
            await asyncio.sleep(1.5)
            async with s.get(base + "/history/" + pid) as r:
                h = ((await r.json(content_type=None)) or {}).get(pid)
            if h and (h.get("status") or {}).get("status_str") == "error":
                raise ValueError("the local model failed while building the JSON")
            if h and h.get("outputs"):
                txt = "".join(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for o in h["outputs"].values() for v in (o.get("text") or []))
                return extrae_json(txt), round(time.time() - t0, 1)
    raise ValueError("the local model took too long")


SAM3_CKPT = "sam3.1_multiplex_fp16.safetensors"


def _entrega_mascara(ruta, nombre, m):
    """L mask (255 = what changes) at image size → input/: the mask (RGB) + a preview with the area in red + coverage %."""
    from PIL import Image
    orig = Image.open(ruta).convert("RGB")
    if m.size != orig.size:
        m = m.resize(orig.size, Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
    rojo = Image.composite(Image.new("RGB", orig.size, (255, 40, 60)), orig, m.point(lambda v: 120 if v else 0))
    n_m = _guarda_png(Image.merge("RGB", (m, m, m)), "samask" + nombre + str(time.time()))
    n_v = _guarda_png(rojo, "samvista" + nombre + str(time.time()))
    cobertura = round(100.0 * sum(1 for v in m.resize((64, 64)).getdata() if v) / 4096, 1)
    return {"mascara": n_m, "vista": n_v, "ancho": orig.size[0], "alto": orig.size[1], "cobertura": cobertura}


async def sam3_mascaras(nombre, pos, neg, texto, individual, espera=150, caja=None):
    """Run SAM3 on an input/ image → (width, height, [binary L masks at image size], seconds).
    Points (0–1) and text go through separate SAM3_Detect inputs and are merged; without text no "conditioning" is sent (points only).
    individual=True returns each detection separately.
    caja = (x0, y0, x1, y1) px → core PrimitiveBoundingBox node into SAM3_Detect's `bboxes`: without text, SAM3 segments what is inside."""
    import aiohttp
    import uuid
    from PIL import Image
    _imagen_input(nombre)
    with Image.open(os.path.join(INPUT, nombre)) as im0:
        W, H = im0.size
    pts = lambda L: json.dumps([{"x": int(round(min(1, max(0, float(p["x"]))) * (W - 1))), "y": int(round(min(1, max(0, float(p["y"]))) * (H - 1)))} for p in (L or [])[:16]])
    prefijo = "np_fibo_sam/" + uuid.uuid4().hex[:10]
    P = {"1": {"class_type": "LoadImage", "inputs": {"image": nombre}},
         "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_CKPT}},
         "4": {"class_type": "SAM3_Detect", "inputs": {"model": ["2", 0], "image": ["1", 0], "threshold": 0.3, "refine_iterations": 2, "individual_masks": bool(individual)}},
         "5": {"class_type": "GrowMask", "inputs": {"mask": ["4", 0], "expand": 1, "tapered_corners": True}},
         "6": {"class_type": "MaskToImage", "inputs": {"mask": ["5", 0]}},
         "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefijo}}}
    if texto:
        P["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": texto, "clip": ["2", 1]}}
        P["4"]["inputs"]["conditioning"] = ["3", 0]
    if caja:
        x0, y0, x1, y1 = [int(v) for v in caja]
        P["bb"] = {"class_type": "PrimitiveBoundingBox", "inputs": {"x": x0, "y": y0, "width": max(1, x1 - x0), "height": max(1, y1 - y0)}}
        P["4"]["inputs"]["bboxes"] = ["bb", 0]
    if pos:
        P["4"]["inputs"]["positive_coords"] = pts(pos)
    if neg:
        P["4"]["inputs"]["negative_coords"] = pts(neg)
    base = "http://127.0.0.1:%s" % (getattr(PromptServer.instance, "port", None) or 8000)
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        async with s.post(base + "/prompt", json={"prompt": comun.modelos_os(P)}) as r:
            res = await r.json(content_type=None)
        pid = (res or {}).get("prompt_id")
        if not pid:
            raise ValueError("ComfyUI rejected the SAM3 job: %s" % str((res or {}).get("node_errors") or (res or {}).get("error"))[:240])
        while time.time() - t0 < espera:
            await asyncio.sleep(1.0)
            async with s.get(base + "/history/" + pid) as r:
                h = ((await r.json(content_type=None)) or {}).get(pid)
            st = (h or {}).get("status") or {}
            if st.get("status_str") == "error":
                raise ValueError("SAM3 failed")
            if h and st.get("completed"):
                imgs = [v for o in (h.get("outputs") or {}).values() for v in o.get("images") or []]
                if not imgs:
                    raise ValueError("SAM3 finished without a mask")
                L = []
                for img in imgs:
                    carpeta = folder_paths.get_output_directory() if img.get("type", "output") == "output" else folder_paths.get_temp_directory()
                    mi = Image.open(os.path.join(carpeta, img.get("subfolder") or "", img["filename"])).convert("L")
                    if mi.size != (W, H):
                        mi = mi.resize((W, H), Image.BILINEAR)
                    L.append(mi.point(lambda v: 255 if v > 127 else 0))
                return W, H, L, round(time.time() - t0, 1)
    raise ValueError("SAM3 took too long")


async def mascara_sam3(nombre, pos, neg, texto, espera=150):
    """SAM3: clicked points (0–1) and/or text ("person", "cat", "face"…) → PNG mask in input/ + preview with the area in red."""
    from PIL import Image
    ruta = _imagen_input(nombre)
    W, H, masks, seg = await sam3_mascaras(nombre, pos, neg, texto or "person", bool(neg), espera)
    # With negative (red) points SAM3 returns each detection separately and any detection under a red point is discarded
    # (a text query like "face" would otherwise still find every face)
    ng = [(int(round(min(1, max(0, float(p["x"]))) * (W - 1))), int(round(min(1, max(0, float(p["y"]))) * (H - 1)))) for p in (neg or [])[:16]]
    m = Image.new("L", (W, H), 0)
    for mi in masks:
        if any(mi.getpixel(q) for q in ng):
            continue
        m.paste(255, mask=mi)
    if ng and not m.getbbox():
        raise ValueError("nothing is left selected with the red points: remove one (↶) or add a green point on what you want")
    if not m.getbbox():
        raise ValueError("SAM3 found nothing there: click closer to the centre or describe it (person, cat, jacket…)")
    res = _entrega_mascara(ruta, nombre, m)
    res["segundos"] = seg
    return res


def _mascara_de_png(data_url, W, H):
    import base64, io
    from PIL import Image
    b = str(data_url or "")
    if "," in b:
        b = b.split(",", 1)[1]
    raw = base64.b64decode(b)
    if len(raw) > 60 * 1024 * 1024:
        raise ValueError("the mask file is too large")
    im = Image.open(io.BytesIO(raw))
    im.load()
    m = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
    if m.size != (W, H):
        m = m.resize((W, H), Image.BILINEAR)
    return m.point(lambda v: 255 if v > 100 else 0)


def _ps_mod():
    """Optional Photoshop round-trip host module (None when running standalone)."""
    return sys.modules.get((__package__ or "") + ".ps_srv")


def _ps_reg_ruta():
    return os.path.join(INPUT, "np_ps", "mascaras.json")


def _ps_preparar(nombre_img, mascara):
    """Copy for Photoshop: the image with the current selection painted pure red → input/np_ps/<stem>_mascara_psN.png."""
    from PIL import Image
    ruta = _imagen_input(nombre_img)
    carpeta = os.path.join(INPUT, "np_ps")
    os.makedirs(carpeta, exist_ok=True)
    tallo = re.sub(r"[^A-Za-z0-9_\-]+", "_", os.path.splitext(nombre_img)[0]).strip("_")[:50] or "imagen"
    n = 1
    while any(os.path.exists(os.path.join(carpeta, "%s_mascara_ps%d%s" % (tallo, n, e))) for e in (".png", ".psd", ".jpg", ".jpeg", ".tif", ".tiff")):
        n += 1
    nombre = "%s_mascara_ps%d" % (tallo, n)
    im = Image.open(ruta).convert("RGB")
    if mascara:
        try:
            m = Image.open(_imagen_input(str(mascara))).convert("L")
            if m.size != im.size:
                m = m.resize(im.size, Image.BILINEAR)
            im.paste(Image.new("RGB", im.size, (255, 0, 0)), mask=m.point(lambda v: 255 if v > 127 else 0))
        except ValueError:
            pass
    dst = os.path.join(carpeta, nombre + ".png")
    im.save(dst)
    reg = comun.leer_json(_ps_reg_ruta(), {}) or {}
    reg[nombre] = {"archivo": nombre_img, "mtime": os.path.getmtime(dst), "t": time.time()}
    reg = dict(sorted(reg.items(), key=lambda kv: kv[1].get("t", 0))[-100:])
    comun.escribir_json(_ps_reg_ruta(), reg)
    return nombre, dst


def _ps_leer(nombre):
    """If Photoshop saved a new version: the RED paint (100 %, not present in the original) → mask."""
    import numpy as np
    from PIL import Image
    reg = comun.leer_json(_ps_reg_ruta(), {}) or {}
    r = reg.get(nombre)
    if not r:
        raise ValueError("unknown Photoshop mask")
    carpeta = os.path.join(INPUT, "np_ps")
    mejor = None
    for e in (".png", ".psd", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = os.path.join(carpeta, nombre + e)
        if os.path.isfile(p) and (not mejor or os.path.getmtime(p) > mejor[1]):
            mejor = (p, os.path.getmtime(p))
    if not mejor or mejor[1] <= r["mtime"] + 0.5 or time.time() - mejor[1] < 1.2:
        return None
    ruta = _imagen_input(r["archivo"])
    orig = Image.open(ruta).convert("RGB")
    with Image.open(mejor[0]) as g:
        g.load()
        g = g.convert("RGB")
    if g.size != orig.size:
        g = g.resize(orig.size, Image.BILINEAR)
    a = np.asarray(g).astype(np.int16)
    o = np.asarray(orig).astype(np.int16)
    rojo = (a[..., 0] > 150) & (a[..., 0] - np.maximum(a[..., 1], a[..., 2]) > 90)
    cambio = np.abs(a - o).sum(axis=2) > 60
    m = Image.fromarray(((rojo & cambio) * 255).astype(np.uint8), "L")
    r["mtime"] = mejor[1]
    reg[nombre] = r
    comun.escribir_json(_ps_reg_ruta(), reg)
    if not m.getbbox():
        return {"vacia": True}
    return _entrega_mascara(ruta, r["archivo"], m)

