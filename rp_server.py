# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Server side of RenderPasses: improves the finish of a FOLDER of approved backgrounds (BGs) with several image-edit engines and
# extracts render passes from the results. Every engine uses a tested recipe:
#   · Krea 2  → Krea2EditModelPatch + Krea2EditGroundedEncode (official Turbo, 8 steps, cfg 1), ~2 MP.
#   · Klein 3 → rp_klein.armar with "base" (undistilled Base, euler 25, cfg 3.5: the distilled model warms the colour), up to 4 MP.
#   · Fibo 2  → rp_fibo2.armar in "editar" mode (Edit Turbo, 4 steps, cfg 1), at its trained size (~1 MP).
# Each BG is copied to input/ (bglook_<batch>_<name>) and saved to output/renderpasses/<batch>/<engine>/<name>_NNNNN_.png. Jobs are queued
# grouped by engine to avoid swapping models on every image. Contact sheets are built per BG and per batch, and each engine's graph
# (first BG) is saved to user/default/workflows/RenderPasses/ so it can be opened and tweaked in ComfyUI.
# Licences differ per engine (some are non-commercial); the page shows them.
#   GET  /renderpasses/prompt              → {prompt, defecto, motores, licencias, pases_ia, capacidades: {hdr_ltx}}
#   POST /renderpasses/prompt {prompt}     → saves the master prompt (input/renderpasses/prompt_maestro.txt)
#   POST /renderpasses/lanzar {carpeta, prompt?, motores?: [krea2, klein3, fibo2], semilla?, limite?} → {lote, trabajos, avisos}
#   GET  /renderpasses/estado?lote=        → {lote, total, hechos, bgs: [{nombre, original, salidas: {motor: url}, hoja}], hoja}
#   GET  /renderpasses/lotes               → batches, newest first
#   GET  /renderpasses/requisitos          → install check: missing models (with their folder) and nodes per engine and pass
#   POST /renderpasses/cancelar {lote}     → removes the batch's jobs and stops its background tasks
#   POST /renderpasses/cripto_psd          → Cryptomatte, Objects and per-layer mattes from the artist's layered PSD or a colour mask + legend (rp_psd.py)
#   POST /renderpasses/splat3d {lote, motor, bg, rehacer?} → builds the plate's live relightable 3D splat (rp_splat3d.py); GET /renderpasses/splat3d/<lote>/<motor>/<bg>/ opens it in the PBR Director
import asyncio
import contextvars
import datetime
import glob
import json
import os
import random
import re
import shutil
import sys
import time

from aiohttp import web
from server import PromptServer

from . import rp_comun as comun
from . import rp_psd

try:
    import folder_paths
    INPUT = folder_paths.get_input_directory()
    OUTPUT = folder_paths.get_output_directory()
    USER = folder_paths.get_user_directory()
except Exception:
    INPUT, OUTPUT, USER = os.path.abspath("input"), os.path.abspath("output"), os.path.abspath("user")

R = PromptServer.instance.routes
DIR_CONF = os.path.join(INPUT, "renderpasses")
DIR_SAL = os.path.join(OUTPUT, "renderpasses")
DIR_WF = os.path.join(USER, "default", "workflows", "RenderPasses")
EXT = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
MOTORES = {"krea2": "Krea 2", "klein3": "Klein 3", "fibo2": "Fibo 2", "krea2v2": "Krea 2 V2", "moss": "MOSS", "moss_prof": "MOSS · depth + color"}
LICENCIAS = {"krea2": "Krea 2 Community: commercial use only if the company earns < US$1M/year (otherwise a Krea enterprise licence)",
             "klein3": "FLUX Kommunity: non-commercial (commercial outputs < US$5M/year)", "fibo2": "CC BY-NC: non-commercial only",
             "krea2v2": "General Edit V2 (conradlocke): personal / non-commercial only; commercial use = author's licence. Plus the Krea 2 licence",
             "moss": "MOSS-Image (OpenMOSS): Apache 2.0, commercial use; official Comfy-Org edit LoRA. NOTE: it changes the framing (to improve an approved BG use 'MOSS · depth')",
             "moss_prof": "MOSS-Image (Apache 2.0) + colour/depth control LoRA (darask0, via Comfy-Org) + MoGe-3 or MoGe-2 (MIT): check the adapter licence before production"}
# Strength of the FINISH improvement (not the lighting): appended to the master prompt.
INTENSIDAD = {
    "normal": "",   # the master prompt is used as-is (the old "fuerte" sentence pushed Fibo to invent texture and shadows)
    "sutil": " Keep the improvement subtle: a light polish of the same image.",
    "media": "",
    "fuerte": (" Push the finish clearly beyond the original: noticeably more texture detail and material richness, deeper ambient occlusion, "
               "more dimensionality, crisper and more polished overall, as a final film frame — without changing the light and without adding "
               "anything new to the scene.")}
# The goal is the FINISH, not the lighting. Written as an instruction (concrete edit instructions work; a "change nothing" description
# barely changes anything). It names no scene elements and forbids adding things: naming "plants, water…" made Fibo invent them on plain
# backgrounds. The prompt is content-agnostic so it works on any image; it can be edited on the page.
PROMPT_DEFECTO = (
    "Upgrade the finish of this exact image to the quality of a final frame of a high-end animated feature film from a top animation studio, "
    "in its own existing style: richer and "
    "more detailed textures and materials on what is already there, subtle specular highlights, soft ambient occlusion and contact shadows, "
    "clean crisp edges and a polished cinematic color grade. Do not add, remove or replace anything: no new objects, elements, scenery or "
    "background details; plain or simple areas stay plain and simple. Keep the same framing, composition, lighting direction, time of day, "
    "subjects and art style.")
_RX_LOTE = re.compile(r"^\d{8}_\d{6}$")


def _limpio(nombre):
    """Safe name for input/ (the Klein recipes only accept letters, digits, space, dot, dash and parentheses)."""
    base = re.sub(r"[^A-Za-z0-9_.()\- ]+", "_", os.path.splitext(os.path.basename(nombre))[0]).strip(" ._") or "bg"
    return base[:80]


def prompt_maestro():
    try:
        with open(os.path.join(DIR_CONF, "prompt_maestro.txt"), encoding="utf-8") as f:
            t = f.read().strip()
            return t or PROMPT_DEFECTO
    except OSError:
        return PROMPT_DEFECTO


# ───────── Per-batch cancel ─────────
# POST /renderpasses/cancelar removes a batch's ComfyUI jobs, but the Python background tasks (Cryptomatte, objects, parallax, MOSS tiles,
# HDR, 4K upscale, per-plate prompts) would keep queueing new ones. Cancelling stamps the batch; every task started for that batch BEFORE
# the stamp stops at its next step (_seguir(), also called by _correr_texto / _correr_imagenes while they wait) and writes "cancelled" to
# its json. A new launch for the batch clears the flag (_reanudar): tasks started after it run normally.
_CANCELADO = {}                                              # batch → monotonic time of its last cancel
_LANZADO = {}                                                # batch → monotonic time of its last launch
_TAREA = contextvars.ContextVar("rp_tarea", default=None)   # (batch, start time) of the background task running in this context


class Cancelado(Exception):
    def __str__(self):
        return "cancelled"


def _reanudar(lote):
    """A new job for this batch is being launched: its cancel flag no longer applies (older tasks still stop)."""
    _LANZADO[lote] = time.monotonic()


def lote_cancelado(lote):
    """True if the batch was cancelled and nothing was launched for it since."""
    return bool(lote) and _CANCELADO.get(lote, 0.0) > _LANZADO.get(lote, 0.0)


def _seguir():
    """Raises Cancelado if the batch of the running background task was cancelled after the task started."""
    t = _TAREA.get()
    if t and _CANCELADO.get(t[0], -1.0) >= t[1]:
        raise Cancelado()


def _en_fondo(lote, coro):
    """Runs a coroutine as a background task of `lote` (it can then be cancelled with the batch)."""
    t0 = time.monotonic()

    async def tarea():
        _TAREA.set((lote, t0))
        try:
            await coro
        except Cancelado:
            print("[RenderPasses] batch %s: background task cancelled" % lote)
    return asyncio.get_running_loop().create_task(tarea())


def _estado_fallo(e):
    """What a background task writes to its json when it stops: "cancelled" or "error: …"."""
    return "cancelled" if isinstance(e, Cancelado) else "error: %s" % e


def _terminado(estado):
    """A background task's json state that will not change any more."""
    e = str(estado or "")
    return e == "listo" or e == "cancelled" or e.startswith("error")


def _lote_de(ruta):
    """The batch a file or folder under output/renderpasses/ belongs to (None if it is not inside one)."""
    try:
        p = os.path.relpath(os.path.abspath(ruta), os.path.abspath(DIR_SAL)).replace("\\", "/").split("/")[0]
    except ValueError:
        return None
    return p if _RX_LOTE.match(p) else None


def _mod(nombre):
    """Engine modules ship in this package (rp_*.py); maps the legacy module names to them."""
    nombre = {"klein_movil_srv": "rp_klein", "fibo2_srv": "rp_fibo2", "escalar_imagen_srv": "rp_seedvr2", "traductor_srv": "rp_translate"}.get(nombre)
    if not nombre:
        return None
    import importlib
    return importlib.import_module("." + nombre, __package__)


def _tam(ruta, mp):
    from PIL import Image
    with Image.open(ruta) as im:
        w, h = im.size
    k = (mp * 1e6 / (w * h)) ** 0.5
    return max(256, int(round(w * k / 16)) * 16), max(256, int(round(h * k / 16)) * 16)


def grafo_krea2(prompt, imagen, tam, semilla):
    """Krea 2 Turbo edit recipe: Turbo + krea2_identity_edit_v1_2 LoRA, qwen_image VAE, the edit patch with the image and VAE, and a negative
    grounded on the same image (8 steps, cfg 1). The Turbo alone does not edit (without the LoRA the output came out cropped and tinted)."""
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
            "1b": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "krea2_identity_edit_v1_2.safetensors", "strength_model": 1.0}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "4": {"class_type": "LoadImage", "inputs": {"image": imagen}, "_meta": {"title": "RP image"}},
            "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "megapixels": round(tam[0] * tam[1] / 1e6, 2), "resolution_steps": 16}},
            "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
            "7": {"class_type": "Krea2EditModelPatch", "inputs": {"model": ["1b", 0], "source_latent": ["6", 0], "vae": ["3", 0], "source_image": ["5", 0],
                                                                  "target_latent": ["11", 0],   # VAE-encode before sampling (otherwise it happens mid-run and is slower)
                                                                  "ref_boost": 1, "ref_boost_a": 1, "fit_mode": "fit"}},
            "8": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": ["2", 0], "prompt": prompt, "image": ["5", 0], "grounding_px": 768, "system_prompt": ""},
                  "_meta": {"title": "RP prompt"}},
            "9": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": ["2", 0], "prompt": "", "image": ["5", 0], "grounding_px": 768, "system_prompt": ""}},
            "11": {"class_type": "EmptyLatentImage", "inputs": {"width": tam[0], "height": tam[1], "batch_size": 1}},
            "12": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["11", 0], "seed": semilla,
                                                         "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
            "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "renderpasses/krea2"}, "_meta": {"title": "RP output"}}}


def grafo_krea2v2(prompt, imagen, tam, semilla):
    """Krea 2 Turbo + "General Edit V2" r128 LoRA (by conradlocke, made for editing: remove, recolour, relight…). Recipe from its README:
    8 steps, cfg 1, ref_boost 4, grounding 768, "aligned" reference; needs comfyui-krea2edit >= 2.0.0. Best relight with the composition
    intact in our tests. The Krea 2 DEPTH LoRA does not work with the edit patch (it invents content), so it is not used here."""
    G = grafo_krea2(prompt, imagen, tam, semilla)
    G["1b"]["inputs"]["lora_name"] = "krea2_general_edit_v2_r128.safetensors"
    G["7"]["inputs"].update(ref_boost=4, ref_position="aligned (trained)", ref_position_a="aligned (trained)")
    return G


# ───────── Create: plates from a prompt with Krea 2 (text to image) ─────────
# The official ComfyUI template image_krea2_turbo_t2i (Comfy-Org/Krea-2): Turbo, qwen3vl_4b text encoder (type krea2), qwen_image VAE,
# empty negative, 8 steps · euler · simple · cfg 1. Without the template's LoRA and prompt expander, so the plate follows the prompt.
# Any other Krea 2 checkpoint in diffusion_models (a fine-tune) is used if the Turbo is not installed, with 12 steps.
KREA2_T2I_UNET = ("krea2_turbo_int8_convrot.safetensors", "krea2_turbo_fp8_scaled.safetensors", "krea2_turbo_bf16.safetensors")
KREA2_T2I_CLIP = ("qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_4b_int8_convrot.safetensors", "qwen3vl_4b_bf16.safetensors")
KREA2_VAE = "qwen_image_vae.safetensors"
FORMATOS_CREAR = {"16:9": (1920, 1088), "21:9": (2304, 992), "3:2": (1792, 1200), "1:1": (1440, 1440), "9:16": (1088, 1920)}


def modelos_crear(listar=None):
    """→ (unet, text encoder, steps) for Create, or raises ValueError naming what is missing. The official Turbo first; otherwise any
    file with "krea2" / "krea-2" in its name in diffusion_models (a fine-tune of Krea 2, 12 steps)."""
    listar = listar or comun.lista_modelos
    dm, te, va = listar("diffusion_models"), listar("text_encoders"), listar("vae")
    unet = next((comun.nombre_listado(n, dm) for n in KREA2_T2I_UNET if comun.nombre_listado(n, dm)), None)
    pasos = 8
    if not unet:
        unet = next((x for x in dm if re.search(r"krea[-_ ]?2", os.path.basename(x), re.I) and "edit" not in x.lower()), None)
        pasos = 12
    clip = next((comun.nombre_listado(n, te) for n in KREA2_T2I_CLIP if comun.nombre_listado(n, te)), None)
    faltan = [n for n, ok in (("diffusion_models/" + KREA2_T2I_UNET[1], unet), ("text_encoders/" + KREA2_T2I_CLIP[0], clip),
                              ("vae/" + KREA2_VAE, comun.nombre_listado(KREA2_VAE, va))) if not ok]
    if faltan:
        raise ValueError("Create needs Krea 2 (huggingface.co/Comfy-Org/Krea-2): " + ", ".join(faltan))
    return unet, clip, pasos


def grafo_crear(prompt, tam, semilla, prefijo, unet=KREA2_T2I_UNET[1], clip=KREA2_T2I_CLIP[0], pasos=8, n=1):
    """The n plates are ONE batch sampled by RenderPassesBatchPreviewKSampler (ComfyUI's KSampler with a live preview of every image of
    the batch in a grid), so the Create panel shows all of them advancing at once."""
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": KREA2_VAE}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}, "_meta": {"title": "RP prompt"}},
            "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": tam[0], "height": tam[1], "batch_size": n}},
            "7": {"class_type": "RenderPassesBatchPreviewKSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": semilla,
                                                        "steps": pasos, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefijo}, "_meta": {"title": "RP output"}}}


_CREAR = {}          # id → {"carpeta", "total", "archivos", "errores", "hecho"}


async def crear(d):
    """{prompt, n 1–4, formato, semilla?} → queues n Krea 2 plates; they are copied as they finish into input/renderpasses/subidas/<id>/,
    the same kind of folder as an upload, so the page selects them and the batch works on them like on any loaded image."""
    prompt = re.sub(r"\s+", " ", str(d.get("prompt") or "")).strip()
    if len(prompt) < 8:
        raise ValueError("describe the plate first (a sentence or two)")
    try:
        n = min(4, max(1, int(d.get("n") or 1)))
    except (TypeError, ValueError):
        n = 1
    tam = FORMATOS_CREAR.get(d.get("formato"), FORMATOS_CREAR["16:9"])
    try:
        semilla = int(d["semilla"]) if str(d.get("semilla") or "").strip() else random.randint(0, 2 ** 31)
    except (TypeError, ValueError):
        semilla = random.randint(0, 2 ** 31)
    unet, clip, pasos = await asyncio.to_thread(modelos_crear)
    ident = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_krea2"
    carpeta = os.path.join(DIR_CONF, "subidas", ident)
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "_prompt.txt"), "w", encoding="utf-8") as f:
        f.write("%s\n\nKrea 2 · %s · %d steps · %dx%d · seed %d…%d\n" % (prompt, unet, pasos, tam[0], tam[1], semilla, semilla + n - 1))
    E = _CREAR[ident] = {"carpeta": carpeta, "total": n, "archivos": [], "errores": [], "hecho": False, "modelo": os.path.basename(unet), "pasos": pasos,
                         "prompt_id": None}

    async def todas():
        try:
            sal = await _correr_imagenes(grafo_crear(prompt, tam, semilla, "renderpasses/crear/" + ident, unet, clip, pasos, n), espera=1800,
                                         al_encolar=lambda pid: E.update(prompt_id=pid))
            if not sal:
                raise ValueError("no image came out")
            for i, p in enumerate(sorted(sal)[:n], 1):
                nombre = "krea2_%s_%d.png" % (ident[:15], i)
                await asyncio.to_thread(shutil.copy2, p, os.path.join(carpeta, nombre))
                E["archivos"].append(nombre)
        except Exception as e:
            E["errores"].append(str(e)[:200])
        E["hecho"] = True
    E["tarea"] = asyncio.ensure_future(todas())      # keep a reference so the task is not collected
    return {"id": ident, "carpeta": carpeta, "total": n, "modelo": E["modelo"], "pasos": pasos, "tam": list(tam), "semilla": semilla}


@R.post("/renderpasses/crear")
async def r_crear(request):
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    try:
        return web.json_response(await crear(await request.json()))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)


@R.get("/renderpasses/crear")
async def r_crear_estado(request):
    E = _CREAR.get(request.query.get("id", ""))
    if not E:
        return web.json_response({"error": "unknown create job"}, status=404)
    return web.json_response({k: E[k] for k in ("carpeta", "total", "archivos", "errores", "hecho", "modelo", "pasos", "prompt_id")})


def grafo_moss(prompt, imagen, tam, semilla, comprobar=True):
    """The official ComfyUI 0.43 image_moss_image_edit template as-is (SFT int8 + edit LoRA 1.0, skip-layer guidance on layer 22 for the whole
    run, 30 steps euler · normal · cfg 4, TextEncodeMOSSImageEdit "Moss-Image-Edit" with image and VAE); only the image/latent size changes
    to "tam" (the template uses 1 MP). Fast, but it sometimes changes scene content: compare before using it."""
    if comprobar and "TextEncodeMOSSImageEdit" not in __import__("nodes").NODE_CLASS_MAPPINGS:
        raise ValueError("MOSS needs ComfyUI 0.43 or newer (the TextEncodeMOSSImageEdit node is missing): update ComfyUI")
    neg = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
    cod = lambda txt: {"class_type": "TextEncodeMOSSImageEdit", "inputs": {"clip": ["2", 0], "prompt": txt, "edit_format": "Moss-Image-Edit",
                                                                         "images.image_0": ["5", 0], "image_strength": 1.0, "vae": ["3", 0]}}
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "moss_image_sft_int8_convrot.safetensors", "weight_dtype": "default"}},
            "1b": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "moss_image_edit_lora_rank_64_bf16.safetensors", "strength_model": 1.0}},
            "1c": {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {"model": ["1b", 0], "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 1.0}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_w4a8_gptq.safetensors", "type": "moss_image", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae_small.safetensors"}},
            "4": {"class_type": "LoadImage", "inputs": {"image": imagen}, "_meta": {"title": "RP image"}},
            "5": {"class_type": "ImageScale", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "width": tam[0], "height": tam[1], "crop": "disabled"}},
            "8": dict(cod(prompt), _meta={"title": "RP prompt"}),
            "9": cod(neg),
            "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": tam[0], "height": tam[1], "batch_size": 1}},
            "12": {"class_type": "KSampler", "inputs": {"model": ["1c", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["11", 0], "seed": semilla,
                                                         "steps": 30, "cfg": 4.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
            "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "renderpasses/moss"}, "_meta": {"title": "RP output"}}}


MODELOS_MOGE = ("moge_3_vitg_fp16.safetensors", "moge_3_vitl_fp16.safetensors", "moge_2_vitl_normal_fp16.safetensors")   # preferred first


def _moge_para_moss(preferido=None):
    """The depth model the official MOSS template recommends is MoGe-3 (vitg); if it is missing, MoGe-2."""
    hay = comun.lista_modelos("geometry_estimation")
    for m in ([preferido] if preferido else []) + list(MODELOS_MOGE):
        if comun.nombre_listado(m, hay):
            return m
    return MODELOS_MOGE[-1]


def _mapa_color(imagen, tam):
    """The "color_map" of the Color & Depth template: the original flattened into single-colour regions (edge-preserving mean-shift + 24
    k-means colours). For MOSS the colours are only region IDs: they tell it where each thing goes. Saved to input/ next to the original."""
    import cv2
    import numpy as np
    nombre = "renderpasses/mapas_color/%s_%dx%d.png" % (os.path.splitext(os.path.basename(imagen))[0], tam[0], tam[1])
    dest = os.path.join(INPUT, nombre)
    if not os.path.isfile(dest):
        o = cv2.imread(os.path.join(INPUT, imagen))
        if o is None:
            raise ValueError("could not read %s for the colour map" % imagen)
        sm = cv2.pyrMeanShiftFiltering(cv2.resize(o, tuple(tam), interpolation=cv2.INTER_AREA), 12, 24)
        z = sm.reshape(-1, 3).astype(np.float32)
        _, lab, cen = cv2.kmeans(z, 24, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
        cm = cv2.medianBlur(cen[lab.ravel()].reshape(tam[1], tam[0], 3).astype(np.uint8), 5)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cv2.imwrite(dest, cm)
    return nombre


def grafo_moss_prof(prompt, imagen, tam, semilla, fin_control=None, moge=None, fuerza=None, creatividad=0.3, mapa=None, comprobar=True):
    """MOSS edit (sees the original) + the official "MOSS-Image Color & Depth Control" template: control LoRA with MoGe depth of the SAME
    original plus a colour map of its regions (locks framing and shapes, lets the end add detail); ER-SDE 25 steps, TSR 0.95, AuraFlow 5,
    SLG layer 24, cfg 3. Colour is restored afterwards with "colour from original"."""
    # Creativity slider (0 = more fidelity, 1 = lets it invent small details): 0 = control over the whole sampling (100 %), strength 1.3
    # and colour map; 0.3 = the default recipe (85 %, 1.3, colour map: edge score 0.71); 1 = like the free template (50 %, strength 0.7, no map)
    c = min(1.0, max(0.0, float(creatividad)))
    fin_control = 1.0 - 0.5 * c if fin_control is None else fin_control
    fuerza = (1.3 if c <= 0.3 else 1.3 - 0.6 * (c - 0.3) / 0.7) if fuerza is None else fuerza
    G = grafo_moss(prompt, imagen, tam, semilla, comprobar)
    G["1b"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "moss_image_color_depth_control.safetensors", "strength_model": 1.0}}
    G["1e"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1b", 0], "lora_name": "moss_image_edit_lora_rank_64_bf16.safetensors", "strength_model": 1.0}}
    G["16"] = {"class_type": "TemporalScoreRescaling", "inputs": {"model": ["1e", 0], "tsr_k": 0.95, "tsr_sigma": 1.0}}
    G["36"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["16", 0], "shift": 5.0}}
    G["1c"] = {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {"model": ["36", 0], "double_layers": "", "single_layers": "24", "start_percent": 0.0, "end_percent": 1.0}}
    G["m1"] = {"class_type": "LoadMoGeModel", "inputs": {"model_name": _moge_para_moss(moge)}}
    G["m2"] = {"class_type": "MoGeInference", "inputs": {"moge_model": ["m1", 0], "image": ["5", 0], "resolution_level": 9, "fov_x_degrees": 0.0, "batch_size": 1,
                                                          "force_projection": True, "apply_mask": True, "refine_steps": 3}}
    G["m3"] = {"class_type": "MoGeRender", "inputs": {"moge_geometry": ["m2", 0], "output": "depth"}}
    # With control up to 60 % MOSS moved a staircase, added a large window and shifted the pillars on an interior test (edge score 0.50; flat
    # walls give little to lock on): control up to 85 % and strength 1.3
    G["53"] = {"class_type": "MOSSImageControl", "inputs": {"positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0], "strength": fuerza, "start_percent": 0.0,
                                                             "end_percent": fin_control, "depth_map": ["m3", 0]}}
    # The template is Color & Depth, not depth only. With the region map of the original the interior test stopped gaining a window and a
    # railing: edge score 0.61 → 0.71 (same seed), and "colour from original" applies
    if c < 0.7:
        G["c1"] = {"class_type": "LoadImage", "inputs": {"image": mapa or _mapa_color(imagen, tam)}}
        G["53"]["inputs"]["color_map"] = ["c1", 0]
    G["21"] = {"class_type": "SamplerER_SDE", "inputs": {"solver_type": "ER-SDE", "max_stage": 3, "eta": 1.0, "s_noise": 1.05}}
    G["22"] = {"class_type": "BasicScheduler", "inputs": {"model": ["1c", 0], "scheduler": "simple", "steps": 25, "denoise": 1.0}}
    G["12"] = {"class_type": "SamplerCustom", "inputs": {"model": ["1c", 0], "add_noise": True, "noise_seed": semilla, "cfg": 3.0, "positive": ["53", 0],
                                                          "negative": ["53", 1], "sampler": ["21", 0], "sigmas": ["22", 0], "latent_image": ["11", 0]}}
    G["13"]["inputs"]["samples"] = ["12", 0]
    G["14"]["inputs"]["filename_prefix"] = "renderpasses/moss_prof"
    return G


# ───────── MOSS 4K / 8K in tiles (generating at 4K or 8K adds a lot of detail)
# MOSS straight at 4K crashes ComfyUI (the edit feeds the reference image as tokens: at 3840×2272 it runs out of memory). It works like a
# "tile upscale": the MOSS · depth output (2 MP) is scaled with lanczos to the requested side, split into ~1.6 × 1.1 K tiles with 192 px
# of overlap, and MOSS refines each one (edit with the tile as reference + MoGe depth of the tile + colour map of the ORIGINAL in that
# tile, control over the whole sampling, denoise 0.45); tiles are blended with ramps in the overlap. An interior at 3840: 9 tiles,
# ~7.5 min, sharp wood grain and no seams. Output: <batch>/moss_prof_4k/<stem>_00001_.png (the card's "4K ↗").
MOSS_LADOS = (4096, 8192)
SOLAPE_MOSAICO = 192


async def _mosaico_moss(lote, b, prompt, semilla, lado, original):
    import cv2
    import numpy as np
    carpeta = os.path.join(DIR_SAL, lote, "moss_prof_4k")
    os.makedirs(carpeta, exist_ok=True)
    estado_f = os.path.join(carpeta, b["stem"] + ".json")

    def guarda(**kw):
        with open(estado_f, "w", encoding="utf-8") as f:
            json.dump(kw, f, ensure_ascii=False)
    try:
        guarda(estado="waiting for the MOSS · depth output…", lado=lado)
        crudo, base = None, None
        for _ in range(1800):                                # the MOSS · depth output at 2 MP (up to 1 h in the queue)
            _seguir()
            crudo = _salida(lote, "moss_prof", b["stem"], crudo=True)
            if crudo and os.path.isfile(os.path.join(DIR_SAL, lote, "lote.json")):
                break
            await asyncio.sleep(2)
        mix = os.path.join(DIR_SAL, lote, "moss_prof_color", b["stem"] + ".png")
        for _ in range(60):                                  # and its "colour from original" (the 85 % recipe), if the blend applies
            _seguir()
            try:
                _mezclas(lote, _leer_lote(lote))
            except Exception:
                pass
            if not crudo or os.path.isfile(mix) or os.path.isfile(os.path.splitext(mix)[0] + ".desalineado"):
                break
            await asyncio.sleep(2)
        base = _salida(lote, "moss_prof", b["stem"]) if crudo else None
        if not base:
            raise ValueError("MOSS · depth produced no output")
        B = cv2.imread(base)
        O = cv2.imread(original) if original and os.path.isfile(original) else cv2.imread(os.path.join(INPUT, b["input"]))
        h0, w0 = B.shape[:2]
        W = int(lado if w0 >= h0 else round(lado * w0 / h0)) // 16 * 16
        H = int(round(W * h0 / w0) if w0 >= h0 else lado) // 16 * 16
        up = cv2.resize(B, (W, H), interpolation=cv2.INTER_LANCZOS4)
        Og = cv2.resize(O, (W, H), interpolation=cv2.INTER_AREA)
        nx, ny = max(1, int(np.ceil(W / 1600))), max(1, int(np.ceil(H / 1100)))
        cw, ch = int(np.ceil(W / nx)), int(np.ceil(H / ny))
        S = SOLAPE_MOSAICO
        trozos = []
        for j in range(ny):
            for i in range(nx):
                x0, y0 = max(0, i * cw - S), max(0, j * ch - S)
                x1, y1 = min(W, (i + 1) * cw + S), min(H, (j + 1) * ch + S)
                trozos.append((x0, y0, x0 + (x1 - x0) // 16 * 16, y0 + (y1 - y0) // 16 * 16))
        rel = "renderpasses/mosaicos/%s_%s" % (lote, b["stem"])
        os.makedirs(os.path.join(INPUT, rel), exist_ok=True)
        acc = np.zeros((H, W, 3), np.float32)
        peso = np.zeros((H, W, 1), np.float32)
        t0 = time.time()
        for k, (x0, y0, x1, y1) in enumerate(trozos):
            _seguir()
            guarda(estado="MOSS is refining tile %d of %d (%d×%d)…" % (k + 1, len(trozos), W, H), trozo=k, total=len(trozos), lado=lado)
            tn, cn = "%s/t%d.png" % (rel, k), "%s/c%d.png" % (rel, k)
            cv2.imwrite(os.path.join(INPUT, tn), up[y0:y1, x0:x1])
            sm = cv2.pyrMeanShiftFiltering(Og[y0:y1, x0:x1], 12, 24)
            z = sm.reshape(-1, 3).astype(np.float32)
            _, lab, cen = cv2.kmeans(z, 24, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
            cv2.imwrite(os.path.join(INPUT, cn), cv2.medianBlur(cen[lab.ravel()].reshape(sm.shape).astype(np.uint8), 5))
            G = grafo_moss_prof(prompt, tn, (x1 - x0, y1 - y0), semilla + k, creatividad=0.0, mapa=cn)
            G["11"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}}
            G["22"]["inputs"]["denoise"] = 0.45
            G["14"]["inputs"]["filename_prefix"] = "renderpasses/%s/moss_prof_4k/_trozos/%s_t%d" % (lote, b["stem"], k)
            sal = await _correr_imagenes(G, espera=1800)
            t = cv2.imread(sal[0]).astype(np.float32)
            th, tw = y1 - y0, x1 - x0
            if t.shape[:2] != (th, tw):
                t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            rx = np.minimum(np.minimum(np.arange(tw) + 1, tw - np.arange(tw)) / S, 1.0)
            ry = np.minimum(np.minimum(np.arange(th) + 1, th - np.arange(th)) / S, 1.0)
            if x0 == 0: rx[:S] = 1
            if x1 >= W - 15: rx[-S:] = 1
            if y0 == 0: ry[:S] = 1
            if y1 >= H - 15: ry[-S:] = 1
            w = (ry[:, None] * rx[None, :])[..., None].astype(np.float32)
            acc[y0:y1, x0:x1] += t * w
            peso[y0:y1, x0:x1] += w
        fin = np.where(peso > 0, acc / np.maximum(peso, 1e-6), up)
        cv2.imwrite(os.path.join(carpeta, b["stem"] + "_00001_.png"), np.clip(fin, 0, 255).astype(np.uint8))
        guarda(estado="listo", lado=lado, tam=[W, H], trozos=len(trozos), segundos=round(time.time() - t0))
        shutil.rmtree(os.path.join(INPUT, rel), ignore_errors=True)
    except Exception as e:
        print("[RenderPasses] bglook MOSS tiles:", e)
        guarda(estado=_estado_fallo(e))


def armar(motor, prompt, imagen, semilla, tam=None, creatividad=0.3):
    """Graph of ONE engine for ONE image in input/. Raises ValueError if something is missing."""
    ruta = os.path.join(INPUT, imagen)
    if motor == "krea2":
        return grafo_krea2(prompt, imagen, _tam(ruta, 2.0), semilla)
    if motor == "krea2v2":
        return grafo_krea2v2(prompt, imagen, _tam(ruta, 2.0), semilla)
    if motor == "moss":
        return grafo_moss(prompt, imagen, _tam(ruta, 2.0), semilla)
    if motor == "moss_prof":
        return grafo_moss_prof(prompt, imagen, _tam(ruta, 2.0), semilla, creatividad=creatividad)
    if motor == "klein3":
        km = _mod("klein_movil_srv")
        if km is None:
            raise ValueError("Klein 3 is not loaded (restart ComfyUI?)")
        return km.armar({"prompt": prompt, "refs": [imagen], "base": True, "mejorar": False, "conservar": False, "semilla": semilla})["grafo"]
    if motor == "fibo2":
        fb = _mod("fibo2_srv")
        if fb is None:
            raise ValueError("Fibo 2 is not loaded (restart ComfyUI?)")
        return fb.armar({"modo": "editar", "refs": [imagen], "prompt": prompt, "semilla": semilla, **({"tam": list(tam)} if tam else {})})["grafo"]
    raise ValueError("unknown engine: %s" % motor)


def _tam_salida(G, imagen):
    """Output size of the engine: Krea (EmptyLatentImage "11"), Klein (EmptyFlux2LatentImage "44"), Fibo ("5"; 0×0 = its trained size)."""
    for k in ("11", "44", "5"):
        i = G.get(k, {}).get("inputs", {})
        if "width" in i:
            if i["width"] and i["height"]:
                return int(i["width"]), int(i["height"])
            from PIL import Image
            with Image.open(os.path.join(INPUT, imagen)) as im:
                return _mod("fibo2_srv")._tam_nativo(*im.size)
    raise ValueError("cannot tell the output size of this graph")


RESOLUCIONES = ("original", "2k", "nativa")
LADO_BASE = 2048   # "Base at 2K": long side of the batch original


def objetivo(ruta, res):
    """Final batch size: "original" = the BG's size (multiple of 16), "2k" = long side 2048 with the BG's aspect, "nativa" = whatever each
    engine outputs (None)."""
    if res not in ("original", "2k"):
        return None
    from PIL import Image
    with Image.open(ruta) as im:
        w, h = im.size
    if res == "2k":
        k = 2048.0 / max(w, h)
        w, h = w * k, h * k
    return max(256, int(round(w / 16)) * 16), max(256, int(round(h / 16)) * 16)


def a_resolucion(G, imagen, obj, semilla, limpio=False):
    """Brings the engine output to size "obj" in the SAME job: if it is much smaller (Fibo at ~1 MP) SeedVR2 upscales it; a final lanczos
    ImageScale makes it EXACT so the original, the colour blend and the passes line up pixel for pixel."""
    guardar = next(k for k, n in G.items() if n.get("class_type") == "SaveImage")
    src = G[guardar]["inputs"]["images"]
    w, h = _tam_salida(G, imagen)
    if (w, h) == tuple(obj):
        return "already %d×%d" % obj
    nota = "%d×%d → %d×%d" % (w, h, obj[0], obj[1])
    # SeedVR2 can hallucinate too, so it is only used above ×1.5; below that a lanczos resize, which invents nothing.
    # limpio (passes) → always lanczos.
    if not limpio and max(obj) / float(max(w, h)) >= 1.5:
        ES = _mod("escalar_imagen_srv")
        if ES is None or ES.faltan_modelos():
            raise ValueError("SeedVR2 is missing (needed to upscale to %d×%d)" % obj)
        ES.encadenar(G, src, w, h, "l%d" % max(obj), semilla, guardar, None)
        src = ["sv59", 0]
        nota += " with SeedVR2"
    G["res"] = {"class_type": "ImageScale", "inputs": {"image": src, "upscale_method": "lanczos", "width": obj[0], "height": obj[1], "crop": "disabled"},
                "_meta": {"title": "RP batch resolution"}}
    G[guardar]["inputs"]["images"] = ["res", 0]
    return nota


def escalar_4k(G, imagen, semilla, pref, wh=None):
    """4K upscale (rp_seedvr2.encadenar: SeedVR2 3B int8, "lab" colour) AFTER the engine, in the SAME job. Adds a SECOND SaveImage (…_4k);
    the normal output is kept for the contact sheets."""
    ES = _mod("escalar_imagen_srv")
    if ES is None:
        raise ValueError("the upscaler is missing (restart ComfyUI)")
    if ES.faltan_modelos():
        raise ValueError("SeedVR2 is missing: %s" % ", ".join(ES.faltan_modelos()))
    guardar = next(k for k, n in G.items() if n.get("class_type") == "SaveImage")
    G["s4k"] = {"class_type": "SaveImage", "inputs": {"images": G[guardar]["inputs"]["images"], "filename_prefix": pref}, "_meta": {"title": "RP output 4K"}}
    w, h = wh or _tam_salida(G, imagen)   # wh: batch size if it was already brought to "Original" / "2K"
    return ES.encadenar(G, G[guardar]["inputs"]["images"], w, h, "l4096", semilla, "s4k", pref)   # cinema 4K (long side 4096), not 3840


def _prefijo(G, pref):
    for n in G.values():
        if n.get("class_type") == "SaveImage":
            n["inputs"]["filename_prefix"] = pref
    return G


def _guardar_wf(motor, G):
    """Saves the first BG's graph as a workflow to open in ComfyUI (API format: ComfyUI loads and arranges it)."""
    try:
        os.makedirs(DIR_WF, exist_ok=True)
        with open(os.path.join(DIR_WF, "RenderPasses_%s.json" % MOTORES[motor].replace(" ", "")), "w", encoding="utf-8") as f:
            json.dump(G, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _leer_lote(lote):
    if not _RX_LOTE.match(lote or ""):
        return None
    try:
        with open(os.path.join(DIR_SAL, lote, "lote.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _url(ruta, tipo):
    base = INPUT if tipo == "input" else OUTPUT
    rel = os.path.relpath(ruta, base).replace("\\", "/")
    i = rel.rfind("/")
    from urllib.parse import quote
    try:   # file mtime in the URL: colour blends are rebuilt with the same name and the browser showed the cached one
        t = "&t=%d" % int(os.path.getmtime(ruta))
    except OSError:
        t = ""
    return "/view?filename=%s&subfolder=%s&type=%s%s" % (quote(rel[i + 1:]), quote(rel[:max(0, i)]), tipo, t)


def _salida(lote, motor, stem, crudo=False):
    """An engine's output for one BG. If the batch has "colour from original", the blend (<engine>_color/<stem>.png) wins: the page, the
    sheets and the passes use it. crudo=True = the raw engine output."""
    if motor == "original":                                  # passes can also be extracted from the batch original itself
        L0 = _leer_lote(lote) or {}
        b0 = next((b for b in L0.get("bgs", []) if b["stem"] == stem), None)
        r0 = os.path.join(INPUT, b0["input"]) if b0 else None
        return r0 if r0 and os.path.isfile(r0) else None
    if not crudo:
        mix = os.path.join(DIR_SAL, lote, motor + "_color", stem + ".png")
        if os.path.isfile(mix):
            return mix
    L = sorted(glob.glob(os.path.join(DIR_SAL, lote, motor, glob.escape(stem) + "_[0-9][0-9][0-9][0-9][0-9]_.png")))
    return L[-1] if L else None


def mezcla_color(original, ia, destino, detalle=0.6):
    """"Colour from original": in Lab, LOW frequencies (colour, light, large shapes) come from the ORIGINAL and HIGH frequencies (texture,
    fine detail) are split: "detalle" from the AI output, the rest from the original.
    result = low(O) + detalle·(AI − low(AI)) + (1 − detalle)·(O − low(O)), with a blur of ~1.2 % of the long side."""
    import cv2
    import numpy as np
    F = cv2.imread(ia, cv2.IMREAD_COLOR)
    H, W = F.shape[:2]
    O = cv2.resize(cv2.imread(original, cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_AREA)
    # If the engine changed the FRAMING, the original's light/colour would land in the wrong place. When edges do not match (typical
    # correlation: Klein 0.94 · Krea 0.97 · MOSS depth 0.70 · plain MOSS 0.57) skip the blend and leave a ".desalineado" marker.
    def bordes(x):
        x = cv2.cvtColor(cv2.resize(x, (512, max(2, round(512 * H / W)))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        e = cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1)), (0, 0), 2)
        return (e - e.mean()) / (e.std() + 1e-6)
    parecido = float((bordes(O) * bordes(F)).mean())
    if parecido < 0.65:
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        with open(os.path.splitext(destino)[0] + ".desalineado", "w", encoding="utf-8") as f:
            f.write("%.3f" % parecido)
        return None
    lab = lambda x: cv2.cvtColor(x.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    Fl, Ol = lab(F), lab(O)
    s = max(2.0, 0.012 * max(W, H))
    bajo = lambda x: cv2.GaussianBlur(x, (0, 0), s)
    R = bajo(Ol) + detalle * (Fl - bajo(Fl)) + (1 - detalle) * (Ol - bajo(Ol))
    out = np.clip(cv2.cvtColor(R, cv2.COLOR_LAB2BGR), 0, 1)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    cv2.imwrite(destino, (out * 255 + 0.5).astype(np.uint8))
    return destino


RECETA_COLOR = {"moss_prof": 0.85}   # engines that always get "colour from original" (value = AI detail amount)


def _mezclas(lote, L, maximo=None):
    """Builds the missing "colour from original" blends (all of them, or up to "maximo" per call so the status poll stays fast)."""
    C = L.get("color") or {}
    # MOSS · depth ALWAYS gets "colour from original" at 85 % (its colour drifts warm). With the batch switch on, the batch value wins.
    motores = L["motores"] if C.get("on") else [m for m in L["motores"] if m in RECETA_COLOR]
    if not motores:
        return 0
    n = 0
    for m in motores:
        det = float(C.get("detalle", 0.6)) if C.get("on") else RECETA_COLOR[m]
        for b in L["bgs"]:
            dest = os.path.join(DIR_SAL, lote, m + "_color", b["stem"] + ".png")
            r = _salida(lote, m, b["stem"], crudo=True)
            if r and not os.path.isfile(dest) and not os.path.isfile(os.path.splitext(dest)[0] + ".desalineado"):
                mezcla_color(os.path.join(INPUT, b["input"]), r, dest, det)
                n += 1
                if maximo and n >= maximo:
                    return n
    return n


def _hoja(filas, destino, alto=360):
    """Contact sheet: one row per BG (original + each engine), each column labelled."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        fuente = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        fuente = ImageFont.load_default()
    celdas = []
    for fila in filas:
        c = []
        for etq, ruta in fila:
            if ruta:
                with Image.open(ruta) as im:
                    im = im.convert("RGB")
                    im = im.resize((max(1, int(im.width * alto / im.height)), alto), Image.LANCZOS)
            else:
                im = Image.new("RGB", (int(alto * 16 / 9), alto), (40, 40, 40))
            c.append((etq, im))
        celdas.append(c)
    ancho = max(sum(im.width for _, im in c) + 8 * (len(c) + 1) for c in celdas)
    H = len(celdas) * (alto + 40) + 8
    hoja = Image.new("RGB", (ancho, H), (18, 18, 20))
    d = ImageDraw.Draw(hoja)
    y = 8
    for c in celdas:
        x = 8
        for etq, im in c:
            hoja.paste(im, (x, y + 32))
            d.text((x + 4, y + 4), etq, fill=(235, 235, 235), font=fuente)
            x += im.width + 8
        y += alto + 40
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    hoja.save(destino, quality=90)
    return destino


def estado(lote):
    L = _leer_lote(lote)
    if not L:
        return None
    bgs, hechos, filas = [], 0, []
    try:
        _mezclas(lote, L, maximo=4)                # blends for outputs that just finished (a few per call)
    except Exception as e:
        print("[RenderPasses] bglook color:", e)
    mosaico = {}
    for b in L["bgs"]:
        sal, sal4k = {}, {}
        fila = [("Original", os.path.join(INPUT, b["input"]))]
        for m in L["motores"]:
            r = _salida(lote, m, b["stem"])
            con4k = L.get("escalar4k") and m != "moss_prof"
            r4 = _salida(lote, m + "_4k", b["stem"], crudo=True) if (con4k or m == "moss_prof") else None
            if r4:
                sal4k[m] = _url(r4, "output")
            if m == "moss_prof":                             # the MOSS 4K/8K is requested by hand from the card, once the 2 MP one is seen
                try:
                    with open(os.path.join(DIR_SAL, lote, "moss_prof_4k", b["stem"] + ".json"), encoding="utf-8") as f:
                        e4 = json.load(f)
                    mosaico[b["stem"]] = e4.get("estado", "") + ((" · %dK" % (e4["lado"] // 1024)) if e4.get("lado") and e4.get("estado") == "listo" else "")
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if r and (r4 or not con4k):    # with 4K the job is done when the 4K image exists (it comes last in the same graph)
                hechos += 1
                sal[m] = _url(r, "output")
            fila.append((MOTORES[m], r))
        hoja = os.path.join(DIR_SAL, lote, "comparativa", b["stem"] + ".jpg")
        if len(sal) == len(L["motores"]) and not os.path.isfile(hoja):
            _hoja([fila], hoja, alto=540)
        bgs.append({"nombre": b["nombre"], "stem": b["stem"], "original": _url(os.path.join(INPUT, b["input"]), "input"), "salidas": sal, "salidas_4k": sal4k,
                    "hoja": _url(hoja, "output") if os.path.isfile(hoja) else None, "prompt": b.get("prompt"), "prompt_es": b.get("prompt_es"), "fijado": bool(b.get("fijado")),
                    "sin_color": [m for m in L["motores"] if os.path.isfile(os.path.join(DIR_SAL, lote, m + "_color", b["stem"] + ".desalineado"))],
                    "mosaico": mosaico.get(b["stem"], "")})
        filas.append(fila)
    total = len(L["bgs"]) * len(L["motores"])
    general = os.path.join(DIR_SAL, lote, "comparativa", "_LOTE_%s.jpg" % lote)
    if hechos == total and not os.path.isfile(general):
        _hoja(filas, general, alto=300)
    return {"lote": lote, "carpeta": L["carpeta"], "prompt": L["prompt"], "motores": L["motores"], "escalar4k": bool(L.get("escalar4k")),
            "color": L.get("color") or {"on": False, "detalle": 0.6}, "total": total, "hechos": hechos, "bgs": bgs, "qwen": L.get("qwen") or "",
            "hoja": _url(general, "output") if os.path.isfile(general) else None, "salida": os.path.join(DIR_SAL, lote)}


async def lanzar(d):
    carpeta = str(d.get("carpeta") or "").strip().strip('"')
    if not carpeta or not os.path.isdir(carpeta):
        raise ValueError("folder not found: \"%s\"" % carpeta[:200])
    archivos = sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXT) and os.path.isfile(os.path.join(carpeta, f)))
    try:
        limite = int(d.get("limite") or 0)
    except (TypeError, ValueError):
        limite = 0
    elegidas = [os.path.basename(str(x)) for x in (d.get("archivos") or [])]   # files ticked in the picker
    if elegidas:
        archivos = [f for f in archivos if f in set(elegidas)]
    elif limite > 0:
        archivos = archivos[:limite]
    if not archivos:
        raise ValueError("the folder has no images (png, jpg, webp, tif)")
    motores = [m for m in (d.get("motores") or list(MOTORES)) if m in MOTORES] or list(MOTORES)
    prompt = re.sub(r"\s+", " ", str(d.get("prompt") or prompt_maestro())).strip()
    if len(prompt) < 12:
        raise ValueError("the master prompt is empty")
    escalar = bool(d.get("escalar4k"))
    base2k = d.get("base2k", True) is not False
    res = d.get("resolucion") if d.get("resolucion") in RESOLUCIONES else "original"
    fibo_directo = bool(d.get("fibo_directo"))
    try:
        creatividad = min(1.0, max(0.0, float(d.get("creatividad", 0.3))))
    except (TypeError, ValueError):
        creatividad = 0.3
    try:
        moss_lado = int(d.get("moss_lado") or 0)
    except (TypeError, ValueError):
        moss_lado = 0
    moss_lado = moss_lado if moss_lado in MOSS_LADOS else 0
    intensidad = d.get("intensidad") if d.get("intensidad") in INTENSIDAD else "media"
    # Fibo ignores the intensity sentence, so intensity also drives "colour from original": 30 % (sutil) or 60 % (media) AI detail;
    # fuerte = raw output. It can be changed later on the page without re-rendering.
    color = {"on": True, "detalle": 0.3} if intensidad == "sutil" else {"on": True, "detalle": 0.6} if intensidad == "media" else {"on": False, "detalle": 0.6}
    prompt += INTENSIDAD[intensidad]
    try:
        semilla = int(d["semilla"]) if d.get("semilla") not in (None, "") else random.randint(0, 2 ** 31)
    except (TypeError, ValueError):
        semilla = random.randint(0, 2 ** 31)
    lote = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bgs, vistos = [], set()
    from PIL import Image
    for f in archivos:
        stem = _limpio(f)
        while stem in vistos:
            stem += "_b"
        vistos.add(stem)
        dest = "bglook_%s_%s.png" % (lote, stem)
        src = os.path.join(carpeta, f)
        with Image.open(src) as im:
            w, h = im.size
            if base2k and max(w, h) > LADO_BASE:
                # "Base at 2K": large BGs are downscaled to 2K (lanczos, multiples of 16) and THAT becomes the batch original that engines,
                # sheets, colour blend and passes work on; the optional 4K upscale is a separate step at the end.
                k = LADO_BASE / float(max(w, h))
                im.convert("RGB").resize((max(16, round(w * k / 16) * 16), max(16, round(h * k / 16) * 16)), Image.LANCZOS).save(os.path.join(INPUT, dest))
            elif f.lower().endswith(".png"):
                shutil.copy2(src, os.path.join(INPUT, dest))
            else:                                         # jpg / webp / tif → png (LoadImage and the recipes expect a plain file)
                im.convert("RGB").save(os.path.join(INPUT, dest))
        bgs.append({"nombre": f, "stem": stem, "input": dest, "clave": _clave_bg(src)})
    con_qwen = bool(d.get("prompt_qwen"))
    qwen_estado = ""
    aviso_qwen = []
    if con_qwen and not _vlm_listo():                        # no local VLM: the master prompt for every plate (instead of N failed jobs)
        con_qwen = False
        aviso_qwen = ["Per-plate prompt is off: the local Qwen VLM (%s in text_encoders) is not installed, so the master prompt is used." % QWEN38]

    async def encolar(publicar=True):
        trabajos, avisos = [], []
        import aiohttp
        async with aiohttp.ClientSession() as s:
            for m in (motores if publicar else []):            # grouped by engine: one model swap per engine
                for i, b in enumerate(bgs):
                    _seguir()                                  # (only inside the per-plate prompt task) the batch was cancelled
                    obj = objetivo(os.path.join(INPUT, b["input"]), res)
                    directo = obj if (m == "fibo2" and fibo_directo) else None   # experimental: Fibo generated directly at the target size
                    try:
                        G = _prefijo(armar(m, b.get("prompt") or prompt, b["input"], semilla, tam=directo, creatividad=creatividad), "renderpasses/%s/%s/%s" % (lote, m, b["stem"]))
                        if obj:
                            nota = a_resolucion(G, b["input"], obj, semilla)
                            if i == 0:
                                avisos.append("%s: %s%s" % (MOTORES[m], "generated directly at %d×%d" % obj if directo else nota, "" if not directo or obj[0] * obj[1] <= 2.3e6 else " (Fibo was trained at ~1 MP: it may repeat details)"))
                    except ValueError as e:
                        avisos.append("%s · %s: %s" % (MOTORES[m], b["nombre"], e))
                        continue
                    if escalar and not (m == "moss_prof" and moss_lado):   # MOSS · depth with tiles: its 4K comes from MOSS, not SeedVR2
                        try:
                            escalar_4k(G, b["input"], semilla, "renderpasses/%s/%s_4k/%s" % (lote, m, b["stem"]), wh=obj)
                        except (ValueError, StopIteration) as e:
                            avisos.append("%s · %s: no 4K (%s)" % (MOTORES[m], b["nombre"], e or "no SaveImage node"))
                    if i == 0:
                        _guardar_wf(m, G)
                    async with s.post("http://127.0.0.1:%s/prompt" % comun.puerto(), json={"prompt": comun.modelos_os(G)}) as q:
                        r = await q.json()
                    if r.get("prompt_id"):
                        trabajos.append({"motor": m, "bg": b["stem"], "id": r["prompt_id"]})
                    else:
                        avisos.append("%s · %s: ComfyUI rejected the graph (%s)" % (MOTORES[m], b["nombre"], str(r.get("error") or r)[:200]))
        os.makedirs(os.path.join(DIR_SAL, lote), exist_ok=True)
        with open(os.path.join(DIR_SAL, lote, "lote.json"), "w", encoding="utf-8") as f:
            json.dump({"lote": lote, "carpeta": carpeta, "prompt": prompt, "prompt_qwen": con_qwen, "qwen": qwen_estado, "intensidad": intensidad, "color": color, "escalar4k": escalar, "resolucion": res, "fibo_directo": fibo_directo, "base2k": base2k, "creatividad": creatividad, "moss_lado": moss_lado, "semilla": semilla, "motores": motores, "bgs": bgs, "trabajos": trabajos,
                       "avisos": avisos}, f, ensure_ascii=False, indent=1)
        return trabajos, avisos

    if con_qwen:
        # the batch exists now (the page shows it as working 0/N); Qwen describes each BG and jobs are queued in the background
        os.makedirs(os.path.join(DIR_SAL, lote), exist_ok=True)
        qwen_estado = "Qwen is describing the BGs to write their prompts…"
        with open(os.path.join(DIR_SAL, lote, "lote.json"), "w", encoding="utf-8") as f:
            json.dump({"lote": lote, "carpeta": carpeta, "prompt": prompt, "prompt_qwen": True, "qwen": qwen_estado, "intensidad": intensidad, "color": color,
                       "escalar4k": escalar, "resolucion": res, "fibo_directo": fibo_directo, "base2k": base2k, "creatividad": creatividad, "moss_lado": moss_lado, "semilla": semilla, "motores": motores, "bgs": bgs,
                       "trabajos": [], "avisos": []}, f, ensure_ascii=False, indent=1)

        async def con_qwen_y_encolar():
            nonlocal qwen_estado
            sin = 0
            fijos = _fijos()
            for b in bgs:
                if lote_cancelado(lote):                     # cancelled while Qwen was describing: nothing is queued
                    qwen_estado = "cancelled"
                    await encolar(publicar=False)
                    return
                fj = fijos.get(b.get("clave") or "")
                if fj and fj.get("en"):      # a pinned prompt for this BG is used as-is
                    b["prompt"], b["prompt_es"], b["fijado"] = prompt_de_qwen(fj["en"]), fj.get("es"), True
                    if b["prompt"]:
                        continue
                try:
                    txt = await _correr_texto(grafo_prompt_bg(b["input"]))
                    b["prompt"] = prompt_de_qwen(txt)
                except Cancelado:
                    qwen_estado = "cancelled"
                    await encolar(publicar=False)
                    return
                except Exception as e:
                    b["prompt"] = None
                    print("[RenderPasses] bglook qwen:", e)
                if b.get("prompt"):     # also keep a Spanish translation of the prompt for the page
                    try:
                        tr = _mod("traductor_srv")
                        t = re.sub(r"\s+", " ", txt).strip()
                        b["prompt_es"] = (await asyncio.to_thread(tr.traducir, t, "en", "es")).get("texto") if tr else None
                    except Exception as e:
                        print("[RenderPasses] bglook traducir:", e)
                if not b["prompt"]:
                    sin += 1
            qwen_estado = "" if not sin else "%d BG(s) without a Qwen prompt: the master prompt was used" % sin
            try:
                await encolar()
            except Cancelado:
                qwen_estado = "cancelled"
                await encolar(publicar=False)
            except Exception as e:
                print("[RenderPasses] bglook encolar:", e)
        _en_fondo(lote, con_qwen_y_encolar())
        return {"lote": lote, "trabajos": 0, "bgs": len(bgs), "semilla": semilla,
                "avisos": ["Qwen describes each BG and writes its prompt; renders are queued as soon as it finishes (~20 s per BG)."]
                + ["%s: %s" % (MOTORES[m], LICENCIAS[m]) for m in motores]}
    trabajos, avisos = await encolar()
    avisos += aviso_qwen + ["%s: %s" % (MOTORES[m], LICENCIAS[m]) for m in motores]
    return {"lote": lote, "trabajos": len(trabajos), "bgs": len(bgs), "semilla": semilla, "avisos": avisos}


@R.get("/renderpasses/prompt")
async def r_prompt(request):
    return web.json_response({"prompt": prompt_maestro(), "defecto": PROMPT_DEFECTO, "motores": MOTORES, "licencias": LICENCIAS,
                              "pases_ia": {k: {"nombre": v[0], "motores": [MOTORES[m] for m in _editores(k)] + (["MOSS"] if k in MOSS_PASES else [])} for k, v in PASES_IA.items()},
                              "capacidades": capacidades()})


@R.post("/renderpasses/prompt")
async def r_prompt_guardar(request):
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    d = await request.json()
    os.makedirs(DIR_CONF, exist_ok=True)
    with open(os.path.join(DIR_CONF, "prompt_maestro.txt"), "w", encoding="utf-8") as f:
        f.write(str(d.get("prompt") or "").strip()[:6000])
    return web.json_response({"ok": True, "prompt": prompt_maestro()})


async def cancelar(lote):
    """Removes every queued job of a batch (engines and passes: their SaveImage writes to renderpasses/<batch>/) and interrupts the running one."""
    if not _RX_LOTE.match(lote or ""):
        raise ValueError("invalid batch")
    import aiohttp
    base = "http://127.0.0.1:%s" % comun.puerto()
    del_lote = lambda item: any(str(n.get("inputs", {}).get("filename_prefix", "")).startswith("renderpasses/%s/" % lote)
                                for n in (item[2] or {}).values() if isinstance(n, dict) and n.get("class_type") == "SaveImage")
    async with aiohttp.ClientSession() as s:
        async with s.get(base + "/queue") as q:
            Q = await q.json()
        pend = [it[1] for it in Q.get("queue_pending", []) if del_lote(it)]
        corre = [it[1] for it in Q.get("queue_running", []) if del_lote(it)]
        if pend:
            async with s.post(base + "/queue", json={"delete": pend}) as q:
                await q.read()
        if corre:
            async with s.post(base + "/interrupt", json={"prompt_id": corre[0]}) as q:
                await q.read()
    return {"quitados": len(pend), "cortado": len(corre)}


# ───────── Pick images from the folder ─────────
def _imagenes_de(carpeta):
    if not carpeta or not os.path.isdir(carpeta):
        raise ValueError("folder not found: \"%s\"" % str(carpeta)[:200])
    return sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXT) and os.path.isfile(os.path.join(carpeta, f)))




@R.get("/renderpasses/carpeta")
async def r_carpeta(request):
    """?ruta= → {archivos: [{nombre, kb}]} (local requests only: reads the disk)."""
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    ruta = str(request.query.get("ruta") or "").strip().strip('"')
    try:
        L = await asyncio.to_thread(_imagenes_de, ruta)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"ruta": ruta, "archivos": [{"nombre": f, "kb": os.path.getsize(os.path.join(ruta, f)) // 1024} for f in L]})


@R.get("/renderpasses/mini")
async def r_mini(request):
    """?ruta=&f=&h= → JPEG thumbnail, cached in the temp folder."""
    if not comun.es_local(request):
        return web.Response(status=403)
    ruta, f = str(request.query.get("ruta") or ""), os.path.basename(str(request.query.get("f") or ""))
    p = os.path.join(ruta, f)
    if not f.lower().endswith(EXT) or not os.path.isfile(p):
        return web.Response(status=404)
    h = max(90, min(480, int(request.query.get("h") or 180)))
    import hashlib
    cache = os.path.join(folder_paths.get_temp_directory(), "renderpasses")
    os.makedirs(cache, exist_ok=True)
    destino = os.path.join(cache, "bglook_%s.jpg" % hashlib.sha1(("%s|%d|%d" % (p, h, os.path.getmtime(p))).encode("utf-8")).hexdigest()[:16])
    if not os.path.isfile(destino):
        await asyncio.to_thread(_hacer_mini, p, destino, h)
    return web.FileResponse(destino, headers={"Cache-Control": "max-age=86400"}) if os.path.isfile(destino) else web.Response(status=404)


def _hacer_mini(origen, destino, alto):
    """JPEG thumbnail `alto` px high (16-bit and alpha images are converted to 8-bit RGB)."""
    from PIL import Image
    with Image.open(origen) as im:
        im.thumbnail((alto * 4, alto))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGBA").convert("RGB") if "A" in im.mode else im.convert("RGB")
        im.save(destino, "JPEG", quality=82)

# ───────── Upscale passes to 4K (all at once or one by one) ─────────
# IMAGE passes (layers, objects, clean plate, clay, specular, albedo…) → SeedVR2 (with alpha: colour through SeedVR2, alpha resized separately
# and snapped to the new colour edges). DATA passes (depth, normals, mattes, shading, fog, key, shadows) → clean non-AI resize (SeedVR2 would
# invent texture); depth keeps its 16 bits. Output: <BG folder>/4k/<pass>_4k.png. Progress in 4k/escalado.json.
_RX_DATOS = re.compile(r"^(depth|normals|mat_|sombreado|raw_lighting|niebla|ia_[a-z0-9]+_(key|sombras)_cpu|ia_[a-z0-9]+_sombras_v2|ia_[a-z0-9]+_mate_|objmask_)")
LADO_4K = 3840


def _ruta_de_url(u):
    from urllib.parse import urlparse, parse_qs
    q = parse_qs(urlparse(u).query)
    p = os.path.normpath(os.path.join(OUTPUT, (q.get("subfolder") or [""])[0], (q.get("filename") or [""])[0]))
    return p if comun.dentro(p, DIR_SAL) and os.path.isfile(p) else None


async def _escalar_pases(rutas, semilla=42):
    import cv2
    import numpy as np
    ES = _mod("escalar_imagen_srv")
    for p in rutas:
        _seguir()                                            # batch cancelled: the remaining passes are not upscaled
        carpeta4k = os.path.join(os.path.dirname(p), "4k")
        os.makedirs(carpeta4k, exist_ok=True)
        base = re.sub(r"_\d{5}_$", "", os.path.splitext(os.path.basename(p))[0])
        dest = os.path.join(carpeta4k, base + "_4k.png")
        try:
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            H, W = im.shape[:2]
            k = LADO_4K / max(W, H)
            W2, H2 = int(round(W * k)), int(round(H * k))
            if _RX_DATOS.match(base) or ES is None or ES.faltan_modelos():
                cv2.imwrite(dest, cv2.resize(im, (W2, H2), interpolation=cv2.INTER_LANCZOS4 if im.dtype == np.uint16 or im.ndim == 2 else cv2.INTER_CUBIC))
                continue
            entrada = "bgesc_%s.png" % re.sub(r"[^A-Za-z0-9_]", "_", os.path.relpath(p, DIR_SAL))[-90:]
            cv2.imwrite(os.path.join(INPUT, entrada), im[..., :3] if im.ndim == 3 and im.shape[2] == 4 else im)
            G = ES.grafo(entrada, round(k, 4), semilla, "renderpasses/_escalado/" + re.sub(r"[^A-Za-z0-9_]", "_", base))
            del G["50"]
            G["57"]["inputs"]["input"] = ["1", 0]                     # colour only (no JoinImageWithAlpha)
            salidas = await _correr_imagenes(G)
            if not salidas:
                continue
            C = cv2.imread(salidas[0], cv2.IMREAD_COLOR)
            C = cv2.resize(C, (W2, H2), interpolation=cv2.INTER_AREA) if C.shape[:2] != (H2, W2) else C
            if im.ndim == 3 and im.shape[2] == 4:                    # alpha: resized and snapped to the new colour edges
                a = cv2.resize(im[..., 3].astype(np.float32) / 255.0, (W2, H2), interpolation=cv2.INTER_CUBIC)
                a = _guiado(cv2.cvtColor(C, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, a, max(2, W2 // 800), 1e-4)
                C = np.dstack([C, (np.clip(a, 0, 1) * 255).astype(np.uint8)])
            cv2.imwrite(dest, C)
            os.remove(salidas[0])
        except Exception as e:
            print("[RenderPasses] bglook escalar %s: %s" % (os.path.basename(p), e))


@R.post("/renderpasses/escalar")
async def r_escalar(request):
    """{urls: [pass url]} → upscales them to 4K in the background (one, or all of a BG's passes)."""
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    d = await request.json()
    rutas = [r for r in (_ruta_de_url(u) for u in (d.get("urls") or [])) if r]
    if not rutas:
        return web.json_response({"error": "those passes were not found"}, status=400)
    lote = _lote_de(rutas[0])
    if lote:
        _reanudar(lote)
    _en_fondo(lote, _escalar_pases(rutas))
    datos = sum(1 for r in rutas if _RX_DATOS.match(os.path.basename(r)))
    return web.json_response({"trabajos": len(rutas), "seedvr2": len(rutas) - datos, "limpio": datos})


@R.post("/renderpasses/subir")
async def r_subir(request):
    """multipart "archivo" (one or more images) → saved to input/renderpasses/subidas/<date_time>/ → {carpeta, archivos}.
    Lets the user work on single images without a folder path."""
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    carpeta = os.path.join(DIR_CONF, "subidas", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(carpeta, exist_ok=True)
    guardadas = []
    lector = await request.multipart()
    while True:
        parte = await lector.next()
        if parte is None:
            break
        nombre = os.path.basename(parte.filename or "")
        if not nombre or not nombre.lower().endswith(EXT):
            continue
        nombre = re.sub(r"[^A-Za-z0-9_.()\- ]+", "_", nombre)
        with open(os.path.join(carpeta, nombre), "wb") as f:
            while True:
                trozo = await parte.read_chunk(1 << 20)
                if not trozo:
                    break
                f.write(trozo)
        guardadas.append(nombre)
    if not guardadas:
        os.rmdir(carpeta)
        return web.json_response({"error": "no image received (png, jpg, webp or tif)"}, status=400)
    return web.json_response({"carpeta": carpeta, "archivos": guardadas})


@R.post("/renderpasses/prompts_bg")
async def r_prompts_bg(request):
    """{carpeta, archivos?, limite?, rehacer?} → [{nombre, clave, en, es, fijado}]. Qwen writes each BG's prompt WITHOUT rendering, so it
    can be edited before the render (a pinned prompt is returned as-is unless "rehacer")."""
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    d = await request.json()
    carpeta = str(d.get("carpeta") or "").strip().strip('"')
    if not os.path.isdir(carpeta):
        return web.json_response({"error": "folder not found"}, status=400)
    archivos = sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXT) and os.path.isfile(os.path.join(carpeta, f)))
    elegidas = [os.path.basename(str(x)) for x in (d.get("archivos") or [])]
    try:
        limite = int(d.get("limite") or 0)
    except (TypeError, ValueError):
        limite = 0
    archivos = [f for f in archivos if f in set(elegidas)] if elegidas else (archivos[:limite] if limite > 0 else archivos)
    fijos, out = _fijos(), []
    tr = _mod("traductor_srv")
    for f in archivos[:12]:
        ruta = os.path.join(carpeta, f)
        clave = _clave_bg(ruta)
        fj = fijos.get(clave)
        if fj and not d.get("rehacer"):
            out.append({"nombre": f, "clave": clave, "en": fj.get("en"), "es": fj.get("es"), "fijado": True})
            continue
        try:
            txt = re.sub(r"\s+", " ", await _correr_texto(grafo_prompt_bg(await asyncio.to_thread(_copia_qwen, ruta)))).strip().strip('"')
            es = (await asyncio.to_thread(tr.traducir, txt, "en", "es")).get("texto") if tr else ""
            out.append({"nombre": f, "clave": clave, "en": txt, "es": es, "fijado": False})
        except Exception as e:
            out.append({"nombre": f, "clave": clave, "error": str(e)[:200]})
    return web.json_response({"bgs": out})


@R.post("/renderpasses/prompt_bg")
async def r_prompt_bg(request):
    """{clave, es?, en?, quitar?} → pins (or unpins) a BG's prompt. If only Spanish arrives, the translator turns it into English."""
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    d = await request.json()
    clave = str(d.get("clave") or "")
    if "|" not in clave:
        return web.json_response({"error": "missing BG key"}, status=400)
    fijos = _fijos()
    if d.get("quitar"):
        fijos.pop(clave, None)
        _guardar_fijos(fijos)
        return web.json_response({"ok": True, "fijado": False})
    es, en = str(d.get("es") or "").strip(), str(d.get("en") or "").strip()
    if es and not en:
        tr = _mod("traductor_srv")
        if not tr:
            return web.json_response({"error": "the translator is not loaded"}, status=400)
        en = (await asyncio.to_thread(tr.traducir, es, "es", "en")).get("texto") or ""
    t = en.lower()
    if "keep exactly" not in t or "refine only" not in t:
        return web.json_response({"error": "the prompt must have both parts: \"Keep exactly:\" and \"Refine only:\""}, status=400)
    fijos[clave] = {"en": en[:3000], "es": es[:3000], "t": time.time()}
    _guardar_fijos(fijos)
    return web.json_response({"ok": True, "fijado": True, "en": en, "es": es})


@R.post("/renderpasses/desde_input")
async def r_desde_input(request):
    """{archivo (path inside input/), carpeta?} → copies the image to a working folder (the session's one if given and inside
    subidas/; otherwise a new one) → {carpeta, nombre}."""
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    d = await request.json()
    src = comun.resolver(INPUT, str(d.get("archivo") or ""))
    if not src or not os.path.isfile(src) or not src.lower().endswith(EXT):
        return web.json_response({"error": "that image is not in input"}, status=400)
    carpeta = str(d.get("carpeta") or "")
    if not (carpeta and comun.dentro(carpeta, os.path.join(DIR_CONF, "subidas")) and os.path.isdir(carpeta)):
        carpeta = os.path.join(DIR_CONF, "subidas", datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_galeria")
        os.makedirs(carpeta, exist_ok=True)
    nombre = re.sub(r"[^A-Za-z0-9_.()\- ]+", "_", os.path.basename(src))
    shutil.copy2(src, os.path.join(carpeta, nombre))
    return web.json_response({"carpeta": carpeta, "nombre": nombre})


@R.post("/renderpasses/lanzar")
async def r_lanzar(request):
    if not comun.es_local(request):                       # reads folders from disk: local requests only
        return web.json_response({"error": "local requests only"}, status=403)
    try:
        return web.json_response(await lanzar(await request.json()))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)


@R.post("/renderpasses/color")
async def r_color(request):
    """{lote, on, detalle 0–1} → stores the option in the batch, deletes old blends and sheets and rebuilds them (in a thread: CPU, seconds)."""
    import asyncio
    d = await request.json()
    lote = str(d.get("lote") or "")
    L = _leer_lote(lote)
    if not L:
        return web.json_response({"error": "batch not found"}, status=404)
    try:
        det = max(0.0, min(1.0, float(d.get("detalle", 0.6))))
    except (TypeError, ValueError):
        det = 0.6
    L["color"] = {"on": bool(d.get("on")), "detalle": round(det, 2)}
    with open(os.path.join(DIR_SAL, lote, "lote.json"), "w", encoding="utf-8") as f:
        json.dump(L, f, ensure_ascii=False, indent=1)
    for m in L["motores"]:
        shutil.rmtree(os.path.join(DIR_SAL, lote, m + "_color"), ignore_errors=True)
    shutil.rmtree(os.path.join(DIR_SAL, lote, "comparativa"), ignore_errors=True)
    n = await asyncio.to_thread(_mezclas, lote, L)
    return web.json_response({"ok": True, "mezclas": n, "color": L["color"]})


@R.post("/renderpasses/hdr_ltx")
async def r_hdr_ltx(request):
    """{lote, motor, nombre} → that BG's colour passes to HDR EXR with LTX 2.5, one by one, in the background."""
    if _mod("ltx_efectos_srv") is None:
        return web.json_response({"error": LTX_NO}, status=501)
    d = await request.json()
    lote, motor = str(d.get("lote") or ""), str(d.get("motor") or "")
    E = estado_pases(lote, motor)
    if not E:
        return web.json_response({"error": "no passes for that batch"}, status=404)
    b = next((x for x in E["bgs"] if x["nombre"] == d.get("nombre")), None)
    if not b:
        return web.json_response({"error": "BG not found"}, status=404)
    rutas = [(k, _ruta_de_url(u)) for k, u in b["pases"].items() if not k.startswith("_") and PASES_LTX.search(k)]
    rutas = [(k, r) for k, r in rutas if r and os.path.isfile(r)]
    if not rutas:
        return web.json_response({"error": "this BG has no colour passes (specular, highlights, reflections, emission, SSS, fill, clay)"}, status=400)
    _reanudar(lote)
    _en_fondo(lote, _pases_hdr_ltx(b["carpeta"], rutas))
    return web.json_response({"ok": True, "pases": [k for k, _ in rutas], "minutos": 5 * len(rutas)})


@R.post("/renderpasses/moss_mosaico")
async def r_moss_mosaico(request):
    """{lote, stem, lado 4096|8192} → that BG's MOSS · depth at 4K or 8K in tiles, in the background (requested from the card once the
    2K result has been seen)."""
    d = await request.json()
    lote = str(d.get("lote") or "")
    L = _leer_lote(lote)
    if not L or "moss_prof" not in L.get("motores", []):
        return web.json_response({"error": "that batch has no MOSS · depth"}, status=400)
    b = next((x for x in L["bgs"] if x["stem"] == d.get("stem")), None)
    if not b or not _salida(lote, "moss_prof", b["stem"], crudo=True):
        return web.json_response({"error": "MOSS · depth has not finished for that BG yet"}, status=400)
    try:
        lado = int(d.get("lado") or 4096)
    except (TypeError, ValueError):
        lado = 4096
    if lado not in MOSS_LADOS:
        return web.json_response({"error": "lado: 4096 or 8192"}, status=400)
    f = os.path.join(DIR_SAL, lote, "moss_prof_4k", b["stem"] + ".json")
    try:
        with open(f, encoding="utf-8") as fh:
            e = json.load(fh).get("estado", "")
        if e and not _terminado(e):
            return web.json_response({"error": "already running: " + e}, status=409)
    except (OSError, ValueError):
        pass
    for v in glob.glob(os.path.join(DIR_SAL, lote, "moss_prof_4k", glob.escape(b["stem"]) + "_[0-9]*_.png")):
        os.remove(v)                                         # the one at the other size (4K ↔ 8K) is replaced
    _reanudar(lote)
    _en_fondo(lote, _mosaico_moss(lote, b, b.get("prompt") or L["prompt"], int(L.get("semilla") or 42), lado,
                                                         os.path.join(L.get("carpeta") or "", b["nombre"])))
    return web.json_response({"ok": True, "lado": lado, "minutos": 8 if lado == 4096 else 30})


@R.get("/renderpasses/estado")
async def r_estado(request):
    E = estado(request.query.get("lote", ""))
    return web.json_response(E) if E else web.json_response({"error": "batch not found"}, status=404)


def _mini_lote(j):
    """Batch thumbnail (mini.jpg, 360 px): first engine output of the first BG, or its original. Built once."""
    d = os.path.join(DIR_SAL, j["lote"])
    mini = os.path.join(d, "mini.jpg")
    if not os.path.isfile(mini):
        try:
            import cv2
            src = None
            for b in j["bgs"][:1]:
                for m in j["motores"]:
                    src = _salida(j["lote"], m, b["stem"])
                    if src:
                        break
                src = src or os.path.join(INPUT, b["input"])
            im = cv2.imread(src, cv2.IMREAD_COLOR) if src else None
            if im is None:
                return None
            h, w = im.shape[:2]
            cv2.imwrite(mini, cv2.resize(im, (360, max(1, round(h * 360 / w))), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
        except Exception:
            return None
    return _url(mini, "output")


@R.post("/renderpasses/lote_nombre")
async def r_lote_nombre(request):
    """{lote, nombre} → gives the batch a display name."""
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    d = await request.json()
    L = _leer_lote(str(d.get("lote") or ""))
    if not L:
        return web.json_response({"error": "batch not found"}, status=404)
    L["nombre"] = re.sub(r"\s+", " ", str(d.get("nombre") or "")).strip()[:80]
    with open(os.path.join(DIR_SAL, L["lote"], "lote.json"), "w", encoding="utf-8") as f:
        json.dump(L, f, ensure_ascii=False)
    return web.json_response({"ok": True, "nombre": L["nombre"]})


@R.get("/renderpasses/lotes")
async def r_lotes(request):
    L = []
    for p in sorted(glob.glob(os.path.join(DIR_SAL, "*", "lote.json")), reverse=True)[:30]:
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            pases = [m for m in j["motores"] + ["original"] if os.path.isfile(os.path.join(DIR_SAL, j["lote"], "pases_" + m, "pases.json"))]
            L.append({"lote": j["lote"], "carpeta": j["carpeta"], "bgs": len(j["bgs"]), "motores": j["motores"], "pases": pases,
                      "nombre": j.get("nombre") or "", "nombres_bg": [b["stem"] for b in j["bgs"]][:6], "ruta": os.path.join(DIR_SAL, j["lote"]),
                      "mini": _mini_lote(j)})
        except (OSError, ValueError, KeyError):
            pass
    return web.json_response(L)


# ───────────────────────── STAGE 2 · compositing passes and layers ─────────────────────────
# On the BG already improved by the chosen engine, one per BG of the batch:
#   GPU · depth (Depth Anything V2 L, relative: white = near) · normals and albedo (Marigold V2: Qwen-Image-Edit 2509 + marigold_v2_* LoRA,
#         1 step) · RGBA layers (Qwen-Image-Layered fp8, official image_qwen_image_layered template: 640 px, 20 steps, cfg 2.5; Apache-2.0).
#   CPU (when done, in /pases/estado) · shading = beauty / albedo in linear (light without material colour) · shadow matte (shading below
#         half its median, smoothed) · fog (He et al. dark channel guided by depth: white = more haze).
# NOTE: these are ESTIMATES from a 2D image, not render AOVs; key and GI are not truly separated (shading carries both).
# Output: output/renderpasses/<batch>/pases_<engine>/<bg>/ {depth, normals, albedo, capa_N (RGBA), sombreado, sombras, niebla}.png
#   POST /renderpasses/pases {lote, motor, pases?: [depth, normals, albedo, capas], capas?: 4, texto?} → {trabajos, avisos}
#   GET  /renderpasses/pases/estado?lote=&motor= → {bgs: [{nombre, beauty, pases: {nombre: url}}], hechos, total}
PASES_GPU = ("depth", "normals", "albedo", "capas", "capas_stable", "objetos", "cripto", "paralaje", "position", "moge", "hdr")   # no "materiales": Cryptomatte's CryptoMaterial layer replaces it
# Stable-Layers (Stability AI): LoRA on Qwen-Image-Layered (StabilityLabs/Stable-Layers, PEFT converted to ComfyUI keys with alpha 64/r32,
# in models/loras/StableLayers). The author's recipe is STRICT (anything else garbles): Heun · 50 steps · cfg 1 · 640 px · 4 layers · the fixed
# training text. Stability AI Community licence: free for tests and companies < US$1M/year; above that, enterprise licence.
LORA_STABLE = "StableLayers/stable_layers_comfy.safetensors"   # comun.modelos_os turns it into the exact string ComfyUI lists on this OS
TEXTO_STABLE = "a clean, well composed image"
# Material mattes: SAM3 with text prompts per material, all matches merged. A single term per material missed most instances (one rock, one
# plant), so each material uses SEVERAL terms, threshold 0.2, masks merged.
MATERIALES = {"cielo": ("Sky", ["sky", "clouds"]), "agua": ("Water", ["sea water", "ocean waves"]), "arena": ("Sand", ["sand", "beach"]),
              "piedra": ("Rock", ["rock", "stone path", "cobblestones", "boulder"]),
              "madera": ("Wood", ["wooden chest", "wooden plank", "wooden sign", "wood"])}
SAM3_CKPT = "sam3.1_multiplex_fp16.safetensors"


# SAM3 ALWAYS returns the closest match even when the material is absent (sand = the whole sky…). So Qwen3.8 first says which materials are
# present and SAM3 runs only for those. The list is saved to materiales.json in the passes folder.
PEDIDO_MATERIALES = (
    "Look at this image. Which of these materials are clearly visible in it as real surfaces: sky; water (sea, ocean, river, lake); sand "
    "(beach, dunes); rock or stone (rocks, boulders, stone paths, cliffs); wood (wooden objects, planks, chests, tree trunks)? Answer ONLY with "
    "the matching words from this list, separated by commas: sky, water, sand, rock, wood. If none of them is visible, answer: none.")
_PALABRA_MATERIAL = {"sky": "cielo", "water": "agua", "sand": "arena", "rock": "piedra", "stone": "piedra", "wood": "madera"}


# ───────── Per-BG prompt written by Qwen ─────────
# A GENERIC prompt either barely improves or invents content; one that NAMES what is there works well, but a fixed one invents what it names
# on other BGs. So Qwen3.8 looks at each BG and writes what to keep and what to polish, using only things that are VISIBLE.
PROMPT_BASE_QWEN = ("Improve only the rendering quality of this exact image, as the final frame of a high-end animated feature film, in its own "
                    "existing art style. Nothing is added, removed or replaced: the same objects, the same background, the same sky, the same time "
                    "of day, the same framing. Every area keeps its exact shape, material and amount of detail; plain or smooth areas stay plain "
                    "and smooth, keeping only the marks they already have. Keep the same lighting direction, light sources, brightness and the "
                    "exact hue of every colour.")
PEDIDO_PROMPT_BG = (
    "You are preparing an instruction for an image-editing model that must polish this exact frame of an animated film WITHOUT changing its "
    "content. Look carefully at the image and write exactly two sentences in English, nothing else.\n"
    "Sentence 1 starts with 'Keep exactly:' and lists what is visible so that it stays the same: the main subjects or objects with their "
    "colours and positions, the ground or floor and whether it is smooth or textured, the background and sky elements (clouds, mist, "
    "vegetation, buildings…) and where they are, the light sources, the light direction, the light colours and the time of day, and the art "
    "style. Mention ONLY things you can actually see.\n"
    "Sentence 2 starts with 'Refine only:' and gives 4 to 6 concrete finishing touches of a high-end final render, each one about something "
    "you listed in sentence 1 (for example: more refined painted texture on a named surface that already has texture, crisp bevelled edges "
    "and a thin rim light on a named object, soft ambient occlusion where two named things touch, a gentle glow or bloom on a named light "
    "source, a faint soft reflection of a named light on a named surface that is already smooth or shiny, deeper shadows and a richer "
    "cinematic colour grade). Never mention any object, material, light or effect that is not already in the image. Never ask for texture, "
    "detail, cracks, tiles or pattern on a surface that is smooth, flat or plain: such a surface stays smooth and may only get a faint sheen "
    "or a faint reflection of a visible light. Name the light colours exactly as they look. If a light source is visible and there is a "
    "smooth or shiny surface near it, always include, as one of the touches, a soft warm reflection of that light, in its own colour, on that "
    "surface.")   # the warm-reflection touch is explicit because Qwen otherwise asks only for "a sheen"; texture on smooth surfaces made Fibo add tiles


def grafo_prompt_bg(img):
    return {"i": {"class_type": "LoadImage", "inputs": {"image": img}},
            "e": {"class_type": "ImageScaleToMaxDimension", "inputs": {"image": ["i", 0], "upscale_method": "lanczos", "largest_size": 1024}},
            "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN38, "type": "stable_diffusion", "device": "default"}},
            "g": {"class_type": "TextGenerate", "inputs": {"clip": ["c", 0], "prompt": PEDIDO_PROMPT_BG, "image": ["e", 0], "max_length": 420,
                                                          "sampling_mode": "off", "thinking": False, "use_default_template": True, "mtp": "auto", "system_prompt": ""}},
            "p": {"class_type": "PreviewAny", "inputs": {"source": ["g", 0]}}}


# A touch more surface detail without exaggerating: fine painted GRAIN and soft satin reflections, never new shapes.
DETALLE_FINO = ("Add the fine surface detail of a high-end render without exaggerating it: a delicate, subtle painted grain and light mottling "
                "on the surfaces that already exist, and soft, blurred, satin reflections that keep the exact hue of their light; no new shapes, "
                "cracks, tiles, stones or patterns.")


def _clave_bg(ruta):
    return "%s|%d" % (os.path.basename(ruta), os.path.getsize(ruta))


def _fijos():
    try:
        with open(os.path.join(DIR_CONF, "prompts_bg.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _guardar_fijos(d):
    os.makedirs(DIR_CONF, exist_ok=True)
    with open(os.path.join(DIR_CONF, "prompts_bg.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def _copia_qwen(ruta):
    """Image Qwen looks at for per-BG prompts: the same 2K base that Render uses (so it describes what the engine will see)."""
    from PIL import Image
    import hashlib
    nombre = "bglook_q_%s.png" % hashlib.md5(_clave_bg(ruta).encode()).hexdigest()[:16]
    destino = os.path.join(INPUT, nombre)
    if not os.path.isfile(destino):
        with Image.open(ruta) as im:
            w, h = im.size
            k = min(1.0, LADO_BASE / float(max(w, h)))
            im.convert("RGB").resize((max(16, round(w * k / 16) * 16), max(16, round(h * k / 16) * 16)), Image.LANCZOS).save(destino)
    return nombre


def prompt_de_qwen(txt):
    """Builds the final prompt from Qwen's text; returns None if the two sentences are missing (the master prompt is used)."""
    t = re.sub(r"\s+", " ", str(txt or "")).strip().strip('"')
    if "keep exactly" not in t.lower() or "refine only" not in t.lower() or len(t) < 80:
        return None
    return PROMPT_BASE_QWEN + " " + t[:2500] + " " + DETALLE_FINO + " Do not add anything that is not already in the image."


async def _seguir_o_quitar(s, base, pid):
    """_seguir() for a task waiting on job `pid`: if its batch was cancelled, the job is also removed from the queue (some jobs, like the
    4K upscale, do not carry the batch name the cancel route looks for)."""
    try:
        _seguir()
    except Cancelado:
        try:
            async with s.post(base + "/queue", json={"delete": [pid]}) as q:
                await q.read()
        except Exception:
            pass
        raise


async def _correr_texto(G, espera=900):
    """Queues a graph with a text PreviewAny and waits for its result (the queue may be busy: up to 15 min)."""
    import asyncio
    import aiohttp
    base = "http://127.0.0.1:%s" % comun.puerto()
    _seguir()                                                # a cancelled batch queues nothing more
    async with aiohttp.ClientSession() as s:
        async with s.post(base + "/prompt", json={"prompt": comun.modelos_os(G)}) as q:
            pid = (await q.json()).get("prompt_id")
        if not pid:
            raise ValueError("ComfyUI rejected the Qwen graph")
        async for _ in _turnos(s, base, pid, espera):
            await _seguir_o_quitar(s, base, pid)
            async with s.get(base + "/history/" + pid) as q:
                h = (await q.json()).get(pid)
            if h and (h.get("status") or {}).get("completed"):
                for o in (h.get("outputs") or {}).values():
                    if "text" in o:
                        t = o["text"]
                        return " ".join(t) if isinstance(t, list) else str(t)
                return ""
            if h and (h.get("status") or {}).get("status_str") == "error":
                raise ValueError("Qwen failed")
    raise ValueError("Qwen took too long")


# ───────── Per-object layers ─────────
# 1. Qwen3.8 lists what a compositor would separate (characters, creatures, props, large foreground objects).
# 2. SAM3 cuts EACH one on the full-res beauty → RGBA with the beauty's exact pixels (decontaminated edge).
# 3. Klein 3 Base fills the background behind them all (WITHOUT naming the object — naming it makes it redraw it); only the zone is
#    replaced → fondo_limpio.png. No "ghosts" as with Qwen Layered.
# Outputs obj_N.png (RGBA), fondo_limpio.png and objetos.json [{n, nombre, archivo}].
PEDIDO_OBJETOS = (
    "List the separate foreground elements of this image that a compositor would cut out as their own layers: characters, creatures, "
    "animals, props and large foreground objects. Do not list the ground, sky, water, sand, rocks of the terrain or background scenery. "
    "Answer ONLY with short English noun phrases separated by commas, at most 6, for example: red crab, wooden chest, small boy. "
    "If there is none, answer: none.")
# A grey zone with the object's exact silhouette makes Klein redraw a similar object. So the zone is grown into a SHAPELESS blob and
# pre-filled with surrounding colours (OpenCV inpaint): Klein only repaints a blurry background area.
PEDIDO_RELLENO = ("Repaint the blurry, smeared area of this image as the natural background that continues around it — the same ground, "
                  "sand, water, sky, plants and scenery, with the same lighting, perspective and art style. It must be empty background: no "
                  "creature, character, animal or object there. Keep everything outside the blurry area exactly the same.")


async def _turnos(s, base, pid, espera, espera_cola=6 * 3600):
    """Ticks every 2 s while waiting for a queued job. The `espera` limit only counts while the job is running: on a shared, busy
    queue a job can wait its turn for a long time (up to `espera_cola`) without that being a failure."""
    corriendo, t0, n = 0, time.time(), 0
    while corriendo < espera and time.time() - t0 < espera_cola:
        await asyncio.sleep(2)
        n += 1
        if n % 5 == 0:                                       # every ~10 s: is it still waiting in the queue?
            try:
                async with s.get(base + "/queue") as q:
                    cola = await q.json()
                if not any(it[1] == pid for it in cola.get("queue_pending") or []):
                    corriendo += 10
            except Exception:
                corriendo += 10
        yield


async def _correr_imagenes(G, espera=1200, espera_cola=6 * 3600, al_encolar=None):
    """Queues a graph and waits for its output images → [paths in output] (see _turnos for the time limits)."""
    import aiohttp
    base = "http://127.0.0.1:%s" % comun.puerto()
    _seguir()                                                # a cancelled batch queues nothing more
    async with aiohttp.ClientSession() as s:
        async with s.post(base + "/prompt", json={"prompt": comun.modelos_os(G)}) as q:
            pid = (await q.json()).get("prompt_id")
        if not pid:
            raise ValueError("ComfyUI rejected the graph")
        if al_encolar:
            al_encolar(pid)
        async for _ in _turnos(s, base, pid, espera, espera_cola):
            await _seguir_o_quitar(s, base, pid)
            async with s.get(base + "/history/" + pid) as q:
                h = (await q.json()).get(pid)
            if h and (h.get("status") or {}).get("completed"):
                return [os.path.join(OUTPUT, im.get("subfolder") or "", im["filename"]) for o in (h.get("outputs") or {}).values()
                        for im in o.get("images") or [] if im.get("type", "output") == "output"]
            if h and (h.get("status") or {}).get("status_str") == "error":
                raise ValueError("the graph failed")
    raise ValueError("it took too long")


async def _capas_objetos(img, pref, carpeta, beauty, semilla=42):
    import cv2
    import numpy as np
    os.makedirs(carpeta, exist_ok=True)
    estado_f = os.path.join(carpeta, "objetos.json")
    def guarda(**kw):
        with open(estado_f, "w", encoding="utf-8") as f:
            json.dump(kw, f, ensure_ascii=False)
    guarda(estado="Qwen is looking for what to separate…", objetos=[])
    try:
        G = {"i": {"class_type": "LoadImage", "inputs": {"image": img}},
             "e": {"class_type": "ImageScaleToMaxDimension", "inputs": {"image": ["i", 0], "upscale_method": "lanczos", "largest_size": 768}},
             "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN38, "type": "stable_diffusion", "device": "default"}},
             "g": {"class_type": "TextGenerate", "inputs": {"clip": ["c", 0], "prompt": PEDIDO_OBJETOS, "image": ["e", 0], "max_length": 80,
                                                           "sampling_mode": "off", "thinking": False, "use_default_template": True, "mtp": "auto", "system_prompt": ""}},
             "p": {"class_type": "PreviewAny", "inputs": {"source": ["g", 0]}}}
        txt = (await _correr_texto(G)).strip().strip(".")
        nombres = [] if re.fullmatch(r"\s*none\s*", txt, re.I) else [re.sub(r"[^a-zA-Z \-]", "", x).strip() for x in txt.split(",")]
        nombres = [x for x in nombres if 2 <= len(x) <= 40][:6]
        B = cv2.imread(beauty, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        H, W = B.shape[:2]
        guia = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
        objetos, union = [], np.zeros((H, W), np.float32)
        for n, nombre in enumerate(nombres, 1):
            guarda(estado="SAM3 is cutting out \"%s\" (%d of %d)…" % (nombre, n, len(nombres)), objetos=objetos, qwen=txt)
            rutas = await _correr_imagenes(grafo_sam3(img, pref + "objmask_%d" % n, nombre, umbral=0.3))
            if not rutas:
                continue
            m = cv2.resize(cv2.imread(rutas[0], cv2.IMREAD_GRAYSCALE), (W, H)).astype(np.float32) / 255.0
            if m.mean() < 0.002 or m.mean() > 0.85:          # nothing, or "everything" (SAM3 did not really find it)
                continue
            a = afinar_borde(B, m)
            rgb = decontaminar(B, a)
            dest = os.path.join(carpeta, "obj_%d.png" % n)
            cv2.imwrite(dest, np.dstack([np.clip(rgb, 0, 1) * 255, a * 255]).astype(np.uint8))
            objetos.append({"n": n, "nombre": nombre, "archivo": os.path.basename(dest)})
            union = np.maximum(union, (a > 0.3).astype(np.float32))
        if objetos:
            guarda(estado="Klein is filling the background behind the objects…", objetos=objetos, qwen=txt)
            r = max(10, W // 30)                                  # blob without the object's silhouette: dilate a lot and close
            zona = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
            zona = cv2.morphologyEx(zona, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 1,) * 2))
            pequeña = cv2.resize((B * 255).astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            mz = cv2.resize((zona > 0.5).astype(np.uint8) * 255, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
            relleno = cv2.resize(cv2.inpaint(pequeña, mz, 9, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
            relleno = cv2.GaussianBlur(relleno, (0, 0), max(4, W // 150))   # blurred: surrounding colours, no shapes
            zs = cv2.GaussianBlur(zona, (0, 0), max(3, W // 300))[..., None]
            gris = B * (1 - zs) + relleno * zs
            nombre_in = "bgplate_%s.png" % re.sub(r"[^A-Za-z0-9_]", "_", os.path.basename(carpeta))[:80]
            cv2.imwrite(os.path.join(INPUT, nombre_in), (np.clip(gris, 0, 1) * 255).astype(np.uint8))
            km = _mod("klein_movil_srv")
            G = _prefijo(km.armar({"prompt": PEDIDO_RELLENO, "refs": [nombre_in], "base": True, "mejorar": False, "conservar": False,
                                   "semilla": semilla})["grafo"], pref + "fondo_klein")
            rutas = await _correr_imagenes(G)
            if rutas:
                K = cv2.resize(cv2.imread(rutas[0], cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
                suave = cv2.GaussianBlur(zona, (0, 0), max(2, W // 400))[..., None]   # only inside the zone, with a soft edge
                cv2.imwrite(os.path.join(carpeta, "fondo_limpio.png"), (np.clip(B * (1 - suave) + K * suave, 0, 1) * 255).astype(np.uint8))
        guarda(estado="listo", objetos=objetos, qwen=txt)
    except Exception as e:
        print("[RenderPasses] bglook objetos:", e)
        guarda(estado=_estado_fallo(e), objetos=[])


async def _quitar_klein(B, nombres, pega, pref, nombre, semilla=42):
    """Instruction edit on the SHARP image ("remove the …"), pasted only inside "pega" (soft edge). The blurry-zone fill tended to redraw a
    creature where the large one was."""
    import cv2
    import numpy as np
    H, W = B.shape[:2]
    nombre_in = "bgplate_%s.png" % re.sub(r"[^A-Za-z0-9_]", "_", nombre)[:90]
    cv2.imwrite(os.path.join(INPUT, nombre_in), (np.clip(B, 0, 1) * 255 + 0.5).astype(np.uint8))
    km = _mod("klein_movil_srv")
    G = _prefijo(km.armar({"prompt": PEDIDO_QUITAR % (" and the ".join(nombres), ", no ".join(nombres)), "refs": [nombre_in], "base": True, "mejorar": False,
                           "conservar": False, "semilla": semilla})["grafo"], pref)
    rutas = await _correr_imagenes(G)
    if not rutas:
        raise ValueError("Klein returned no background")
    K = cv2.resize(cv2.imread(rutas[0], cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
    suave = cv2.GaussianBlur(pega.astype(np.float32), (0, 0), max(2, W // 400))[..., None]
    return np.clip(B * (1 - suave) + K * suave, 0, 1)


async def _rellenar_klein(B, zona, pega, pref, nombre, semilla=42):
    """Klein 3 Base repaints a zone as empty background (per-object layers recipe: the zone is a blurry blob of surrounding colours without
    the silhouette, so it does not redraw what was there). Pasted only inside "pega" (soft edge). B is float BGR 0–1."""
    import cv2
    import numpy as np
    H, W = B.shape[:2]
    pequeña = cv2.resize((B * 255).astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    mz = cv2.resize((zona > 0.5).astype(np.uint8) * 255, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
    relleno = cv2.resize(cv2.inpaint(pequeña, mz, 9, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    relleno = cv2.GaussianBlur(relleno, (0, 0), max(4, W // 150))
    zs = cv2.GaussianBlur(zona.astype(np.float32), (0, 0), max(3, W // 300))[..., None]
    nombre_in = "bgplate_%s.png" % re.sub(r"[^A-Za-z0-9_]", "_", nombre)[:90]
    cv2.imwrite(os.path.join(INPUT, nombre_in), (np.clip(B * (1 - zs) + relleno * zs, 0, 1) * 255).astype(np.uint8))
    km = _mod("klein_movil_srv")
    G = _prefijo(km.armar({"prompt": PEDIDO_RELLENO, "refs": [nombre_in], "base": True, "mejorar": False, "conservar": False,
                           "semilla": semilla})["grafo"], pref)
    rutas = await _correr_imagenes(G)
    if not rutas:
        raise ValueError("Klein returned no fill")
    K = cv2.resize(cv2.imread(rutas[0], cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
    suave = cv2.GaussianBlur(pega.astype(np.float32), (0, 0), max(2, W // 400))[..., None]
    return np.clip(B * (1 - suave) + K * suave, 0, 1)


# ───────── Parallax ─────────
# The BG split into depth PLANES (front, mid, far, background) to move the camera in 2.5D without holes. Cuts come from depth (multi-Otsu on
# non-sky); with Cryptomatte each compact element goes WHOLE to one plane, while things that extend in depth (ground, water) are cut by depth.
# Then, front to back, Klein fills what each plane hid: each rear plane carries a margin of the hidden area (so no hole shows when the camera
# moves) and the last one is a full plate. Outputs plano_N.png (RGBA, last one opaque) and paralaje.json with each plane's median distance.
# Removal recipe: on the ORIGINAL, everything in front at once, stating explicitly that none of it remains. "including every part of them"
# is needed, otherwise Klein removes the body and leaves limbs.
PEDIDO_QUITAR = ("Remove the %s completely from this image, including every part of them (legs, claws, arms, tails, wings, eyes, fins). Show "
                 "only the empty background that was hidden behind them: the same ground, water, sky, clouds and distant scenery, continuing "
                 "naturally, with the same lighting, perspective and art style. There is no %s, no part of them anywhere in the image, and no "
                 "new creature, character, animal, tower or object.")
_AMPLIO_RX = re.compile(r"\b(sky|cloud|clouds|sand|ground|water|dust|mist|fog|sea|ocean|terrain|floor|background|grass|desert|beach)\b", re.I)
def _hecho_o_no_pedido(carpeta, archivo, pase):
    """For parallax: has that pass finished, or was it not requested in this batch (then it is not waited for)?"""
    f = os.path.join(carpeta, archivo)
    if pase == "moge":
        return os.path.isfile(f) or not _pase_pedido(carpeta, "moge")
    try:
        with open(f, encoding="utf-8") as fh:
            e = json.load(fh).get("estado", "")
        return _terminado(e)
    except (OSError, ValueError):
        return not _pase_pedido(carpeta, pase)


def _pase_pedido(carpeta, pase):
    try:
        with open(os.path.join(os.path.dirname(carpeta), "pases.json"), encoding="utf-8") as fh:
            return pase in json.load(fh).get("pases", [])
    except (OSError, ValueError):
        return False


# ───────── Beauty HDR (a real EXR): the engines return 8 bits (a blown-out light clips at 1.0). The SDR → HDR effect (LTX 2.5 IC-LoRA,
# ltx_efectos_srv module) rebuilds the range above white and writes 16-bit EXR in ACEScg. It only takes video: the beauty goes in as a still
# 9-frame clip (8k+1) and the middle frame is kept. Outputs beauty_hdr.exr (ACEScg, half) and a preview with lights > 1 marked.
# The LTX VAE in float32 ran out of memory with a 4K beauty (4096×2432 × 9 frames): LTX works with the long side at 2048 at most and the HDR
# goes back to the PNG size with _hdr_a_tamano: the detail comes from the large PNG and LTX provides the range.
# ltx_efectos_srv is NOT bundled with RenderPasses: without it every HDR-LTX path fails with LTX_NO.
LTX_LADO = 2048
LTX_NO = "the SDR → HDR effect (LTX 2.5) is not available in this installation"


def _ltx():
    """The LTX 2.5 SDR → HDR effect module, or ValueError(LTX_NO) if this installation does not have it."""
    le = _mod("ltx_efectos_srv")
    if le is None:
        raise ValueError(LTX_NO)
    return le


def _ffmpeg(le=None):
    """ffmpeg for the LTX clips on any OS: the LTX module's own if it has one, else comun.ffmpeg() (PATH or imageio-ffmpeg)."""
    try:
        ff = le._ffmpeg() if le is not None and hasattr(le, "_ffmpeg") else None
    except Exception:
        ff = None
    ff = ff or comun.ffmpeg()
    if not ff:
        raise ValueError("ffmpeg was not found: install it (on the PATH) or pip install imageio-ffmpeg")
    return ff


def _escala_ltx(png):
    from PIL import Image
    with Image.open(png) as im:
        w, h = im.size
    k = min(1.0, LTX_LADO / float(max(w, h)))
    return "scale=%d:%d" % (max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2))


def _hdr_a_tamano(exr, png):
    """The (small) LTX EXR → at the PNG size: the EXR scaled as-is (LTX values and peaks untouched) PLUS the fine detail only the large PNG
    has (PNG − PNG downscaled and upscaled again), multiplied by the local LTX gain so detail scales with the light. Tested with a beauty HDR
    downscaled by half: less error than scaling alone and the same peaks."""
    import cv2
    import numpy as np
    _, a = _leer_exr(exr)
    S = cv2.cvtColor(cv2.imread(png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = S.shape[:2]
    if a is None or (a.shape[1], a.shape[0]) == (W, H):
        return
    hdr = a[..., :3].astype(np.float32)
    h, w = hdr.shape[:2]
    M = np.array(AP1_DESDE_709, np.float32).T
    lin = lambda x: np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)
    up = lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC)
    Sg = lin(S) @ M
    Sc = cv2.resize(Sg, (w, h), interpolation=cv2.INTER_AREA)
    detalle = Sg - up(Sc)
    Y = lambda x: x @ np.array([0.2722287, 0.6740818, 0.0536895], np.float32)
    g = np.clip(up(cv2.GaussianBlur(Y(hdr), (0, 0), 1.0)) / np.maximum(up(cv2.GaussianBlur(Y(Sc), (0, 0), 1.0)), 2e-3), 0.25, 64)[..., None]
    out = np.maximum(up(hdr) + detalle * g, 0)
    _exr(exr, out.astype(np.float32))


async def _ltx_a_exr(png, destino, sub):
    """One PNG → HDR EXR (ACEScg, half) with the LTX 2.5 SDR → HDR effect: still 9-frame clip (8k+1), the middle frame is kept."""
    import subprocess
    le = _ltx()
    clip = os.path.splitext(destino)[0] + "_clip.mp4"
    r = subprocess.run([_ffmpeg(le), "-v", "error", "-y", "-loop", "1", "-i", png, "-frames:v", "9", "-r", "24", "-vf", _escala_ltx(png),
                        "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", clip], capture_output=True, text=True, errors="replace")
    if r.returncode or not os.path.isfile(clip):
        raise ValueError("ffmpeg could not build the clip: " + (r.stderr or "")[-200:])
    P = le._plantilla("ltx25_sdr_hdr_template.json")
    P["9102"]["inputs"]["file"] = le._a_input(clip)
    for f in glob.glob(os.path.join(OUTPUT, sub, "*.exr")):
        os.remove(f)
    P["9012"]["inputs"].update(output_dir=sub, filename_prefix="frame", exposure=0.0)
    P["9013"]["inputs"]["filename_prefix"] = sub + "/hlg"
    P["4852"]["inputs"]["filename_prefix"] = sub + "/preview"
    le._codec(P)
    await _correr_imagenes(P, espera=1800)
    frames = sorted(glob.glob(os.path.join(OUTPUT, sub, "*.exr")))
    if not frames:
        raise ValueError("LTX wrote no EXR")
    shutil.copyfile(frames[len(frames) // 2], destino)
    await asyncio.to_thread(_hdr_a_tamano, destino, png)
    try:
        os.remove(clip)
    except OSError:
        pass
    return destino


# The COLOUR passes also go through LTX 2.5, each one (<pass>_hdr_ltx.exr); compared with the gain method below, the LTX EXRs look better.
# LTX lifts the blacks a little and tints the shadows: check it in compositing. Occlusion and shadows are excluded (0–1 by nature); clay
# with light can exceed 1, so it is included. "HDR with LTX" button per BG; ~5 min per pass.
PASES_LTX = re.compile(r"· (Specular|Highlights|Reflections|Emission · original colour|SSS.*|Fill / GI|Clay · same light)$")


def _ltx_limpia(exr, png):
    """LTX is a video model and "fills in" black areas (on a black emission PNG it invented a whole scene; on highlights and reflections it
    drew faint walls and floor). A pass's HDR is only valid where the PNG has light: it is masked with the PNG luminance (1 % threshold,
    dilated and soft) and, if LTX does not resemble the pass there (log-luminance correlation < 0.6), it is discarded → None."""
    import cv2
    import numpy as np
    P = cv2.cvtColor(cv2.imread(png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lin = np.where(P <= 0.04045, P / 12.92, ((P + 0.055) / 1.055) ** 2.4).astype(np.float32)
    Yp = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    _, a = _leer_exr(exr)
    hdr = a[..., :3].astype(np.float32)
    if hdr.shape[:2] != Yp.shape:
        hdr = cv2.resize(hdr, (Yp.shape[1], Yp.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = np.clip(Yp / 0.01, 0, 1)
    m = cv2.GaussianBlur(cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))), (0, 0), 2.0)
    zona = Yp > 0.01
    if zona.mean() > 0.002:
        Yh = hdr @ np.array([0.2722287, 0.6740818, 0.0536895], np.float32)
        x, y = np.log(Yp[zona] + 1e-3), np.log(np.maximum(Yh[zona], 0) + 1e-3)
        c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-6 and y.std() > 1e-6 else 0.0
        if c < 0.6:
            os.remove(exr)
            return None
    _exr(exr, (hdr * m[..., None]).astype(np.float32))
    return exr


def _pase_vacio(png):
    import cv2
    a = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    return a is None or float((a > 8).mean()) < 0.0005


async def _pases_hdr_ltx(carpeta, rutas):
    estado_f = os.path.join(carpeta, "hdr_ltx.json")

    def guarda(**kw):
        with open(estado_f, "w", encoding="utf-8") as f:
            json.dump(kw, f, ensure_ascii=False)
    if _mod("ltx_efectos_srv") is None:
        guarda(estado="error: " + LTX_NO, hechos=[])
        return
    hechos = []
    for i, (k, r) in enumerate(rutas):
        guarda(estado="LTX 2.5 · %s (%d of %d)…" % (k.split(" · ", 1)[-1], i + 1, len(rutas)), hechos=hechos)
        try:
            _seguir()
            rel = os.path.relpath(carpeta, OUTPUT).replace("\\", "/")
            if _pase_vacio(r):                               # a black pass (e.g. no emission): LTX would invent a whole scene
                continue
            d_ = await _ltx_a_exr(r, os.path.splitext(r)[0] + "_hdr_ltx.exr", rel + "/hdr_ltx_tmp")
            if await asyncio.to_thread(_ltx_limpia, d_, r):
                hechos.append(k)
        except Cancelado:
            shutil.rmtree(os.path.join(carpeta, "hdr_ltx_tmp"), ignore_errors=True)
            guarda(estado="cancelled", hechos=hechos)
            return
        except Exception as e:
            print("[RenderPasses] bglook HDR LTX %s: %s" % (k, e))
    shutil.rmtree(os.path.join(carpeta, "hdr_ltx_tmp"), ignore_errors=True)
    guarda(estado="listo", hechos=hechos)


async def _beauty_hdr(beauty, carpeta, carpeta_rel):
    import subprocess
    import cv2
    import numpy as np
    os.makedirs(carpeta, exist_ok=True)
    estado_f = os.path.join(carpeta, "hdr.json")

    def guarda(**kw):
        with open(estado_f, "w", encoding="utf-8") as f:
            json.dump(kw, f, ensure_ascii=False)
    guarda(estado="Preparing the clip for LTX 2.5…")
    try:
        le = _ltx()
        ff = _ffmpeg(le)
        clip = os.path.join(carpeta, "_hdr_clip.mp4")
        r = subprocess.run([ff, "-v", "error", "-y", "-loop", "1", "-i", beauty, "-frames:v", "9", "-r", "24", "-vf", _escala_ltx(beauty),
                            "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", clip], capture_output=True, text=True, errors="replace")
        if r.returncode or not os.path.isfile(clip):
            raise ValueError("ffmpeg could not build the clip: " + (r.stderr or "")[-200:])
        entrada = le._a_input(clip)
        P = le._plantilla("ltx25_sdr_hdr_template.json")
        P["9102"]["inputs"]["file"] = entrada
        sub = carpeta_rel.rstrip("/") + "/hdr_exr"
        for f in glob.glob(os.path.join(OUTPUT, sub, "*.exr")):
            os.remove(f)
        P["9012"]["inputs"]["output_dir"] = sub
        P["9012"]["inputs"]["filename_prefix"] = "frame"
        P["9012"]["inputs"]["exposure"] = 0.0
        P["9013"]["inputs"]["filename_prefix"] = carpeta_rel.rstrip("/") + "/hdr_hlg"
        P["4852"]["inputs"]["filename_prefix"] = carpeta_rel.rstrip("/") + "/hdr_preview"
        le._codec(P)
        guarda(estado="LTX 2.5 is rebuilding the high range…")
        await _correr_imagenes(P, espera=1800)
        frames = sorted(glob.glob(os.path.join(OUTPUT, sub, "*.exr")))
        if not frames:
            raise ValueError("LTX wrote no EXR")
        medio = frames[len(frames) // 2]
        dest = os.path.join(carpeta, "beauty_hdr.exr")
        shutil.copyfile(medio, dest)
        await asyncio.to_thread(_hdr_a_tamano, dest, beauty)
        _, a = _leer_exr(dest)
        rgb = a[..., :3]
        mx = float(np.percentile(rgb.max(-1), 99.9))
        # preview: simple tone-map (Reinhard) and, in magenta, whatever exceeds 1.0 (the range the PNG did not have)
        v = rgb / (1.0 + rgb)
        v = np.clip(v / max(float(np.percentile(v, 99.5)), 1e-4), 0, 1) ** (1 / 2.2)
        sobre = rgb.max(-1) > 1.0
        v[sobre] = 0.65 * v[sobre] + 0.35 * np.array([1.0, 0.2, 0.8], np.float32)
        cv2.imwrite(os.path.join(carpeta, "beauty_hdr.png"), cv2.cvtColor((v * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        guarda(estado="listo", maximo=round(mx, 2), sobre_1=round(float(sobre.mean()) * 100, 2), espacio="ACEScg", cuadros=len(frames))
    except Exception as e:
        print("[RenderPasses] bglook hdr:", e)
        guarda(estado=_estado_fallo(e))


PLANOS_PARALAJE = 4   # front, mid, far, background (each extra plane = one more Klein fill)


async def _paralaje(img, pref, carpeta, beauty, semilla=42):
    import cv2
    import numpy as np
    import struct
    os.makedirs(carpeta, exist_ok=True)
    estado_f = os.path.join(carpeta, "paralaje.json")

    def guarda(**kw):
        with open(estado_f, "w", encoding="utf-8") as f:
            json.dump(kw, f, ensure_ascii=False)
    guarda(estado="Splitting the BG into depth planes…", planos=[])
    try:
        da = os.path.join(carpeta, "depth_afinado.png")
        # Parallax is launched together with Depth, MoGe and Cryptomatte and WAITS for them to finish (up to 25 min): it cannot work
        # without depth, and with MoGe and Cryptomatte it is much better (metres and whole elements per plane)
        for _ in range(300):
            _seguir()                                        # batch cancelled → "cancelled"
            await _asegurar_depth(carpeta)                   # Depth Anything failed or vanished → MoGe as the depth
            espera = [n for n, ok in (("Depth", os.path.isfile(da)), ("MoGe", _hecho_o_no_pedido(carpeta, "moge.json", "moge")),
                                      ("Cryptomatte", _hecho_o_no_pedido(carpeta, "cripto.json", "cripto"))) if not ok]
            if not espera:
                break
            if not os.path.isfile(da) and not _fuente_depth(carpeta) and _depth_fallido(carpeta):
                raise ValueError(_depth_fallido(carpeta))
            if not os.path.isfile(da) and _fuente_depth(carpeta):   # the refined depth is made on CPU: do not depend on someone watching the page
                try:
                    await asyncio.to_thread(derivados, carpeta, beauty)
                except Exception as e:
                    print("[RenderPasses] bglook paralaje derivados:", e)
                continue
            guarda(estado="Waiting for " + " and ".join(espera) + "…", planos=[])
            await asyncio.sleep(5)
        if not os.path.isfile(da):
            raise ValueError("depth is missing: run the Depth pass first")
        B = cv2.imread(beauty, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        H, W = B.shape[:2]
        D = np.clip(cv2.resize(cv2.imread(da, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0, (W, H), interpolation=cv2.INTER_AREA), 0, 1)
        cf = os.path.join(carpeta, "cripto_cielo.png")
        cielo = (cv2.resize(cv2.imread(cf, cv2.IMREAD_GRAYSCALE), (W, H)) > 127) if os.path.isfile(cf) else (D < 0.04)
        k = PLANOS_PARALAJE
        from skimage.filters import threshold_multiotsu
        try:
            t = np.sort(threshold_multiotsu(D[~cielo][::7], classes=k))
        except Exception:
            t = np.quantile(D[~cielo], np.linspace(0, 1, k + 1)[1:-1])
        banda = lambda x: (k - 1) - np.digitize(x, t)          # 0 = nearest
        plano = banda(D).astype(np.int32)
        cm = os.path.join(carpeta, "cryptomatte.exr")
        compactos = []                                       # (mask, plane, element name) of what goes whole to one plane
        elementos = []                                       # (mask, name) of every named element that is not "broad"
        if os.path.isfile(cm):                               # compact elements go whole to a single plane
            sp, a = _leer_exr(cm)
            if a is not None:
                ids = cv2.resize(a[..., list(sp.channelnames).index("CryptoObject00.R")], (W, H), interpolation=cv2.INTER_NEAREST)
                pre = [x.name for x in sp.extra_attribs if x.name.endswith("/name") and x.value == "CryptoObject"]
                man = json.loads(sp.get_string_attribute(pre[0][:-4] + "manifest") or "{}") if pre else {}
                por_id = {}
                for nm, hx in man.items():
                    h = int(hx, 16)
                    por_id[struct.unpack("<f", struct.pack("<I", h))[0]] = re.sub(r"\s+\d+$", "", nm)
                for v in np.unique(ids):
                    m = (ids == v) & ~cielo
                    if m.sum() < 50:
                        continue
                    nm = por_id.get(float(v))
                    if nm and not _AMPLIO_RX.search(nm):            # named elements are removed even if they span several planes
                        elementos.append((m, nm))
                    p10, p50, p90 = np.percentile(D[m], (10, 50, 90))
                    if p90 - p10 < 0.3:
                        j = int(banda(np.array([p50]))[0])
                        plano[m] = j
                        if por_id.get(float(v)) and por_id[float(v)] not in ("background",):
                            compactos.append((m, j, por_id[float(v)]))
        plano[cielo] = k - 1
        plano = cv2.medianBlur(plano.astype(np.uint8), 5).astype(np.int32)   # no stray islands
        for j in range(k - 1):                               # loose bits (< 0.4 % of the image) move to the plane behind
            n, cc, st, _ = cv2.connectedComponentsWithStats((plano == j).astype(np.uint8), connectivity=8)
            for c in range(1, n):
                if st[c, cv2.CC_STAT_AREA] < 0.004 * H * W:
                    plano[cc == c] = j + 1
        Z = 1.0 / (D / CAMARA["cerca"] + (1.0 - D) / CAMARA["lejos"])
        mz = os.path.join(carpeta, "depth_moge.exr")
        if os.path.isfile(mz):                               # with MoGe-2 the card distances are in metres
            _, zm = _leer_exr(mz)
            if zm is not None:
                Z = cv2.resize(zm[..., 0], (W, H), interpolation=cv2.INTER_NEAREST)
        margen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (W // 25) + 1,) * 2)
        borde = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(4, W // 200) + 1,) * 2)
        nombres = {3: ["front", "mid", "background"], 4: ["front", "mid", "far", "background"]}.get(k) or ["plane %d" % (i + 1) for i in range(k)]
        P, tapado, planos, quitados = B.copy(), np.zeros((H, W), bool), [], []
        for i in range(k):
            reg = plano == i
            if i < k - 1 and reg.mean() < 0.003:             # nearly empty plane: skip
                continue
            dest = os.path.join(carpeta, "plano_%d.png" % (i + 1))
            if i == k - 1:                                   # the background plate: full and opaque
                # Visible background comes from the ORIGINAL. Pixels the fill changed a lot right next to a front card belong to the object
                # (an edge the mask missed): move them to that card so no loose arc stays on the background when the camera moves.
                dif = np.abs(B - P).max(-1) > 0.15
                cerca = cv2.dilate((plano < k - 1).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(6, W // 60) + 1,) * 2)).astype(bool)
                filo = reg & dif & cerca
                if filo.any():
                    _, idx = cv2.distanceTransformWithLabels((plano >= k - 1).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                    lab_cerca = plano[plano < k - 1][np.clip(idx - 1, 0, None)]
                    for o in planos:
                        j = o["n"] - 1
                        mj = filo & (lab_cerca == j)
                        if j < k - 1 and mj.any():
                            f = os.path.join(carpeta, o["archivo"])
                            A = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                            A[mj, :3] = (B[mj] * 255 + 0.5).astype(np.uint8); A[mj, 3] = 255
                            cv2.imwrite(f, A)
                    reg = reg & ~filo
                cv2.imwrite(dest, (np.where(reg[..., None], B, P) * 255 + 0.5).astype(np.uint8))
            else:
                # Cards: hard cut on the contour, 1 px extra with antialiasing and the EXACT original colour, so with a still camera the
                # stacked cards reproduce the original (a matted edge ate object pixels and let the clean plate show through).
                al = cv2.GaussianBlur(cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 0.8)
                al = np.where(reg, 1.0, al).astype(np.float32)
                # hidden-area margin only for blurry-zone fills: if removed BY NAME, the hidden area is already empty background and would
                # leave patches of sky/ground on this card at the wrong distance
                ext = (tapado & cv2.dilate(reg.astype(np.uint8), margen).astype(bool)) if not quitados else np.zeros_like(tapado)
                al = np.maximum(al, cv2.GaussianBlur(ext.astype(np.float32), (0, 0), 2))
                rgb = np.where(ext[..., None], P, B)
                # Seam between cards: two soft edges meet and the clean plate shows through the semi-transparent strip. As in compositing,
                # this card extends a few pixels UNDER the front cards' edge, opaque and with the original colour.
                if tapado.any():
                    sol = max(3, W // 300)
                    bajo = tapado & cv2.dilate(reg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sol + 1,) * 2)).astype(bool)
                    rgb = np.where(bajo[..., None], B, rgb)
                    al = np.where(bajo, 1.0, al).astype(np.float32)
                cv2.imwrite(dest, np.dstack([np.clip(rgb, 0, 1) * 255, al * 255]).astype(np.uint8))
            planos.append({"n": i + 1, "nombre": nombres[i], "archivo": os.path.basename(dest), "z": round(float(np.median(Z[reg])) if reg.any() else float(Z.max()), 2),
                           "area": round(float(reg.mean()), 3)})
            if i < k - 1:
                guarda(estado="Klein is filling what the %s plane hides (%d of %d)…" % (nombres[i], i + 1, k - 1), planos=planos)
                hueco = cv2.dilate(reg.astype(np.uint8), borde).astype(bool)
                r = max(10, W // 30)
                zona = cv2.dilate(hueco.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
                quitar = []
                for m, nm in elementos:                      # named elements with a real part in this plane (even if they continue in others)
                    if nm not in quitar and (m & reg).sum() > max(0.15 * m.sum(), 0.002 * H * W):
                        quitar.append(nm)
                quitados += [x for x in quitar if x not in quitados]   # everything from this plane and the ones in front
                if quitados:                                 # on the ORIGINAL, all at once; pasted over everything hidden so far
                    # pasted with slack: the mask sits a few px inside the real edge and would leave a rim on the background
                    holgura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(8, W // 80) + 1,) * 2)
                    P = await _quitar_klein(B, quitados, cv2.dilate((tapado | hueco).astype(np.uint8), holgura).astype(bool), pref + "plano_klein_%d" % (i + 1),
                                            "%s_plano%d" % (os.path.basename(carpeta), i + 1), semilla)
                else:                                        # no Cryptomatte or only ground: the usual blurry zone
                    P = await _rellenar_klein(P, zona.astype(np.float32), hueco, pref + "plano_klein_%d" % (i + 1),
                                              "%s_plano%d" % (os.path.basename(carpeta), i + 1), semilla)
                planos[-1]["quitado"] = quitar
                tapado |= hueco
        guarda(estado="listo", planos=planos, cortes=[round(float(x), 3) for x in t], metros=os.path.isfile(mz))
    except Exception as e:
        print("[RenderPasses] bglook paralaje:", e)
        guarda(estado=_estado_fallo(e), planos=[])


# ───────── Cryptomatte ─────────
# Qwen lists EVERYTHING visible (characters, props and also background: sky, clouds, ground, rocks…), SAM3 cuts each element and an EXR is
# written with the Cryptomatte standard (Psyop, v1.2): CryptoObject00/01 channels = (id, coverage) pairs of the 4 elements with most
# coverage per pixel; id = MurmurHash3_32 of the name as float32; name/hash/conversion/manifest metadata. Nuke's Cryptomatte node (or the
# After Effects plugin) reads it and the eyedropper picks "volcano", "path"… Uncovered pixels go to "background".
# 2D estimate: only what Qwen names and SAM3 finds; edges come from segmentation, not geometry.
MATERIALES_CRIPTO = ("sky", "cloud", "mist", "rock", "stone", "sand", "soil", "water", "plant", "wood", "metal", "plastic", "glass", "fabric",
                     "skin", "fur", "scales", "paper", "light", "other")
# Few, WHOLE elements, as a compositor would ask for them (otherwise "lizard" and "lizard tail" came out separately).
PEDIDO_CRIPTO = ("You are preparing ID mattes for a compositor. Look at this image and list the few meaningful elements a compositor would want "
                 "to select: each character or creature as ONE whole element (never split it into body, tail, head or legs), plus its eyes as a "
                 "separate element only if they are clearly visible (e.g. iguana eyes), each prop or garment it holds or wears (e.g. plastic bag), "
                 "and the background as broad elements (sky, clouds, vegetation, ground, rocks, mountains, water, buildings). At most 10 lines, the "
                 "most important first, in this exact format: name | material. The name is one to three English words (for a group of similar "
                 "things use the plural, e.g. rocks). The material is one word from this list: " + ", ".join(MATERIALES_CRIPTO) + ". Write nothing else.")


def _murmur3_32(data, seed=0):
    c1, c2, h, n = 0xcc9e2d51, 0x1b873593, seed & 0xffffffff, len(data)
    for i in range(0, n - n % 4, 4):
        k = int.from_bytes(data[i:i + 4], "little")
        k = (k * c1) & 0xffffffff; k = ((k << 15) | (k >> 17)) & 0xffffffff; k = (k * c2) & 0xffffffff
        h ^= k; h = ((h << 13) | (h >> 19)) & 0xffffffff; h = (h * 5 + 0xe6546b64) & 0xffffffff
    t, k = data[n - n % 4:], 0
    for j, by in enumerate(t):
        k |= by << (8 * j)
    if t:
        k = (k * c1) & 0xffffffff; k = ((k << 15) | (k >> 17)) & 0xffffffff; k = (k * c2) & 0xffffffff; h ^= k
    h ^= n; h ^= h >> 16; h = (h * 0x85ebca6b) & 0xffffffff; h ^= h >> 13; h = (h * 0xc2b2ae35) & 0xffffffff; h ^= h >> 16
    return h


def _cripto_id(nombre):
    """(uint32 hash, float32 id) as in the spec: if the float exponent would be 0 or 255 one bit is flipped (no NaN/denormals)."""
    import struct
    h = _murmur3_32(nombre.encode("utf-8"))
    if ((h >> 23) & 255) in (0, 255):
        h ^= 1 << 23
    return h, struct.unpack("<f", struct.pack("<I", h))[0]


def reparto_cripto(B, M):
    """First a HARD partition of the image (each pixel to one element; on overlap the smallest wins; unclaimed pixels go to the nearest: no
    holes), THEN the soft edge between neighbours against the beauty (afinar_borde), renormalised. Smoothing each mask separately left seams.
    M: (n, H, W) masks 0..1. Returns (n, H, W) coverages summing to 1."""
    import cv2
    import numpy as np
    n, H, W = M.shape
    lab = np.full((H, W), -1, np.int32)
    for j in np.argsort(M.reshape(n, -1).sum(1)):
        lab[(M[j] > 0.5) & (lab < 0)] = j
    if (lab >= 0).any() and (lab < 0).any():
        d, idx = cv2.distanceTransformWithLabels((lab < 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        cerca = lab[lab >= 0][np.clip(idx - 1, 0, None)]
        lab = np.where(lab < 0, cerca, lab)
    S = np.stack([afinar_borde(B, (lab == j).astype(np.float32)) if (lab == j).any() else np.zeros((H, W), np.float32) for j in range(n)])
    duro = np.stack([(lab == j).astype(np.float32) for j in range(n)])
    S = S + duro * np.clip(1.0 - S.sum(0), 0, 1)[None]      # where refinement left nobody, the hard partition wins
    return S / np.maximum(S.sum(0), 1e-6)[None]


def escribir_cryptomatte(beauty, objetos, destino_exr, previas, cobertura=None):
    """objetos: [(instance name, mask 0..1, element, material)]. One partition per instance; CryptoAsset and CryptoMaterial are its sums per
    group (the three layers agree). EXR: linear beauty R G B + per layer <layer>00/01 = 4 (id, coverage) pairs + name/hash/conversion/manifest
    metadata (Cryptomatte 1.2 standard). previas: {layer: png}. cobertura: (n, H, W) coverages already split (from a PSD, rp_psd); then
    the masks are not used and no partition is made."""
    import cv2
    import numpy as np
    import OpenImageIO as oiio
    Bbgr = cv2.imread(beauty, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    B = cv2.cvtColor(Bbgr, cv2.COLOR_BGR2RGB)
    H, W = B.shape[:2]
    if cobertura is not None:
        S = np.asarray(cobertura, np.float32)
    elif objetos:
        S = reparto_cripto(Bbgr, np.stack([m for _, m, _, _ in objetos]).astype(np.float32))
    else:
        objetos, S = [("background", None, "background", "background")], np.ones((1, H, W), np.float32)
    capas = []
    for capa, clave in (("CryptoObject", 0), ("CryptoAsset", 2), ("CryptoMaterial", 3)):
        nombres = []
        for o in objetos:
            if o[clave] not in nombres:
                nombres.append(o[clave])
        C = np.stack([sum(S[i] for i, o in enumerate(objetos) if o[clave] == nm) for nm in nombres]).astype(np.float32)
        capas.append((capa, nombres, C))
    lin = np.where(B <= 0.04045, B / 12.92, ((B + 0.055) / 1.055) ** 2.4).astype(np.float32)
    planos, canales, attrs, salida = [lin[..., 0], lin[..., 1], lin[..., 2]], ["R", "G", "B"], [], {}
    for capa, nombres, M in capas:
        ids = [_cripto_id(n) for n in nombres]
        idf = np.array([f for _, f in ids], np.float32)
        k = min(4, len(nombres))
        orden = np.argsort(-M, axis=0)[:k]
        cov = np.take_along_axis(M, orden, axis=0).astype(np.float32)
        idm = np.where(cov > 0, idf[orden], 0).astype(np.float32)
        if k < 4:
            z = np.zeros((4 - k, H, W), np.float32); idm, cov = np.concatenate([idm, z]), np.concatenate([cov, z])
        for n in range(2):
            for i, c in ((2 * n, "RG"), (2 * n + 1, "BA")):
                planos += [idm[i], cov[i]]
                canales += ["%s%02d.%s" % (capa, n, c[0]), "%s%02d.%s" % (capa, n, c[1])]
        pre = "cryptomatte/%s/" % ("%08x" % _murmur3_32(capa.encode()))[:7]
        attrs += [(pre + "name", capa), (pre + "hash", "MurmurHash3_32"), (pre + "conversion", "uint32_to_float32"),
                  (pre + "manifest", json.dumps({nm: "%08x" % h for nm, (h, _) in zip(nombres, ids)}, ensure_ascii=False))]
        salida[capa] = nombres
        if capa in previas:
            col = 0.25 + 0.75 * np.array([[(h >> 0 & 255) / 255.0, (h >> 8 & 255) / 255.0, (h >> 16 & 255) / 255.0] for h, _ in ids], np.float32)
            prev = np.einsum("khw,kc->hwc", M, col)
            cv2.imwrite(previas[capa], cv2.cvtColor((np.clip(prev, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    datos = np.ascontiguousarray(np.stack(planos, -1), np.float32)
    spec = oiio.ImageSpec(W, H, len(canales), oiio.FLOAT)
    spec.channelnames = tuple(canales)
    spec.attribute("compression", "zip")
    for k, v in attrs:
        spec.attribute(k, v)
    o = oiio.ImageOutput.create(destino_exr)
    if not o or not o.open(destino_exr, spec) or not o.write_image(datos):
        raise ValueError("could not write the Cryptomatte EXR")
    o.close()
    return salida


AMPLIOS = ("sky", "cloud", "mist", "sand", "soil", "water")   # searched by text (a box would be the whole image)


def grafo_sam3_cajas(imagen, pref, cajas, W, H):
    """SAM3 cuts INSIDE each Qwen box (one instance per box). Text search kept only one instance on stylised BGs; with boxes all of them come
    out, precisely."""
    G = {"1": {"class_type": "LoadImage", "inputs": {"image": imagen}},
         "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_CKPT}}}
    for k, (x1, y1, x2, y2) in enumerate(cajas):
        G["b%d" % k] = {"class_type": "PrimitiveBoundingBox", "inputs": {"x": int(x1 / 1000 * W), "y": int(y1 / 1000 * H),
                                                                          "width": max(1, int((x2 - x1) / 1000 * W)), "height": max(1, int((y2 - y1) / 1000 * H))}}
        G["d%d" % k] = {"class_type": "SAM3_Detect", "inputs": {"model": ["2", 0], "image": ["1", 0], "bboxes": ["b%d" % k, 0], "threshold": 0.3,
                                                                 "refine_iterations": 2, "individual_masks": False}}
        G["m%d" % k] = {"class_type": "MaskToImage", "inputs": {"mask": ["d%d" % k, 0]}}
        G["s%d" % k] = {"class_type": "SaveImage", "inputs": {"images": ["m%d" % k, 0], "filename_prefix": pref + "_%02d" % k}}
    return G


def cajas_de_qwen(txt, etiquetas):
    """Parses Qwen's [{"label", "bbox_2d"}] (even if truncated) → {label: [boxes 0–1000]}. Qwen sometimes loops on a row of identical boxes
    (invented): same-size boxes repeated 3+ times, tiny boxes and more than 8 per label are dropped."""
    import collections
    por = collections.defaultdict(list)
    for m in re.finditer(r'"label"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', txt):
        l, c = m.group(1).strip().lower(), [int(m.group(i)) for i in range(2, 6)]
        if c[2] > c[0] and c[3] > c[1] and (c[2] - c[0]) * (c[3] - c[1]) >= 1500:     # >= 0.15 % of the image
            mejor = min(etiquetas, key=lambda e: 0 if e == l else (1 if (e in l or l in e) else 2))
            if mejor == l or mejor in l or l in mejor:
                por[mejor].append(c)
    for e, cs in list(por.items()):
        tam = collections.Counter((round((c[2] - c[0]) / 8), round((c[3] - c[1]) / 8)) for c in cs)
        cs = [c for c in cs if tam[(round((c[2] - c[0]) / 8), round((c[3] - c[1]) / 8))] < 3]
        unicos = []
        for c in cs:
            if all(max(abs(c[i] - u[i]) for i in range(4)) > 15 for u in unicos):
                unicos.append(c)
        por[e] = unicos[:8]
    return dict(por)


def grafo_sam3_instancias(imagen, pref, texto, umbral=0.3):
    """SAM3 with individual_masks: one mask per instance (each rock separately) → several output images."""
    G = grafo_sam3(imagen, pref, texto, umbral)
    G["d0"]["inputs"]["individual_masks"] = True
    return G


async def _cryptomatte(img, pref, carpeta, beauty, guia=""):
    """Qwen gives "name | material" per element; SAM3 extracts each INSTANCE; edges refined against the beauty; three EXR layers:
    CryptoObject (each instance: "rocks 1", "rocks 2"…), CryptoAsset (whole element) and CryptoMaterial (per material). No holes: unclaimed
    pixels go to the nearest. Also writes cripto_cielo.png (material "sky") for the shadow pass."""
    import cv2
    import numpy as np
    os.makedirs(carpeta, exist_ok=True)
    estado_f = os.path.join(carpeta, "cripto.json")

    def guarda(**kw):
        with open(estado_f, "w", encoding="utf-8") as f:
            json.dump(kw, f, ensure_ascii=False)
    guarda(estado="Qwen is naming what it sees…", elementos=[])
    # optional per-shot guide: the user lists the elements they want, so Qwen does not have to guess
    pedido = PEDIDO_CRIPTO if not guia else (
        "The compositor wrote, in Spanish, exactly which elements this shot needs as ID mattes: «%s». Return exactly those elements, in that order, "
        "one per line in this exact format: spanish name | english name | material. The spanish name is the compositor's word unchanged. The "
        "english name is two or three English words describing that thing as it looks in THIS image, so a segmentation model can find it "
        "(e.g. funda → plastic bag; a creature stays one whole element; eyes only if asked). The material is one word from this list: %s. "
        "Write nothing else." % (guia[:300], ", ".join(MATERIALES_CRIPTO)))
    try:
        G = {"i": {"class_type": "LoadImage", "inputs": {"image": img}},
             "e": {"class_type": "ImageScaleToMaxDimension", "inputs": {"image": ["i", 0], "upscale_method": "lanczos", "largest_size": 768}},
             "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN38, "type": "stable_diffusion", "device": "default"}},
             "g": {"class_type": "TextGenerate", "inputs": {"clip": ["c", 0], "prompt": pedido, "image": ["e", 0], "max_length": 260,
                                                           "sampling_mode": "off", "thinking": False, "use_default_template": True, "mtp": "auto", "system_prompt": ""}},
             "p": {"class_type": "PreviewAny", "inputs": {"source": ["g", 0]}}}
        txt = (await _correr_texto(G)).strip()
        elems, vistos = [], set()
        consulta = {}
        for linea in re.split(r"[\n;]+", txt):
            partes = [x.strip() for x in linea.split("|")]
            n = re.sub(r"[^a-zA-ZáéíóúñüÁÉÍÓÚÑ \-]", "", partes[0]).strip().lower()
            if guia and len(partes) >= 3:      # user's name | English name as seen in the image (for SAM3) | material
                consulta[n] = re.sub(r"[^a-zA-Z \-]", "", partes[1]).strip().lower() or n
                m = re.sub(r"[^a-z]", "", partes[2].lower())
            else:
                m = re.sub(r"[^a-z]", "", partes[1].lower()) if len(partes) > 1 else "other"
            if 2 <= len(n) <= 40 and n not in vistos and n != "background":
                vistos.add(n); elems.append((n, m if m in MATERIALES_CRIPTO else "other"))
        elems = elems[:10]
        # With a guide, the EXR gets the user's names and SAM3 gets Qwen's English name AS SEEN IN THE IMAGE (a plain translation picked
        # the wrong sense, and non-English queries found only part of the element).
        Bc = cv2.imread(beauty, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        H, W = Bc.shape[:2]
        with __import__("PIL").Image.open(os.path.join(INPUT, img)) as im_in:
            Wi, Hi = im_in.size
        # Qwen locates each element (one box per instance) → SAM3 cuts inside each box; broad things (sky, ground…) go by text
        en = {n: consulta.get(n, n) for n, _ in elems}
        localizar = [n for n, m in elems if m not in AMPLIOS]
        cajas = {}
        if localizar:
            guarda(estado="Qwen is locating each element…", elementos=[], qwen=txt)
            pedido_cajas = ("Locate in this image every separate instance of each of these elements: %s. Answer ONLY with a JSON list like "
                            '[{"label": "%s", "bbox_2d": [x1, y1, x2, y2]}] using exactly those labels and coordinates on a 0-1000 scale; at most 8 '
                            "boxes per label; only things that are really visible." % ("; ".join(en[n] for n in localizar), en[localizar[0]]))
            Gq = dict(G); Gq["g"] = dict(G["g"]); Gq["g"]["inputs"] = dict(G["g"]["inputs"], prompt=pedido_cajas, max_length=1200)
            Gq["e"] = dict(G["e"]); Gq["e"]["inputs"] = dict(G["e"]["inputs"], largest_size=1024)
            try:
                tq = await _correr_texto(Gq)
                porEn = cajas_de_qwen(tq, [en[n] for n in localizar])
                cajas = {n: porEn.get(en[n], []) for n in localizar}
            except Exception as e:
                print("[RenderPasses] bglook cajas qwen:", e)
        objetos = []
        for n, (nombre, mat) in enumerate(elems, 1):
            guarda(estado="SAM3 is cutting out \"%s\" (%d of %d)…" % (nombre, n, len(elems)), elementos=[], qwen=txt)
            inst = []
            cs = cajas.get(nombre) or []
            if cs:
                rutas = await _correr_imagenes(grafo_sam3_cajas(img, pref + "cripto_%d" % n, cs, Wi, Hi))
                for r, (x1, y1, x2, y2) in zip(sorted(rutas), cs):
                    m = cv2.resize(cv2.imread(r, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                    caja = max(1.0, (x2 - x1) * (y2 - y1) / 1e6 * W * H)
                    if m.mean() > 0.0004 and m.sum() / caja > 0.10:    # an almost empty box = an invented box
                        inst.append(m)
            if not inst:                                           # broad, or Qwen did not locate it: by text
                rutas = await _correr_imagenes(grafo_sam3_instancias(img, pref + "cripto_%d" % n, en.get(nombre, nombre)))
                for r in rutas:
                    m = cv2.resize(cv2.imread(r, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                    if 0.0004 < m.mean() < 0.97:
                        inst.append(m)
            inst = sorted(inst, key=lambda x: -float(x.sum()))[:8]
            for k, m in enumerate(inst, 1):
                objetos.append((nombre if len(inst) == 1 else "%s %d" % (nombre, k), m, nombre, mat))
        guarda(estado="Writing the EXR…", elementos=[], qwen=txt)
        sal = await asyncio.to_thread(escribir_cryptomatte, beauty, objetos, os.path.join(carpeta, "cryptomatte.exr"),
                                      {"CryptoObject": os.path.join(carpeta, "cryptomatte.png"), "CryptoMaterial": os.path.join(carpeta, "cryptomatte_material.png")})
        cielo = [m for (_, m, _, mt) in objetos if mt == "sky"]
        if cielo:
            cv2.imwrite(os.path.join(carpeta, "cripto_cielo.png"), (np.clip(np.max(np.stack(cielo), 0), 0, 1) * 255 + 0.5).astype(np.uint8))
        guarda(estado="listo", elementos=sal.get("CryptoAsset", []), objetos=sal.get("CryptoObject", []), materiales=sal.get("CryptoMaterial", []), qwen=txt, guia=guia)
    except Exception as e:
        print("[RenderPasses] bglook cryptomatte:", e)
        guarda(estado=_estado_fallo(e), elementos=[])


async def _materiales_con_qwen(img, pref, carpeta):
    import aiohttp
    G = {"i": {"class_type": "LoadImage", "inputs": {"image": img}},
         "e": {"class_type": "ImageScaleToMaxDimension", "inputs": {"image": ["i", 0], "upscale_method": "lanczos", "largest_size": 768}},
         "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN38, "type": "stable_diffusion", "device": "default"}},
         "g": {"class_type": "TextGenerate", "inputs": {"clip": ["c", 0], "prompt": PEDIDO_MATERIALES, "image": ["e", 0], "max_length": 60,
                                                       "sampling_mode": "off", "thinking": False, "use_default_template": True, "mtp": "auto", "system_prompt": ""}},
         "p": {"class_type": "PreviewAny", "inputs": {"source": ["g", 0]}}}
    try:
        txt = (await _correr_texto(G)).lower()
    except Cancelado:
        raise
    except Exception as e:
        print("[RenderPasses] bglook materiales:", e)
        txt = ""
    presentes = sorted({m for w, m in _PALABRA_MATERIAL.items() if re.search(r"\b%s\b" % w, txt)}, key=list(MATERIALES).index)
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "materiales.json"), "w", encoding="utf-8") as f:
        json.dump({"presentes": presentes, "qwen": txt[:300]}, f, ensure_ascii=False)
    async with aiohttp.ClientSession() as s:
        for k in presentes:
            _seguir()
            async with s.post("http://127.0.0.1:%s/prompt" % comun.puerto(), json={"prompt": comun.modelos_os(grafo_sam3(img, pref + "mat_" + k, MATERIALES[k][1]))}) as q:
                await q.read()


def grafo_sam3(imagen, pref, textos, umbral=0.2):
    """One SAM3_Detect per term (rp_fibo2.sam3_mascaras recipe) with masks merged (MaskComposite "or") → one white mask."""
    G = {"1": {"class_type": "LoadImage", "inputs": {"image": imagen}},
         "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_CKPT}}}
    previa = None
    for i, t in enumerate([textos] if isinstance(textos, str) else textos):
        G["t%d" % i] = {"class_type": "CLIPTextEncode", "inputs": {"text": t, "clip": ["2", 1]}}
        G["d%d" % i] = {"class_type": "SAM3_Detect", "inputs": {"model": ["2", 0], "image": ["1", 0], "conditioning": ["t%d" % i, 0], "threshold": umbral,
                                                             "refine_iterations": 2, "individual_masks": False}}
        if previa is None:
            previa = ["d%d" % i, 0]
        else:
            G["u%d" % i] = {"class_type": "MaskComposite", "inputs": {"destination": previa, "source": ["d%d" % i, 0], "x": 0, "y": 0, "operation": "or"}}
            previa = ["u%d" % i, 0]
    G["6"] = {"class_type": "MaskToImage", "inputs": {"mask": previa}}
    G["7"] = {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": pref}}
    return G
# No "shadow matte" from the darkest shading (it mixed shadows, cracks and dark materials); the shadow pass is "clay − fill".
PASES_CPU = ("depth_afinado", "raw_lighting", "niebla", "position", "cavidad")   # "suelo" and the light sphere are computed but not shown


def _exr(ruta, a):
    """Linear float32 EXR (zip) via OpenImageIO (the bundled OpenCV may lack an EXR writer)."""
    import OpenImageIO as oiio
    import numpy as np
    a = np.ascontiguousarray(a[..., None] if a.ndim == 2 else a, dtype=np.float32)   # a 2D array crashes OpenImageIO (access violation)
    h, w, c = a.shape
    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    spec.channelnames = ("Y",) if c == 1 else ("R", "G", "B", "A")[:c]
    spec.attribute("compression", "zip")
    o = oiio.ImageOutput.create(ruta)
    if not o or not o.open(ruta, spec) or not o.write_image(a):
        raise ValueError("could not write %s" % os.path.basename(ruta))
    o.close()
NIEBLA = (0.35, 1.0, 1.6)   # fog ramp over inverted depth: starts at 35 % of the distance, white at the back, curve 1.6
TEXTO_CAPAS = "A stylized 3D animated feature film background environment with its foreground, middle ground, background and sky."


MP_PASES = 4.2e6


def copia_pases(beauty, img):
    """Pass models (Marigold, Qwen, SAM3, Klein, Fibo) never get more than ~4 MP (very large BGs hung ComfyUI): with a bigger beauty they
    work on a downscaled copy and results are brought back to the beauty size afterwards (refined depth, raw lighting and fog on CPU; AI
    passes with lanczos). Rebuilds the copy if an older, larger one exists."""
    from PIL import Image
    destino = os.path.join(INPUT, img)
    with Image.open(beauty) as im:
        w, h = im.size
        k = min(1.0, (MP_PASES / (w * h)) ** 0.5)
        tam = (max(16, int(w * k) // 16 * 16), max(16, int(h * k) // 16 * 16)) if k < 1 else (w, h)
        if os.path.isfile(destino):
            try:
                with Image.open(destino) as c:
                    if c.size == tam:
                        return tam
            except OSError:
                pass
        if k < 1:
            im.convert("RGB").resize(tam, Image.LANCZOS).save(destino)
        else:
            shutil.copy2(beauty, destino)
    return tam


def grafo_depth(imagen, pref, lado=None):
    # Depth Anything's "resolution" is the SHORT side. Use the BG short side × 2.25 (multiple of 64, up to 2688): always >= 2.2× oversampling
    # (less gives stair-stepped edges); the CPU refinement downsamples it with antialiasing and snaps it to the beauty's colour edges.
    res = min(2688, max(1024, int(round((lado or 768) * 2.25 / 64)) * 64))
    return {"1": {"class_type": "LoadImage", "inputs": {"image": imagen}},
            "2": {"class_type": "DepthAnythingV2Preprocessor", "inputs": {"image": ["1", 0], "ckpt_name": "depth_anything_v2_vitl.pth", "resolution": res}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": pref}}}


def grafo_moge(imagen, carpeta_rel):
    """MoGe-3 or MoGe-2 (ComfyUI nodes) → RenderPassesMoGeEXR: P in metres, depth, normals and the lens, as EXR (rp_moge_node.py)."""
    return {"1": {"class_type": "LoadImage", "inputs": {"image": imagen}},
            "2": {"class_type": "LoadMoGeModel", "inputs": {"model_name": _moge_para_moss()}},   # MoGe-3 if installed, otherwise MoGe-2
            "3": {"class_type": "MoGeInference", "inputs": {"moge_model": ["2", 0], "image": ["1", 0], "resolution_level": 9, "fov_x_degrees": 0.0,
                                                             "batch_size": 1, "force_projection": True, "apply_mask": False, "refine_steps": 3}},
            "4": {"class_type": "RenderPassesMoGeEXR", "inputs": {"moge_geometry": ["3", 0], "carpeta": carpeta_rel}}}


def grafo_marigold(tipo, imagen, pref):
    """Marigold V2 for one image (official template, 1 step, sigmas 0.5 → 0), with LoadImage."""
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2509_int8_convrot.safetensors", "weight_dtype": "default"}},
            "7": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "marigold_v2_%s.safetensors" % tipo, "strength_model": 1}},
            "8": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["7", 0], "shift": 1.73, "sampling": "img_to_img_velocity"}},
            "9": {"class_type": "ConditioningLoader", "inputs": {"conditioning_name": "marigold_v2_%s_conditioning.safetensors" % tipo}},
            "14": {"class_type": "VAELoader", "inputs": {"vae_name": "marigold_v2_%s_vae.safetensors" % tipo}},
            "V": {"class_type": "LoadImage", "inputs": {"image": imagen}},
            "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["V", 0], "vae": ["14", 0]}},
            "19": {"class_type": "DisableNoise", "inputs": {}},
            "18": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "20": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.5, 0"}},
            "10": {"class_type": "BasicGuider", "inputs": {"model": ["8", 0], "conditioning": ["9", 0]}},
            "3": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["19", 0], "guider": ["10", 0], "sampler": ["18", 0], "sigmas": ["20", 0], "latent_image": ["13", 0]}},
            "15": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["14", 0]}},
            "17": {"class_type": "MarigoldV2PostProcess", "inputs": {"image": ["15", 0], "prediction": tipo}},
            "S": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": pref}}}


QWEN38 = "qwen3.8_27b_w4a8.safetensors"


def _vlm_listo():
    """The local Qwen VLM (TextGenerate + the QWEN38 text encoder) is installed. Without it the layers use a generic scene text."""
    try:
        import nodes
        if "TextGenerate" not in nodes.NODE_CLASS_MAPPINGS:
            return False
    except Exception:
        pass
    return comun.hay_modelo("text_encoders", QWEN38)


PEDIDO_DESCRIPCION = (
    "Describe this image in one detailed paragraph for an image-to-layers model. Name every distinct element from the foreground to the "
    "background — characters, creatures, props, ground, rocks, plants, water, hills and sky — with its colors and where it is in the frame, "
    "and describe what is partly hidden behind the foreground objects. Output only the description, no lists, no Markdown.")


def grafo_capas_stable(imagen, pref, n=4, semilla=42, hd=None):
    """Stable-Layers: the Qwen-Image-Layered graph + its LoRA and recipe (Heun 50, cfg 1, fixed text, 640 px). With hd, like the Qwen
    layers: SeedVR2 in the same job and capas_hd on CPU."""
    G = grafo_capas(imagen, pref, n, TEXTO_STABLE, semilla, hd=hd)
    G["37b"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["37", 0], "lora_name": LORA_STABLE, "strength_model": 1.0}}
    G["66"]["inputs"]["model"] = ["37b", 0]
    G["3"]["inputs"].update(sampler_name="heun", steps=50, cfg=1.0)
    return G


def grafo_capas(imagen, pref, n=4, texto=None, semilla=42, hd=None):
    """Qwen-Image-Layered: the official "Image to Layers" subgraph, expanded (fp8mixed instead of bf16, which does not fit in 32 GB).
    Without text, local Qwen3.8 LOOKS at the BG and describes it in the SAME job: the template expects a description of the whole scene,
    including hidden parts; a generic text tends to give a flat-colour layer 1."""
    G = {"1": {"class_type": "LoadImage", "inputs": {"image": imagen}},
            "2": {"class_type": "ImageScaleToMaxDimension", "inputs": {"image": ["1", 0], "upscale_method": "lanczos", "largest_size": 640}},
            "37": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_layered_fp8mixed.safetensors", "weight_dtype": "default"}},
            "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
            "39": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_layered_vae.safetensors"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": texto or TEXTO_CAPAS}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": ""}},
            "71": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["39", 0]}},
            "69": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["71", 0]}},
            "70": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["71", 0]}},
            "66": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["37", 0], "shift": 1}},
            "78": {"class_type": "GetImageSize", "inputs": {"image": ["2", 0]}},
            "118": {"class_type": "EmptyQwenImageLayeredLatentImage", "inputs": {"width": ["78", 0], "height": ["78", 1], "layers": n, "batch_size": 1}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["66", 0], "positive": ["69", 0], "negative": ["70", 0], "latent_image": ["118", 0], "seed": semilla,
                                                     "steps": 20, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
            "140": {"class_type": "LatentCut", "inputs": {"samples": ["3", 0], "dim": "t", "index": 1, "amount": 16384}},
            "76": {"class_type": "LatentCutToBatch", "inputs": {"samples": ["140", 0], "dim": "t", "slice_size": 1}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["76", 0], "vae": ["39", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": pref}}}
    if not texto and _vlm_listo():
        G.update({"qc": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN38, "type": "stable_diffusion", "device": "default"}},
                  "qg": {"class_type": "TextGenerate", "inputs": {"clip": ["qc", 0], "prompt": PEDIDO_DESCRIPCION, "image": ["2", 0], "max_length": 600,
                                                                 "sampling_mode": "on", "sampling_mode.temperature": 0.4, "sampling_mode.top_k": 40,
                                                                 "sampling_mode.top_p": 0.9, "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05,
                                                                 "sampling_mode.seed": int(semilla), "thinking": False, "use_default_template": True, "mtp": "auto",
                                                                 "system_prompt": ""}},
                  "qp": {"class_type": "PreviewAny", "inputs": {"source": ["qg", 0]}}})   # the description stays in the job history
        G["6"]["inputs"]["text"] = ["qg", 0]
    if hd:   # Qwen Layered is ~640 px: each layer's colour → SeedVR2 up to the BG size in the SAME job (without alpha: SplitImageWithAlpha);
        #       alpha and visible parts are rebuilt on CPU from the beauty (capas_hd). SeedVR2 is a VIDEO model: a batch of layers is treated
        #       as consecutive frames and objects bleed between them → one SeedVR2 chain PER LAYER (ImageFromBatch), each with SaveImage <pref>_hd_<i>.
        ES = _mod("escalar_imagen_srv")
        if ES is not None and not ES.faltan_modelos():
            W, H = hd
            w, h = (640, max(1, round(640 * H / W))) if W >= H else (max(1, round(640 * W / H)), 640)
            G["sa"] = {"class_type": "SplitImageWithAlpha", "inputs": {"image": ["8", 0]}}
            try:
                mult = ES.multiplicador(w, h, "l%d" % max(W, H))
                for i in range(n):
                    G["ib%d" % i] = {"class_type": "ImageFromBatch", "inputs": {"image": ["sa", 0], "batch_index": i, "length": 1}}
                    g = ES.grafo("_", mult, semilla, pref + "_hd_%d" % (i + 1))
                    del g["1"], g["50"]
                    g["57"]["inputs"]["input"] = ["ib%d" % i, 0]
                    for k, nodo in g.items():
                        for kk, v in nodo["inputs"].items():
                            if isinstance(v, list) and len(v) == 2 and v[0] in g:
                                nodo["inputs"][kk] = ["sv%d_%s" % (i, v[0]), v[1]]
                        G["sv%d_%s" % (i, k)] = nodo
            except ValueError:
                pass
    return G


def capas_hd(carpeta, beauty, pre="capa"):
    """Layers at BG resolution. For layer i (1 = background, last one on top):
    · alpha = Qwen's alpha upscaled and snapped to the beauty's real edges;
    · visible_i = alpha_i · Π_{j>i}(1 − alpha_j) = what of that layer is SEEN in the beauty → there the colour IS the beauty (full res);
    · hidden parts (Qwen's invented fill, e.g. the clean plate behind an object) = its colour upscaled with SeedVR2 (capa_hd_*).
    Writes capa_full_N.png (RGBA), once.
    · HALO: edge colour in the beauty mixes in the background → decontaminated (edge strip pulls colour from the opaque interior).
    · GHOST: the upper layer's contour does not match the beauty pixel-exactly → what upper layers cover is grown a few px first.
    · FLAT LAYER (a plain colour Qwen sometimes leaves): nearly opaque with no colour variation → written empty (the page hides it)."""
    import cv2
    import numpy as np
    base = sorted(glob.glob(os.path.join(carpeta, pre + "_[0-9]*.png")))         # pre = "capa" (Qwen Layered) or "scapa" (Stable-Layers)
    sube = sorted(glob.glob(os.path.join(carpeta, pre + "_hd_[0-9]_*.png")))   # <pre>_hd_<layer>_00001_.png (one SeedVR2 chain per layer)
    if not base or len(sube) != len(base):
        return []
    destinos = [os.path.join(carpeta, "%s_full_%d.png" % (pre, i + 1)) for i in range(len(base))]
    if all(os.path.isfile(d) for d in destinos):
        return destinos
    B = cv2.imread(beauty, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    H, W = B.shape[:2]
    guia = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    alfas, planas = [], []
    for p in base:
        a0 = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        a = (a0[..., 3] if a0.ndim == 3 and a0.shape[2] == 4 else np.full(a0.shape[:2], 255, np.uint8)).astype(np.float32) / 255.0
        planas.append(a.mean() > 0.9 and float(a0[..., :3].astype(np.float32).std(axis=(0, 1)).max()) < 6.0)   # opaque and single-coloured
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
        alfas.append(afinar_borde(B, a))                           # snap the 640 px edge to the beauty's real edge (avoids ~12 px halo)
    # how much the covering area grows: ~18 px at 1424. The beauty is used only WELL away from upper objects; less let the object's edge
    # leak into the background layer as an outline.
    abre = max(6, W // 80)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abre + 1, 2 * abre + 1))
    for i, (p, s, d) in enumerate(zip(base, sube, destinos)):
        if planas[i]:
            cv2.imwrite(d, np.zeros((H, W, 4), np.uint8))       # flat layer: empty (not shown)
            continue
        tapa = np.zeros((H, W), np.float32)
        for a in alfas[i + 1:]:
            tapa = np.maximum(tapa, cv2.dilate((a > 0.5).astype(np.float32), k))
        vis = (1 - cv2.GaussianBlur(tapa, (0, 0), abre / 2))[..., None]
        C = cv2.resize(cv2.imread(s, cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
        # The invented fill has a DIFFERENT tone from the beauty, leaving a tinted silhouette. Match it: low-frequency correction
        # (beauty − fill) measured where the layer IS visible and extended to the hidden part (normalised convolution).
        v = ((vis[..., 0] > 0.5) & (alfas[i] > 0.5)).astype(np.float32)
        if v.mean() > 0.02:
            sg = max(8, W // 15)
            corr = cv2.GaussianBlur((B - C) * v[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(v, (0, 0), sg), 1e-3)[..., None]
            C = np.clip(C + corr, 0, 1)
        rgb = decontaminar(vis * B + (1 - vis) * C, alfas[i])
        out = np.dstack([np.clip(rgb, 0, 1) * 255, alfas[i] * 255]).astype(np.uint8)
        cv2.imwrite(d, out)
    return destinos


def afinar_borde(B, a, banda=None):
    import cv2
    import numpy as np
    """Fast matting on the beauty: in a band around the rough edge, alpha comes from the ratio between the nearest INSIDE and OUTSIDE colours
    (projection of I−Bg onto Fg−Bg), so it follows the image's real edge; outside the band it stays hard.
    B beauty float 0–1 HxWx3 (BGR), a rough mask 0–1. Returns alpha 0–1."""
    H, W = a.shape
    banda = banda or max(6, W // 200)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * banda + 1, 2 * banda + 1))
    dura = (a > 0.5).astype(np.uint8)
    dentro = cv2.erode(dura, k).astype(np.float32)          # surely inside
    fuera = (1 - cv2.dilate(dura, k)).astype(np.float32)     # surely outside
    duda = (dentro == 0) & (fuera == 0)
    sg = banda * 0.8

    def media(m):                                            # nearby mean colour of that zone (normalised convolution)
        num = cv2.GaussianBlur(B * m[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(m, (0, 0), sg)[..., None]
        return num / np.maximum(den, 1e-4), den[..., 0]
    F, df = media(dentro)
    G, dg = media(fuera)
    d = F - G
    al = ((B - G) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-4)
    al = np.clip((al - 0.5) * 1.8 + 0.5, 0, 1)                # shorter transition: crisp antialiased edge, not soft
    poco_contraste = (d * d).sum(-1) < 0.004                 # inside and outside nearly the same colour: undecidable → original mask
    al = np.where(poco_contraste, cv2.GaussianBlur(dura.astype(np.float32), (0, 0), 1.0), al)
    out = np.where(duda, al, dura.astype(np.float32))
    return _guiado(B.astype(np.float32), out.astype(np.float32), 1, 1e-5).clip(0, 1)


def _mate_fino(ruta, beauty):
    """Material matte with refined edge (afinar_borde) → <name>_fino.png next to it; SAM3's original is untouched. Built once."""
    import cv2
    import numpy as np
    fino = os.path.splitext(ruta)[0] + "_fino.png"
    if os.path.isfile(fino) or not beauty:
        return fino if os.path.isfile(fino) else ruta
    B = cv2.imread(beauty, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    m = cv2.resize(cv2.imread(ruta, cv2.IMREAD_GRAYSCALE), (B.shape[1], B.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    cv2.imwrite(fino, (afinar_borde(B, m) * 255 + 0.5).astype(np.uint8))
    return fino


def decontaminar(rgb, alfa, opaco=0.92):
    """Removes edge halo: where alpha is partial, colour is taken from the nearest opaque interior (a "pull" with multi-scale normalised
    blurs), like a keyer's edge decontamination. rgb float 0–1 HxWx3, alfa HxW."""
    import cv2
    import numpy as np
    nucleo = (alfa >= opaco).astype(np.float32)
    if nucleo.sum() < 16:
        return rgb
    ext = rgb.copy()
    hueco = alfa < opaco
    for s in (2, 4, 8, 16):
        num = cv2.GaussianBlur(rgb * nucleo[..., None], (0, 0), s)
        den = cv2.GaussianBlur(nucleo, (0, 0), s)[..., None]
        val = den[..., 0] > 1e-3
        poner = hueco & val
        ext[poner] = (num / np.maximum(den, 1e-6))[poner]
        hueco = hueco & ~val
    return np.where((alfa < opaco)[..., None], ext, rgb)


def _dir_pases(lote, motor, stem=None):
    p = os.path.join(DIR_SAL, lote, "pases_" + motor)
    return os.path.join(p, stem) if stem else p


def _ultimo(carpeta, nombre):
    # only SaveImage's numbered outputs (…_00001_.png): "_*" also matched files like "…_especular_brillos.png"
    L = sorted(glob.glob(os.path.join(carpeta, glob.escape(nombre) + "_[0-9][0-9][0-9][0-9][0-9]_.png")))
    return L[-1] if L else None


# ───────── Depth source: Depth Anything V2, or MoGe as fallback ─────────
# Depth Anything V2 comes from an optional custom node pack (comfyui_controlnet_aux) and can be missing or fail at runtime (e.g. an xformers
# build without CUDA kernels). Parallax and the CPU passes (refined depth, fog, position, cavity) must not wait forever for it, so:
#   · when the pack is not installed, the depth pass queues MoGe (core nodes) instead; its RenderPassesMoGeEXR writes depth_moge.png
#     (16-bit, white = near, like Depth Anything's);
#   · when Depth Anything's job fails or disappears, MoGe is queued once as a backup (_asegurar_depth);
#   · _fuente_depth() gives Depth Anything's image when there is one (preferred) and depth_moge.png only once Depth Anything is out.
# depth_job.json in the plate's passes folder records which source was asked for: {fuente, prompt_id?, img, pref, respaldo?}.
DEPTH_JOB = "depth_job.json"


def _clases_nodos():
    try:
        import nodes
        return set(nodes.NODE_CLASS_MAPPINGS)
    except Exception:
        return None                                          # outside ComfyUI: unknown


def _da_instalado():
    c = _clases_nodos()
    return c is None or "DepthAnythingV2Preprocessor" in c


def _moge_instalado():
    c = _clases_nodos()
    nodos = c is None or {"LoadMoGeModel", "MoGeInference", "RenderPassesMoGeEXR"} <= c
    return nodos and any(comun.hay_modelo("geometry_estimation", m) for m in MODELOS_MOGE)


def _trabajo_comfy(pid):
    """State of a ComfyUI job read in-process: "cola" (queued or running), "listo", "error", "perdido" (neither queued nor in the history:
    removed or lost in a restart) or "?" (cannot tell)."""
    try:
        q = PromptServer.instance.prompt_queue
        corre, pend = q.get_current_queue_volatile() if hasattr(q, "get_current_queue_volatile") else q.get_current_queue()
        if any(len(it) > 1 and it[1] == pid for it in list(corre) + list(pend)):
            return "cola"
        h = (q.get_history(prompt_id=pid) or {}).get(pid)
    except Exception:
        return "?"
    if not h:
        return "perdido"
    st = h.get("status") or {}
    return "listo" if st.get("status_str") == "success" or (st.get("completed") and st.get("status_str") != "error") else "error"


def _da_descartado(dp):
    """True when Depth Anything will not give this plate a depth: not requested this way (MoGe was), its job failed or vanished, or it
    finished without an image."""
    J = comun.leer_json(os.path.join(dp, DEPTH_JOB), {}) or {}
    if J.get("fuente") == "depth_anything" and J.get("prompt_id"):
        return _trabajo_comfy(J["prompt_id"]) in ("error", "perdido", "listo")   # "listo" here = it ran but saved nothing
    return True


def _fuente_depth(dp):
    """The depth image of a plate (white = near): Depth Anything V2's if present, otherwise MoGe's once Depth Anything is out."""
    r = _ultimo(dp, "depth")
    if r:
        return r
    m = os.path.join(dp, "depth_moge.png")
    return m if os.path.isfile(m) and _da_descartado(dp) else None


def _depth_fallido(dp):
    """Why this plate will get no depth (text), or None if it has one or one is still coming."""
    if _fuente_depth(dp):
        return None
    J = comun.leer_json(os.path.join(dp, DEPTH_JOB), None)
    if not J:
        return "depth is missing: run the Depth pass first"
    if not _da_descartado(dp):
        return None
    r = J.get("respaldo")
    if r == "pase_moge":                                     # the MoGe pass of the same launch makes it
        return "MoGe finished without a depth" if os.path.isfile(os.path.join(dp, "moge.json")) else None
    if r:
        return "depth failed (Depth Anything V2 and the MoGe fallback)" if _trabajo_comfy(r) in ("error", "perdido") else None
    if not _moge_instalado():
        return "Depth Anything V2 failed and MoGe (the fallback) is not installed"
    return None                                              # _asegurar_depth queues the MoGe backup


def _gris01(ruta):
    """A grey PNG (8 or 16 bit, 1 or 3 channels) as float 0–1."""
    import cv2
    import numpy as np
    a = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError("could not read %s" % os.path.basename(ruta))
    if a.ndim == 3:
        a = a[..., :3].mean(-1) if a.shape[2] >= 3 else a[..., 0]
    return a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 or a.max() > 255 else 255.0)


async def _encolar_moge(s, img, pref):
    """Posts the MoGe graph (P, depth and lens to EXR + depth_moge.png) → prompt_id or None."""
    async with s.post("http://127.0.0.1:%s/prompt" % comun.puerto(), json={"prompt": comun.modelos_os(grafo_moge(img, pref.rstrip("/")))}) as q:
        return (await q.json()).get("prompt_id")


async def _encolar_depth(s, img, pref, lado, dp, moge_en_lote=False):
    """Queues this plate's depth: Depth Anything V2 if its node is installed, otherwise MoGe. → (jobs queued, warnings)."""
    J = {"img": img, "pref": pref}
    if _da_instalado():
        async with s.post("http://127.0.0.1:%s/prompt" % comun.puerto(), json={"prompt": comun.modelos_os(grafo_depth(img, pref + "depth", lado))}) as q:
            r = await q.json()
        if r.get("prompt_id"):
            comun.escribir_json(os.path.join(dp, DEPTH_JOB), dict(J, fuente="depth_anything", prompt_id=r["prompt_id"]))
            return 1, []
        aviso = "Depth Anything V2 was rejected (%s): depth comes from MoGe" % str(r.get("error") or r)[:160]
    else:
        aviso = "Depth Anything V2 (comfyui_controlnet_aux) is not installed: depth comes from MoGe"
    if not _moge_instalado():
        return 0, ["Depth: neither Depth Anything V2 (comfyui_controlnet_aux) nor MoGe (moge_3_vitg_fp16.safetensors in geometry_estimation) is installed"]
    if moge_en_lote:                                         # the MoGe pass of this launch writes depth_moge.png
        comun.escribir_json(os.path.join(dp, DEPTH_JOB), dict(J, fuente="moge", respaldo="pase_moge"))
        return 0, [aviso]
    pid = await _encolar_moge(s, img, pref)
    comun.escribir_json(os.path.join(dp, DEPTH_JOB), dict(J, fuente="moge", respaldo=pid))
    return (1 if pid else 0), [aviso]


async def _asegurar_depth(dp):
    """If Depth Anything failed or vanished for this plate and MoGe has not been asked yet, queue MoGe once as the backup depth."""
    J = comun.leer_json(os.path.join(dp, DEPTH_JOB), None)
    if not J or J.get("respaldo") or _fuente_depth(dp) or not _da_descartado(dp) or os.path.isfile(os.path.join(dp, "depth_moge.png")):
        return
    if lote_cancelado(_lote_de(dp)) or not _moge_instalado() or not J.get("img"):
        return
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            pid = await _encolar_moge(s, J["img"], J["pref"])
    except Exception as e:
        print("[RenderPasses] depth fallback (MoGe):", e)
        return
    J["respaldo"] = pid or "rechazado"
    comun.escribir_json(os.path.join(dp, DEPTH_JOB), J)
    print("[RenderPasses] Depth Anything V2 failed for %s: MoGe queued as the depth fallback" % dp)


# Assumed camera to turn depth (relative: white = near, i.e. a disparity) into distance: 1/Z = D/cerca + (1−D)/lejos, with a 40° horizontal
# FOV (≈ 50 mm full frame). Not the real layout camera: good for position mattes, 3D fog and relighting, not for measuring.
# lejos = 20 spreads the scene well (with 100 everything but the sky collapsed to 1–2); the sky is placed separately at the back.
CAMARA = {"cerca": 1.0, "lejos": 20.0, "fov": 40.0}


def cielo_de(carpeta, D):
    """Sky mask: from Cryptomatte (cripto_cielo.png) if present; otherwise what depth gives as farthest."""
    import cv2
    import numpy as np
    H, W = D.shape
    cf = os.path.join(carpeta, "cripto_cielo.png")
    if os.path.isfile(cf):
        return cv2.resize(cv2.imread(cf, cv2.IMREAD_GRAYSCALE), (W, H)) > 127
    return D < 0.02


def posicion(depth_png, W, H, carpeta=None):
    """depth_afinado (16-bit) → (P, Z): P = camera-space position (X right, Y up, looking down −Z as in Nuke), Z = distance.
    The sky is placed at the "lejos" distance."""
    import cv2
    import numpy as np
    D = cv2.resize(cv2.imread(depth_png, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0, (W, H), interpolation=cv2.INTER_AREA)
    D = np.clip(D, 0, 1)
    Z = 1.0 / (D / CAMARA["cerca"] + (1.0 - D) / CAMARA["lejos"])
    if carpeta:
        Z = np.where(cielo_de(carpeta, D), CAMARA["lejos"], Z)
    f = (W / 2.0) / np.tan(np.radians(CAMARA["fov"]) / 2.0)
    u = (np.arange(W, dtype=np.float32) + 0.5 - W / 2.0) / f
    v = (H / 2.0 - (np.arange(H, dtype=np.float32) + 0.5)) / f
    P = np.stack([u[None, :] * Z, v[:, None] * Z, -Z], -1).astype(np.float32)
    return P, Z.astype(np.float32)


def _guiado(guia, p, r, eps):
    """Guided filter: OpenCV contrib's when it is there (opencv-python 5 ships cv2.ximgproc without guidedFilter), otherwise the same
    filter (He et al.) with box filters on the grey guide."""
    import cv2
    import numpy as np
    f = getattr(getattr(cv2, "ximgproc", None), "guidedFilter", None)
    if f is not None:
        return f(guia, p, r, eps)
    g = guia.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    p = p.astype(np.float32)
    caja = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mg, mp = caja(g), caja(p)
    a = (caja(g * p) - mg * mp) / (caja(g * g) - mg * mg + eps)
    b = mp - a * mg
    return caja(a) * g + caja(b)


def _leer_exr(ruta):
    import OpenImageIO as oiio
    i = oiio.ImageInput.open(ruta)
    if not i:
        return None, None
    sp = i.spec()
    a = i.read_image(oiio.FLOAT)
    i.close()
    return sp, a


# Real HDR for specular, highlights, reflections and emission without running each pass through LTX (they are almost black images with
# spots, nothing like the video it was trained on): the reconstruction LTX 2.5 already made of the BEAUTY is reused. Gain = luminance of
# beauty_hdr ÷ that of the SDR beauty (normalised in the midtones, only ≥ 1), multiplied into the linear pass. Lights the PNG clipped at
# 1.0 get back the range LTX gave them, and specular, highlights, reflections and emission still add up like the beauty. Output in
# ACEScg, like beauty_hdr.exr.
PASES_HDR = re.compile(r"· (Specular|Highlights|Reflections|Emission · original colour)$")
AP1_DESDE_709 = ((0.61319, 0.33951, 0.04737), (0.07021, 0.91634, 0.01345), (0.02062, 0.10957, 0.86961))


def _ganancia_hdr(carpeta, beauty):
    """The range gain (H×W, ≥ 1) of beauty_hdr.exr over the SDR beauty; cached in hdr_ganancia.npy."""
    import cv2
    import numpy as np
    fh = os.path.join(carpeta, "beauty_hdr.exr")
    fg = os.path.join(carpeta, "hdr_ganancia.npy")
    if not os.path.isfile(fh):
        return None
    if os.path.isfile(fg) and os.path.getmtime(fg) >= os.path.getmtime(fh):
        return np.load(fg)
    _, a = _leer_exr(fh)
    B = cv2.imread(beauty, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    H, W = B.shape[:2]
    a = cv2.resize(a[..., :3].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    yh = a @ np.array([0.2722287, 0.6740818, 0.0536895], np.float32)            # AP1 luminance
    lin = np.where(B <= 0.04045, B / 12.92, ((B + 0.055) / 1.055) ** 2.4)
    ys = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    r = yh / np.maximum(ys, 1e-3)
    medios = (ys > 0.04) & (ys < 0.5)
    r /= max(float(np.median(r[medios])) if medios.any() else 1.0, 1e-3)       # same exposure in the midtones
    g = np.clip(cv2.GaussianBlur(r.astype(np.float32), (0, 0), 1.5), 1.0, 64.0)
    np.save(fg, g)
    return g


def pase_hdr(ruta, carpeta, beauty):
    """<pass>_hdr.exr (ACEScg, float) next to the pass; None if there is no beauty HDR."""
    import cv2
    import numpy as np
    dst = os.path.splitext(ruta)[0] + "_hdr.exr"
    fh = os.path.join(carpeta, "beauty_hdr.exr")
    if not os.path.isfile(fh):
        return None
    if os.path.isfile(dst) and os.path.getmtime(dst) >= max(os.path.getmtime(ruta), os.path.getmtime(fh)):
        return dst
    g = _ganancia_hdr(carpeta, beauty)
    P = cv2.imread(ruta, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    if P.shape[:2] != g.shape:
        g = cv2.resize(g, (P.shape[1], P.shape[0]), interpolation=cv2.INTER_LINEAR)
    lin = np.where(P <= 0.04045, P / 12.92, ((P + 0.055) / 1.055) ** 2.4) * g[..., None]
    _exr(dst, lin @ np.array(AP1_DESDE_709, np.float32).T)
    return dst


_DATO_IA = re.compile(r"(oclusion|sombras|mate|key)", re.I)   # AI passes that are data (not converted from sRGB to linear)


def paquete_nuke(carpeta, beauty):
    """One multilayer EXR (pases_nuke.exr) with everything available for the BG + a pases.nk script that reads it and rebuilds the beauty
    (albedo × raw_lighting), with Cryptomatte and ZDefocus ready. Rebuilt only if a source is newer. Returns {exr, nk} or {}."""
    import cv2
    import numpy as np
    import OpenImageIO as oiio
    dst, nk = os.path.join(carpeta, "pases_nuke.exr"), os.path.join(carpeta, "pases.nk")
    ia = {}
    for r in sorted(glob.glob(os.path.join(carpeta, "ia_*.png"))):
        k = re.sub(r"_\d{5}_$", "", os.path.splitext(os.path.basename(r))[0])
        if k.endswith("_cpu"):
            continue
        ia[k] = r                                                  # latest version of each pass
    fuentes = [beauty] + [x for x in (os.path.join(carpeta, f) for f in ("albedo_lineal.exr", "raw_lighting.exr", "depth_afinado.png", "niebla.png",
                                                                          "cryptomatte.exr", "position_moge.exr")) if os.path.isfile(x)]
    nor = _ultimo(carpeta, "normals")
    if nor:
        fuentes.append(nor)
    fuentes += list(ia.values())
    if not os.path.isfile(os.path.join(carpeta, "albedo_lineal.exr")) and not os.path.isfile(os.path.join(carpeta, "depth_afinado.png")):
        return {}
    if os.path.isfile(dst) and os.path.isfile(nk) and os.path.getmtime(dst) >= max(os.path.getmtime(x) for x in fuentes):
        return {"exr": dst, "nk": nk}
    I = cv2.cvtColor(cv2.imread(beauty, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = I.shape[:2]
    lin = lambda a: np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)
    planos, nombres, tipos, capas, attrs = [], [], [], [], []

    def pon(capa, a, flotante=False):
        a = a.astype(np.float32)
        if a.shape[:2] != (H, W):
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        a = a[..., None] if a.ndim == 2 else a
        a = np.repeat(a, 3, 2) if a.shape[2] == 1 else a[..., :3]
        for c in range(3):
            planos.append(a[..., c]); nombres.append("%s.%s" % (capa, "RGB"[c])); tipos.append("float" if flotante else "half")
        capas.append(capa)

    Il = lin(I)
    for c in range(3):
        planos.append(Il[..., c]); nombres.append("RGB"[c]); tipos.append("half")
    planos.append(np.ones((H, W), np.float32)); nombres.append("A"); tipos.append("half")
    for capa, f in (("albedo", "albedo_lineal.exr"), ("raw_lighting", "raw_lighting.exr")):
        sp, a = _leer_exr(os.path.join(carpeta, f)) if os.path.isfile(os.path.join(carpeta, f)) else (None, None)
        if a is not None:
            pon(capa, a, True)
    da = os.path.join(carpeta, "depth_afinado.png")
    pm_, zm_ = os.path.join(carpeta, "position_moge.exr"), os.path.join(carpeta, "depth_moge.exr")
    if os.path.isfile(pm_) and os.path.isfile(zm_):              # MoGe-2's P (metres, estimated lens) wins
        _, Pm = _leer_exr(pm_); _, Zm = _leer_exr(zm_)
        planos.append(cv2.resize(Zm[..., 0], (W, H), interpolation=cv2.INTER_LINEAR)); nombres.append("depth.Z"); tipos.append("float")
        pon("P", cv2.resize(Pm[..., :3], (W, H), interpolation=cv2.INTER_LINEAR), True)
    elif os.path.isfile(da):
        P, Z = posicion(da, W, H, carpeta)
        planos.append(Z); nombres.append("depth.Z"); tipos.append("float")
        pon("P", P, True)
    if nor:
        N = cv2.cvtColor(cv2.imread(nor, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 * 2.0 - 1.0
        pon("N", N, True)
    if os.path.isfile(os.path.join(carpeta, "niebla.png")):
        pon("niebla", cv2.imread(os.path.join(carpeta, "niebla.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
    for k, r in ia.items():
        a = cv2.cvtColor(cv2.imread(r, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pon(re.sub(r"[^A-Za-z0-9_]", "_", k[3:]), a if _DATO_IA.search(k) else lin(a))
    cm = os.path.join(carpeta, "cryptomatte.exr")
    if os.path.isfile(cm):                                         # Cryptomatte channels and metadata as-is: the Cryptomatte node reads this EXR
        sp, a = _leer_exr(cm)
        if a is not None and a.shape[:2] == (H, W):
            for i, n in enumerate(sp.channelnames):
                if n.startswith("Crypto"):
                    planos.append(a[..., i]); nombres.append(n); tipos.append("float")
            attrs = [(x.name, x.value) for x in sp.extra_attribs if x.name.startswith("cryptomatte/")]
    datos = np.ascontiguousarray(np.stack(planos, -1), np.float32)
    spec = oiio.ImageSpec(W, H, len(nombres), oiio.FLOAT)
    spec.channelnames = tuple(nombres)
    spec.channelformats = tuple(oiio.TypeDesc(t) for t in tipos)
    spec.attribute("compression", "zip")
    for k, v in attrs:
        spec.attribute(k, v)
    tmp = dst + ".tmp.exr"
    o = oiio.ImageOutput.create(tmp)
    if not o or not o.open(tmp, spec) or not o.write_image(datos):
        raise ValueError("could not write pases_nuke.exr")
    o.close()
    os.replace(tmp, dst)
    with open(nk, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(script_nuke(dst, W, H, capas, os.path.isfile(cm), "depth.Z" in nombres))
    return {"exr": dst, "nk": nk}


def script_nuke(exr, W, H, capas, cripto, z):
    """pases.nk: Read of the multilayer EXR → Shuffle albedo and raw_lighting → Merge multiply (= beauty, as a check), Cryptomatte and
    ZDefocus (disabled) off the Read, and a sticky note."""
    ruta = exr.replace("\\", "/")
    L = ["#! nuke", "version 13.0", "Root {", " inputs 0", ' format "%d %d 0 0 %d %d 1 renderpasses"' % (W, H, W, H), "}"]
    for c in capas:
        L.append("add_layer {%s %s.red %s.green %s.blue}" % (c, c, c, c))
    L += ["Read {", " inputs 0", ' file "%s"' % ruta, ' format "%d %d 0 0 %d %d 1 renderpasses"' % (W, H, W, H), " origset true",
          " name Pases", " xpos 0", " ypos -300", "}", "set Npases [stack 0]"]
    if "albedo" in capas and "raw_lighting" in capas:
        L += ["Shuffle {", " in albedo", " name Albedo", " xpos 0", " ypos -180", "}", "push $Npases",
              "Shuffle {", " in raw_lighting", " name Raw_lighting", " xpos 160", " ypos -180", "}",
              "Merge2 {", " inputs 2", " operation multiply", " name Beauty_rehecho", " label \"albedo × raw lighting\"", " xpos 80", " ypos -80", "}"]
    if cripto:
        L += ["push $Npases", "Cryptomatte {", " name Cryptomatte1", " xpos 320", " ypos -180", "}"]
    if z:
        L += ["push $Npases", "ZDefocus2 {", " z_channel depth.Z", " math depth", " name ZDefocus", " disable true", " xpos 480", " ypos -180", "}"]
    L += ["StickyNote {", " inputs 0", ' label "RenderPasses · passes of a painted BG (estimated from 2D, not render AOVs)\\n'
          'rgba = linear beauty · albedo × raw_lighting = beauty · P = camera-space position (%d°, near %g, far %g) · depth.Z = distance\\n'
          'N = normals · Crypto* = Cryptomatte (CryptoObject / CryptoAsset / CryptoMaterial)"' % (CAMARA["fov"], CAMARA["cerca"], CAMARA["lejos"]),
          " xpos 0", " ypos -400", "}", ""]
    return "\n".join(L)


def _al_tamano_beauty(carpeta, W, H):
    """The AI passes already return at the beauty size (a_resolucion), but depth, normals, albedo and the MoGe passes stayed at the size of
    the reduced copy (e.g. 2656×1568) or at the one the model returns (e.g. depth 4553×2688). They are brought once to the EXACT beauty size:
    colour and previews with lanczos, normals renormalised, P and depth in float with linear interpolation."""
    import cv2
    import numpy as np
    from PIL import Image
    for pref in ("depth", "normals", "albedo"):
        f = _ultimo(carpeta, pref)
        if not f:
            continue
        with Image.open(f) as im:
            if im.size == (W, H):
                continue
        a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        b = cv2.resize(a, (W, H), interpolation=cv2.INTER_LANCZOS4 if W > a.shape[1] else cv2.INTER_AREA)
        if pref == "normals":
            n = b[..., :3].astype(np.float32) / (65535.0 if b.dtype == np.uint16 else 255.0) * 2 - 1
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
            mx = 65535.0 if b.dtype == np.uint16 else 255.0
            b[..., :3] = np.clip((n * 0.5 + 0.5) * mx + 0.5, 0, mx).astype(b.dtype)
        cv2.imwrite(f, b)
    for n in ("position_moge.png", "depth_moge.png", "cielo_moge.png"):
        f = os.path.join(carpeta, n)
        if os.path.isfile(f):
            with Image.open(f) as im:
                if im.size == (W, H):
                    continue
            a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(f, cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST if n == "cielo_moge.png" else cv2.INTER_LINEAR))
    for n in ("position_moge.exr", "depth_moge.exr", "normales_moge.exr"):
        f = os.path.join(carpeta, n)
        if os.path.isfile(f):
            sp, a = _leer_exr(f)
            if a is None or (sp.width, sp.height) == (W, H):
                continue
            b = cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            if n == "normales_moge.exr":
                b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-6)
            _exr(f, b)


def derivados(carpeta, beauty):
    """CPU passes (shading, shadows, fog) once albedo and depth exist. Written once."""
    import cv2
    import numpy as np
    alb, dep = _ultimo(carpeta, "albedo"), _fuente_depth(carpeta)   # depth: Depth Anything V2, or MoGe when it is out
    hechos = {}
    lin = lambda a: np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    lum = lambda x: 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    I = cv2.cvtColor(cv2.imread(beauty, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = I.shape[:2]
    try:
        _al_tamano_beauty(carpeta, W, H)
    except Exception as e:
        print("[RenderPasses] bglook resize to beauty:", e)
    if alb and not os.path.isfile(os.path.join(carpeta, "raw_lighting.exr")):
        # A true RAW LIGHTING (like V-Ray's RawTotalLighting): beauty ÷ albedo in linear, float32, no blur or normalisation → raw_lighting.exr.
        # With albedo_lineal.exr (the SAME albedo used in the division) albedo × raw_lighting = linear beauty, pixel-exact, like
        # DiffuseFilter × RawTotalLighting. raw_lighting.png is only a preview (99th percentile + gamma 2.2).
        A = cv2.resize(cv2.cvtColor(cv2.imread(alb, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        Al = np.maximum(lin(A), 1e-3).astype(np.float32)                      # floor: pure black albedo must not divide by zero
        Il = lin(I).astype(np.float32)
        # Per-channel division puts complementary colours into the light (Marigold's albedo is more saturated than the beauty), and light
        # colour cannot be recovered reliably from 2D. So: classic Retinex decomposition — NEUTRAL light with exact per-pixel intensity
        # (sharp, no blur) and albedo = beauty ÷ light keeps all the colour → albedo_lineal.exr × raw_lighting.exr = linear beauty, exact.
        Y = np.maximum(lum(Il) / np.maximum(lum(Al), 1e-3), 0).astype(np.float32)
        R = np.repeat(Y[..., None], 3, axis=2)
        Al = (Il / np.maximum(R, 1e-6)).astype(np.float32)
        _exr(os.path.join(carpeta, "albedo_lineal.exr"), Al)
        _exr(os.path.join(carpeta, "raw_lighting.exr"), R)
        v = np.clip(R / max(float(np.percentile(lum(R), 99)), 1e-6), 0, 1) ** (1 / 2.2)
        cv2.imwrite(os.path.join(carpeta, "raw_lighting.png"), cv2.cvtColor((v * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR))
    if dep and not os.path.isfile(os.path.join(carpeta, "depth_afinado.png")):
        # Depth is oversampled by Depth Anything, so it only needs an INTER_AREA downscale (antialiased, crisp edge) to the exact BG size;
        # a guided filter only softened it. Resized in FLOAT so the 16-bit output really has more than 256 levels (no banding).
        Dd = cv2.resize(_gris01(dep), (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(carpeta, "depth_afinado.png"), (np.clip(Dd, 0, 1) * 65535).astype(np.uint16))
    if dep and not os.path.isfile(os.path.join(carpeta, "niebla.png")):
        # As in compositing, a RAMP over inverted depth (far = white) with start, end and curve (the dark-channel method gave odd haze on
        # scenes without fog).
        D = cv2.resize(_gris01(dep), (W, H), interpolation=cv2.INTER_AREA)   # white = near
        lejos = 1.0 - D
        ini, fin, curva = NIEBLA
        bruma = np.clip((lejos - ini) / max(fin - ini, 1e-3), 0, 1) ** curva
        cv2.imwrite(os.path.join(carpeta, "niebla.png"), (cv2.GaussianBlur(bruma, (0, 0), 1.5) * 255).astype(np.uint8))
    da = os.path.join(carpeta, "depth_afinado.png")
    # normal-based passes: MoGe-2 normals if present, otherwise Marigold's
    nm = os.path.join(carpeta, "normales_moge.exr")
    N = None
    if os.path.isfile(nm):
        _, n = _leer_exr(nm)
        if n is not None:
            N = cv2.resize(n[..., :3], (W, H), interpolation=cv2.INTER_LINEAR)
    elif _ultimo(carpeta, "normals"):
        n = cv2.cvtColor(cv2.imread(_ultimo(carpeta, "normals"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 * 2 - 1
        N = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    if N is not None:
        N = N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-6)
        cielo_m = os.path.join(carpeta, "cielo_moge.png")
        cielo = (cv2.resize(cv2.imread(cielo_m, cv2.IMREAD_GRAYSCALE), (W, H)) > 127) if os.path.isfile(cielo_m) else np.zeros((H, W), bool)
        cv_f = os.path.join(carpeta, "cavidad.png")
        fuentes_n = [x for x in (nm if os.path.isfile(nm) else None, _ultimo(carpeta, "normals")) if x]
        if not os.path.isfile(cv_f) or os.path.getmtime(cv_f) < max(os.path.getmtime(x) for x in fuentes_n):
            # CAVITY: screen-space cavity on the refined depth (full res; normals of a PAINTED BG gave side-lit relief): each point vs. the
            # mean of its surroundings at several scales; recessed (corners, contacts, cracks) dark, protruding (edges) light. 50 % grey = flat.
            Dc = cv2.imread(da, cv2.IMREAD_UNCHANGED) if os.path.isfile(da) else None
            Dc = (cv2.resize(Dc.astype(np.float32) / 65535.0, (W, H), interpolation=cv2.INTER_AREA) if Dc is not None
                  else cv2.resize((N[..., 2] + 1) * 0.5, (W, H)))   # no refined depth: facing-the-camera from the normals
            acc = np.zeros_like(Dc)
            for sg, pe in ((W / 1000, 0.3), (W / 400, 0.35), (W / 150, 0.25), (W / 60, 0.1)):
                acc += pe * (Dc - cv2.GaussianBlur(Dc, (0, 0), max(0.8, sg)))
            k = 1.0 / max(float(np.percentile(np.abs(acc[~cielo]), 98)) if (~cielo).any() else 1.0, 1e-5)
            # Fine cavity detail: depth is smooth inside each shape; fine cracks and corners are PAINTED, so whatever is darker than its
            # immediate surroundings (1–3 px) in the beauty is added as recessed
            Lg = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            fino = Lg - cv2.GaussianBlur(Lg, (0, 0), max(1.2, W / 1200))
            kf = 1.0 / max(float(np.percentile(np.abs(fino[~cielo]), 98)) if (~cielo).any() else 1.0, 1e-5)
            acc = acc * k + 0.6 * np.clip(fino * kf, -1, 0.4)
            k = 1.0
            cav = np.clip(0.5 + 0.5 * acc * k, 0, 1)
            cav[cielo] = 0.5
            cv2.imwrite(cv_f, (cav * 255 + 0.5).astype(np.uint8))
            # GROUND: what faces up (normal Y > 0.75) and is not sky: where characters stand and catch their shadow
            su = ((N[..., 1] > 0.75) & ~cielo).astype(np.uint8)
            su = cv2.morphologyEx(su, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cv2.imwrite(os.path.join(carpeta, "suelo.png"), cv2.GaussianBlur(su.astype(np.float32), (0, 0), 1.5).__mul__(255).astype(np.uint8))
        rl = os.path.join(carpeta, "raw_lighting.exr")
        lz = os.path.join(carpeta, "luz.json")
        if os.path.isfile(rl) and (not os.path.isfile(lz) or os.path.getmtime(lz) < os.path.getmtime(cv_f)):
            # LIGHT FOR CHARACTERS: the key light that best explains the BG lighting with its normals (S ≈ fill + key·max(0, N·L)), searched
            # over 600 directions; key and fill colours come from the beauty in lit areas and in areas the key does not reach.
            _, R = _leer_exr(rl)
            S = cv2.resize(R[..., :3].mean(-1), (W, H), interpolation=cv2.INTER_AREA)
            ok = ~cielo & np.isfinite(S)
            idx = np.flatnonzero(ok.ravel())
            if idx.size > 2000:
                rng = np.random.default_rng(0)
                idx = rng.choice(idx, 20000, replace=idx.size < 20000)
                n_ = N.reshape(-1, 3)[idx]; s_ = S.ravel()[idx]
                mejor = None
                g = (np.sqrt(5) - 1) / 2
                for i in range(600):                                   # directions spread over the sphere (Fibonacci spiral)
                    z = 1 - (i + 0.5) / 300.0
                    if z < -0.2:
                        break
                    r = np.sqrt(max(0.0, 1 - z * z)); a = 2 * np.pi * i * g
                    L = np.array([r * np.cos(a), z, r * np.sin(a)], np.float32)   # x right, y up, z towards the camera
                    c = np.maximum(0, n_ @ L)
                    A = np.stack([np.ones_like(c), c], 1)
                    coef, res, *_ = np.linalg.lstsq(A, s_, rcond=None)
                    if coef[1] <= 0:
                        continue
                    e = float(np.mean((A @ coef - s_) ** 2))
                    if mejor is None or e < mejor[0]:
                        mejor = (e, L, coef)
                if mejor is not None:
                    _, L, coef = mejor
                    Il = np.where(I <= 0.04045, I / 12.92, ((I + 0.055) / 1.055) ** 2.4)
                    c_all = np.maximum(0, N @ L)
                    lit, som = ok & (c_all > 0.6), ok & (c_all < 0.1)
                    col = lambda m: Il[m].mean(0) if m.sum() > 50 else np.ones(3)
                    relleno, iluminado = col(som), col(lit)
                    key = np.maximum(iluminado - relleno, 1e-4)
                    hexc = lambda v: "#%02x%02x%02x" % tuple(int(x) for x in np.clip(v / max(v.max(), 1e-6), 0, 1) ** (1 / 2.2) * 255)
                    az = float(np.degrees(np.arctan2(L[0], L[2])))            # 0 = from the camera, + = from the right
                    el = float(np.degrees(np.arcsin(np.clip(L[1], -1, 1))))
                    lado = ("from the right" if az > 25 else "from the left" if az < -25 else "frontal")
                    if abs(az) > 120:
                        lado = "backlit " + ("from the right" if az > 0 else "from the left")
                    info = {"direccion": [round(float(x), 3) for x in L], "azimut": round(az, 1), "elevacion": round(el, 1), "lado": lado,
                            "key": hexc(key), "relleno": hexc(relleno), "proporcion": round(float(key.mean() / max(relleno.mean(), 1e-6)), 2)}
                    with open(lz, "w", encoding="utf-8") as f:
                        json.dump(info, f, ensure_ascii=False)
                    # grey sample sphere lit with that light (to match character lighting)
                    t = 320
                    yy, xx = np.mgrid[:t, :t].astype(np.float32)
                    u, v = (xx + .5) / t * 2 - 1, 1 - (yy + .5) / t * 2
                    d = u * u + v * v
                    nz = np.sqrt(np.clip(1 - d, 0, 1))
                    Ns = np.stack([u, v, nz], -1)
                    alb = 0.6
                    sh = alb * (relleno[None, None] / max(relleno.max(), 1e-6) * 0.35 + key[None, None] / max(key.max(), 1e-6) * np.maximum(0, Ns @ L)[..., None])
                    img = np.where((d <= 1)[..., None], np.clip(sh, 0, 1) ** (1 / 2.2), 0.08)
                    cv2.imwrite(os.path.join(carpeta, "luz_esfera.png"), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    pp = os.path.join(carpeta, "position.png")
    # with MoGe-2 the P from relative depth is not computed
    if os.path.isfile(da) and not os.path.isfile(os.path.join(carpeta, "position_moge.exr")) and (not os.path.isfile(pp) or os.path.getmtime(pp) < os.path.getmtime(da)):
        # Camera-space Position (P) + distance (depth.Z) for ZDefocus, as float EXR; the PNG is only a preview (each channel stretched over
        # non-sky). Rebuilt if depth is newer.
        P, Z = posicion(da, W, H, carpeta)
        _exr(os.path.join(carpeta, "position.exr"), P)
        _exr(os.path.join(carpeta, "depth_z.exr"), Z)
        cielo = Z >= CAMARA["lejos"] - 1e-3
        ref = P[~cielo] if (~cielo).sum() > 100 else P.reshape(-1, 3)
        lo, hi = np.percentile(ref, 2, 0), np.percentile(ref, 98, 0)
        v = np.clip((P - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        v[cielo] *= 0.3
        cv2.imwrite(os.path.join(carpeta, "position.png"), cv2.cvtColor((v * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    for n in PASES_CPU:
        if os.path.isfile(os.path.join(carpeta, n + ".png")):
            hechos[n] = os.path.join(carpeta, n + ".png")
    return hechos


def _paralaje_listo(c):
    try:
        with open(os.path.join(c, "paralaje.json"), encoding="utf-8") as f:
            return json.load(f).get("estado") == "listo"
    except (OSError, ValueError):
        return False


def _elegidos(L, d):
    """BGs of the batch to extract passes from: all, or only those in d["bgs"] (stems)."""
    solo = set(d.get("bgs") or [])
    return [b for b in L["bgs"] if not solo or b["stem"] in solo]


def _fuente_pases(L, motor):
    """Which outputs passes can be made from: any engine of the batch, the original and the tiled 4K of MOSS · depth (the 2K upscaled
    to 4K goes into the passes): the models see a reduced copy (copia_pases) that already carries the tile detail, and each pass returns
    to the exact 4K size."""
    return motor in L["motores"] or motor == "original" or (motor == "moss_prof_4k" and "moss_prof" in L["motores"])


async def lanzar_pases(d):
    lote, motor = str(d.get("lote") or ""), str(d.get("motor") or "")
    L = _leer_lote(lote)
    if not L:
        raise ValueError("batch not found")
    if not _fuente_pases(L, motor):
        raise ValueError("that engine is not in the batch")
    _reanudar(lote)                                          # a new launch: a previous cancel of this batch no longer applies
    pases = [p for p in (d.get("pases") or PASES_GPU) if p in PASES_GPU] or list(PASES_GPU)
    avisos = []
    if "hdr" in pases and not capacidades()["hdr_ltx"]:      # Beauty HDR needs the LTX module (not bundled)
        pases = [p for p in pases if p != "hdr"] or ["depth"]
        avisos.append("Beauty HDR: " + LTX_NO)
    try:
        n = max(2, min(8, int(d.get("capas") or 4)))
    except (TypeError, ValueError):
        n = 4
    texto = str(d.get("texto") or "").strip()[:2000]   # empty = Qwen3.8 describes each BG (grafo_capas)
    trabajos = 0
    import aiohttp
    async with aiohttp.ClientSession() as s:
        for p in pases:                                      # grouped by pass: one model swap per pass
            for b in _elegidos(L, d):
                beauty = _salida(lote, motor, b["stem"])
                if not beauty:
                    avisos.append("%s: no %s output yet" % (b["nombre"], MOTORES.get(motor, motor)))
                    continue
                img = "bgpases_%s_%s_%s.png" % (lote, motor, b["stem"])
                copia_pases(beauty, img)
                pref = "renderpasses/%s/pases_%s/%s/" % (lote, motor, b["stem"])
                if p in ("objetos", "cripto") and rp_psd.fuente(_dir_pases(lote, motor, b["stem"])):
                    # the plate has the artist's PSD or mask (rp_psd): Cryptomatte and Objects come from its layers, not the AI (one task)
                    _capas_fuente_en_fondo(lote, _dir_pases(lote, motor, b["stem"]), beauty)
                    trabajos += 1
                    continue
                if p == "objetos":                           # per-object layers: Qwen lists → SAM3 cuts → Klein fills (background task)
                    for f in glob.glob(os.path.join(_dir_pases(lote, motor, b["stem"]), "obj_*.png")) + [os.path.join(_dir_pases(lote, motor, b["stem"]), x) for x in ("objetos.json", "fondo_limpio.png")]:
                        if os.path.isfile(f):
                            os.remove(f)
                    _en_fondo(lote, _capas_objetos(img, pref, _dir_pases(lote, motor, b["stem"]), beauty))
                    trabajos += 1
                    continue
                if p == "cripto":                            # Cryptomatte: Qwen names everything → SAM3 → standard EXR (background task)
                    for f in [os.path.join(_dir_pases(lote, motor, b["stem"]), x) for x in ("cripto.json", "cryptomatte.exr", "cryptomatte.png", "cryptomatte_material.png", "cripto_cielo.png")]:
                        if os.path.isfile(f):
                            os.remove(f)
                    _en_fondo(lote, _cryptomatte(img, pref, _dir_pases(lote, motor, b["stem"]), beauty, str(d.get("cripto_guia") or "").strip()))
                    trabajos += 1
                    continue
                if p == "hdr":                               # real beauty HDR with the LTX 2.5 SDR → HDR effect (background task)
                    _en_fondo(lote, _beauty_hdr(beauty, _dir_pases(lote, motor, b["stem"]), pref))
                    trabajos += 1
                    continue
                if p == "moge":                              # P in metres + lens with MoGe-3/MoGe-2 (RenderPassesMoGeEXR node)
                    dp = _dir_pases(lote, motor, b["stem"])
                    for f in ("moge.json", "position_moge.exr", "position_moge.png", "depth_moge.exr", "depth_moge.png", "normales_moge.exr", "cielo_moge.png"):
                        if os.path.isfile(os.path.join(dp, f)):
                            os.remove(os.path.join(dp, f))
                    async with s.post("http://127.0.0.1:%s/prompt" % comun.puerto(), json={"prompt": comun.modelos_os(grafo_moge(img, pref.rstrip("/")))}) as q:
                        r = await q.json()
                    if r.get("prompt_id"):
                        trabajos += 1
                    else:
                        avisos.append("MoGe · %s: ComfyUI rejected the graph (%s)" % (b["nombre"], str(r.get("error") or r)[:200]))
                    continue
                if p == "position":                          # computed from depth on CPU: delete so it is rebuilt; queue depth if missing
                    dp = _dir_pases(lote, motor, b["stem"])
                    for f in ("position.png", "position.exr", "depth_z.exr"):
                        if os.path.isfile(os.path.join(dp, f)):
                            os.remove(os.path.join(dp, f))
                    if not _fuente_depth(dp) and "depth" not in pases:
                        from PIL import Image
                        with Image.open(beauty) as im:
                            lado_bg = min(im.size)
                        os.makedirs(dp, exist_ok=True)
                        n_, av_ = await _encolar_depth(s, img, pref, lado_bg, dp, "moge" in pases)
                        trabajos += n_
                        avisos += [a for a in av_ if a not in avisos]
                        pases = pases + ["depth"]
                    trabajos += 1                            # rebuilt on CPU when the passes are polled
                    continue
                if p == "paralaje":                          # depth planes with hidden areas filled by Klein (background task)
                    for f in glob.glob(os.path.join(_dir_pases(lote, motor, b["stem"]), "plano_*.png")) + [os.path.join(_dir_pases(lote, motor, b["stem"]), "paralaje.json")]:
                        if os.path.isfile(f):
                            os.remove(f)
                    _en_fondo(lote, _paralaje(img, pref, _dir_pases(lote, motor, b["stem"]), beauty))
                    trabajos += 1
                    continue
                if p == "materiales":                        # Qwen says which materials exist, SAM3 runs only for those (background task)
                    for f in glob.glob(os.path.join(_dir_pases(lote, motor, b["stem"]), "mat_*.png")) + [os.path.join(_dir_pases(lote, motor, b["stem"]), "materiales.json")]:
                        if os.path.isfile(f):
                            os.remove(f)                     # rebuilt: old ones may be false positives (SAM3 without Qwen)
                    _en_fondo(lote, _materiales_con_qwen(img, pref, _dir_pases(lote, motor, b["stem"])))
                    trabajos += 1
                    continue
                else:
                    if p == "capas":
                        from PIL import Image
                        with Image.open(os.path.join(INPUT, img)) as im:   # the <= 4 MP copy (SeedVR2 on a huge image would hang ComfyUI)
                            tam_bg = im.size
                        for f in glob.glob(os.path.join(_dir_pases(lote, motor, b["stem"]), "capa_*.png")):
                            os.remove(f)                     # rebuilt (with their full-resolution version)
                    if p == "capas_stable":
                        from PIL import Image
                        with Image.open(os.path.join(INPUT, img)) as im:   # the <= 4 MP copy (SeedVR2 on a huge image would hang ComfyUI)
                            tam_bg = im.size
                        for f in glob.glob(os.path.join(_dir_pases(lote, motor, b["stem"]), "scapa_*.png")):
                            os.remove(f)
                    if p == "depth":                          # rebuild depth, its refined version and fog (derived from it)
                        from PIL import Image
                        with Image.open(beauty) as im:
                            lado_bg = min(im.size)
                        dp = _dir_pases(lote, motor, b["stem"])
                        os.makedirs(dp, exist_ok=True)
                        for f in glob.glob(os.path.join(dp, "depth_[0-9]*_.png")) + [os.path.join(dp, x) for x in ("depth_afinado.png", "niebla.png", "depth_z.exr", DEPTH_JOB)]:
                            if os.path.isfile(f):
                                os.remove(f)
                        # Depth Anything V2 when its node pack is installed (preferred); otherwise MoGe (core nodes)
                        n_, av_ = await _encolar_depth(s, img, pref, lado_bg, dp, "moge" in pases)
                        trabajos += n_
                        avisos += [a for a in av_ if a not in avisos]
                        continue
                    Gs = [grafo_capas(img, pref + "capa", n, texto, hd=tam_bg) if p == "capas"
                          else grafo_capas_stable(img, pref + "scapa", n, hd=tam_bg) if p == "capas_stable" else grafo_marigold(p, img, pref + p)]
                for G in Gs:
                    async with s.post("http://127.0.0.1:%s/prompt" % comun.puerto(), json={"prompt": comun.modelos_os(G)}) as q:
                        r = await q.json()
                    if r.get("prompt_id"):
                        trabajos += 1
                    else:
                        avisos.append("%s · %s: ComfyUI rejected the graph (%s)" % (p, b["nombre"], str(r.get("error") or r)[:200]))
    os.makedirs(_dir_pases(lote, motor), exist_ok=True)
    ruta = os.path.join(_dir_pases(lote, motor), "pases.json")
    try:                                                     # MERGED with what was there (AI passes included), not overwritten
        with open(ruta, encoding="utf-8") as f:
            P = json.load(f)
    except (OSError, ValueError):
        P = {"motor": motor, "pases": []}
    P.update(pases=[x for x in PASES_GPU if x in set(P.get("pases", [])) | set(pases)], capas=n, texto=texto)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(P, f, ensure_ascii=False)
    if "albedo" in pases or "depth" in pases:
        avisos.append("Shading, shadows and fog are computed automatically once albedo and depth finish (estimates from 2D, not render AOVs).")
    return {"trabajos": trabajos, "avisos": avisos}


# "AI" passes: an edit model turns the final BG into a compositing pass. Each instruction says what is visible, the pass background colour
# and that geometry must not move (otherwise the model re-renders the scene). Still estimates, not AOVs. Asking for "a pass" does not work;
# asking for EVERYTHING to become WHITE CLAY (ground, rocks, plants, props, characters) does.
_GEOM = " Keep exactly the same framing, camera, geometry, shapes and positions as the original image."
_ARCILLA = "every surface, including the ground, rocks, plants, props, creatures and characters, becomes the same plain white matte clay with no color and no texture"
PASES_IA = {
    "arcilla": ("Clay · same light", "Re-render this scene as a clay render: " + _ARCILLA + ", lit by exactly the same light as the original "
                "(same sun direction, same cast shadows, same sky light)." + _GEOM),
    "relleno": ("Fill / GI", "Re-render this scene as a clay render lit ONLY by soft sky light and bounce light, with no direct sun and no cast "
                "shadows: " + _ARCILLA + "." + _GEOM),
    "oclusion": ("Occlusion", "Re-render this scene as an ambient occlusion clay render: " + _ARCILLA + ", lit by uniform white light from every "
                 "direction with no direct light and no cast shadows; only crevices, corners and contact areas get softly darker." + _GEOM),
    # Atmosphere and sky mattes failed as AI passes: fog comes from depth on CPU, sky from Cryptomatte.
    # SSS (experimental): black, with only light passing through thin translucent parts. Neither engine gives true SSS (Fibo gives more of a
    # backlit rim), so it runs on Klein (and MOSS when chosen).
    "sss": ("SSS","Turn this image into a subsurface scattering pass for compositing: a pure black image where ONLY the light "
            "that passes through thin translucent parts appears — leaves, petals, ears, fingers, skin, thin membranes and feathers lit from "
            "behind — glowing in their warm transmitted color, strongest where the material is thinnest and the light is behind it. Opaque "
            "surfaces, the ground, the sky and everything that is not translucent are pure black." + _GEOM, ("klein3",)),
    "mate_objetos": ("Characters & props matte", "Keep only the characters, creatures and props of this image in their original colors, exactly where "
                     "they are, and turn everything else (ground, rocks, plants, sea and sky) into pure black." + _GEOM),
    # A V-Ray-style AI shadow matte failed with both engines; the shadow pass is "clay − fill" (like RawShadow).
    # Specular works with BOTH engines: Fibo returns the beauty with wet, shiny surfaces (highlights isolated on CPU, see derivados_ia);
    # Klein returns highlights on black.
    # SAM3 does not recognise stylised plants, so Klein (good at character mattes) does the plants matte.
    "mate_plantas": ("Plants matte", "Keep only the plants of this image — cacti, bushes, grass, leaves and trees — in their original colors, exactly "
                     "where they are, and turn everything else (ground, rocks, sand, sea, sky, props, creatures and characters) into pure black." + _GEOM),
    # REFLECTION and RIM passes failed with both engines and were removed.
    # SHADOWS: asking a model for "the shadow pass" fails; REMOVING shadows works. The pass is computed on CPU: beauty ÷ "shadow-free" in
    # linear, normalised and edge-aware filtered (derivados_ia → "sombras_ia").
    "sin_sombras": ("Shadows · via shadow removal", "Remove every cast shadow and shade from this image: show the same scene evenly lit by soft "
                    "bright daylight from every direction, so there are no shadows under objects, no shadows on the ground and no dark shaded "
                    "sides. Keep exactly the same framing, objects, shapes, positions, colors, textures and painted art style; only the shadows "
                    "disappear."),
    # EMISSION: asked for as a MATTE (what Klein does well), not as a "pass": whatever emits its own light.
    "emision": ("Emission", "Keep only the parts of this image that emit their own light — glowing eyes, lamps, lanterns, fire, embers, screens, "
                "lit windows, neon, the sun and magical glows — in their original bright glowing colors, exactly where they are, and turn "
                "everything else, including surfaces that only reflect light, into pure black." + _GEOM),
    "especular": ("Specular", "Convert this image into a specular pass for compositing: a black image where only the shiny highlights and "
                  "reflections on glossy surfaces (metal, wet stone, water, leaves) appear, in white; everything else stays black. Keep exactly "
                  "the same framing, geometry, shapes and positions as the original image; output only this pass, nothing else.", ("klein3",)),
}
EDITORES_IA = ("klein3", "fibo2")
_editores = lambda p: PASES_IA[p][2] if len(PASES_IA[p]) > 2 else ("klein3",)   # which engine does each pass well
# MOSS (with depth control) runs the light passes only in batches where MOSS was chosen as an engine (so regular batches do not take
# longer). It is aligned and very good at clay, occlusion and specular; on mattes it relights the scene ("keep only X, rest black" is not
# understood), its emission invents a sun and its shadow removal is not usable.
MOSS_PASES = ("arcilla", "relleno", "oclusion", "especular", "sss")
_editores_lote = lambda p, L: _editores(p) + (("moss",) if p in MOSS_PASES and {"moss", "moss_prof"} & set((L or {}).get("motores", ())) else ())
# Key and shadows are NOT asked from Klein (it does not isolate the pass): they come from two aligned clay renders on CPU,
# "same light" − "fill" = direct sun (key); where the key does not reach, among faces the fill lights, there is shadow.


def alinear(beauty, ruta):
    """Edit models can shift the image: registers an AI pass against the beauty (affine ECC on EDGES, which clay or a matte share with the
    beauty even if not its colour) and fixes it in place. Once per file (alineados.json). If it does not converge or the motion is absurd
    (> 4 % scale or > 3 % shift) it is left untouched. Returns a note."""
    import cv2
    import numpy as np
    carpeta = os.path.dirname(ruta)
    reg = os.path.join(carpeta, "alineados.json")
    try:
        with open(reg, encoding="utf-8") as f:
            hechos = json.load(f)
    except (OSError, ValueError):
        hechos = {}
    n = os.path.basename(ruta)
    if n in hechos:
        return hechos[n]
    P = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    B = cv2.imread(beauty, cv2.IMREAD_GRAYSCALE)
    H, W = P.shape[:2]
    def bordes(g):
        g = cv2.resize(g, (1024, max(1, round(1024 * H / W))), interpolation=cv2.INTER_AREA).astype(np.float32)
        e = cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 2)
        return e / max(float(e.max()), 1e-6)
    g = P if P.ndim == 2 else cv2.cvtColor(P[..., :3], cv2.COLOR_BGR2GRAY)
    nota = "unchanged"
    try:
        w = np.eye(2, 3, dtype=np.float32)
        _, w = cv2.findTransformECC(bordes(cv2.resize(B, (W, H))), bordes(g), w, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        k = W / 1024.0
        esc = (np.hypot(w[0, 0], w[1, 0]), np.hypot(w[0, 1], w[1, 1]))
        dx, dy = w[0, 2] * k, w[1, 2] * k
        if max(abs(esc[0] - 1), abs(esc[1] - 1)) <= 0.04 and abs(dx) <= 0.03 * W and abs(dy) <= 0.03 * H:
            M = w.copy(); M[:, 2] *= k
            if max(abs(esc[0] - 1), abs(esc[1] - 1)) > 0.0005 or abs(dx) > 0.25 or abs(dy) > 0.25:
                Q = cv2.warpAffine(P, M, (W, H), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                cv2.imwrite(ruta, Q)
                nota = "aligned: scale %.3f × %.3f, shift %+.1f, %+.1f px" % (esc[0], esc[1], dx, dy)
        else:
            nota = "not aligned (odd motion: scale %.3f × %.3f, %+.0f, %+.0f px)" % (esc[0], esc[1], dx, dy)
    except cv2.error:
        nota = "not aligned (did not converge)"
    hechos[n] = nota
    with open(reg, "w", encoding="utf-8") as f:
        json.dump(hechos, f, ensure_ascii=False, indent=1)
    return nota


PEDIDO_PRESENCIA = ("Look at this image and answer with exactly two words separated by a comma, nothing else. First word: yes or no, are there "
                    "any plants visible (trees, bushes, grass, leaves, cacti, flowers)? Second word: yes or no, are there any characters visible "
                    "(people, creatures, animals, animated characters) or loose props held or lying around?")


async def _presencia(img, carpeta):
    """Qwen says whether there are plants and characters/props; without them those mattes are not requested (Klein would return something
    else). Saved to presencia.json in the BG folder."""
    f = os.path.join(carpeta, "presencia.json")
    try:
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        pass
    G = {"i": {"class_type": "LoadImage", "inputs": {"image": img}},
         "e": {"class_type": "ImageScaleToMaxDimension", "inputs": {"image": ["i", 0], "upscale_method": "lanczos", "largest_size": 768}},
         "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN38, "type": "stable_diffusion", "device": "default"}},
         "g": {"class_type": "TextGenerate", "inputs": {"clip": ["c", 0], "prompt": PEDIDO_PRESENCIA, "image": ["e", 0], "max_length": 10,
                                                       "sampling_mode": "off", "thinking": False, "use_default_template": True, "mtp": "auto", "system_prompt": ""}},
         "p": {"class_type": "PreviewAny", "inputs": {"source": ["g", 0]}}}
    t = (await _correr_texto(G)).lower()
    partes = [x.strip() for x in re.split(r"[,\s]+", t) if x.strip() in ("yes", "no")]
    d = {"plantas": len(partes) > 0 and partes[0] == "yes", "personajes": len(partes) > 1 and partes[1] == "yes", "qwen": t[:40]}
    os.makedirs(carpeta, exist_ok=True)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    return d


def emision_color(beauty, k_ruta, destino):
    """Klein finds WHERE there is emitted light but changes its colour and leaks. Its output is used only as a MASK: what Klein marked strongly
    AND is brighter than its surroundings in the original, plus bright pixels of the same hue as detected ones. Colour comes from the ORIGINAL."""
    import cv2
    import numpy as np
    B = cv2.imread(beauty, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    H, W = B.shape[:2]
    K = cv2.resize(cv2.imread(k_ruta, cv2.IMREAD_COLOR), (W, H)).astype(np.float32) / 255.0
    lum = lambda x: 0.0722 * x[..., 0] + 0.7152 * x[..., 1] + 0.2126 * x[..., 2]
    mk = np.clip((cv2.dilate(lum(K), np.ones((7, 7), np.uint8)) - 0.22) / 0.2, 0, 1)
    Lb, V = lum(B), B.max(-1)
    brilla = np.clip((Lb - cv2.GaussianBlur(Lb, (0, 0), W / 60) - 0.04) / 0.10, 0, 1) * np.clip((V - 0.5) / 0.2, 0, 1)
    m0 = mk * brilla
    hsv = cv2.cvtColor((B * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, ss = hsv[..., 0], hsv[..., 1] / 255.0
    sel = (m0 > 0.5) & (ss > 0.35)
    m = m0
    if sel.sum() > 30:
        hist = np.bincount(hh[sel].astype(int), minlength=180).astype(np.float32)
        hist = np.convolve(np.r_[hist[-6:], hist, hist[:6]], np.ones(7) / 7, "same")[6:-6]
        cerca = np.zeros_like(hh, bool)
        for t in [i for i in range(180) if hist[i] > 0.2 * hist.max()]:
            d = np.abs(hh - t); cerca |= np.minimum(d, 180 - d) <= 6
        m = np.maximum(m0, brilla * cerca * np.clip((ss - 0.35) / 0.2, 0, 1))
    # emitters are small spots; large "emitting" regions (> 1.5 % of the image each) are bright surfaces, not lights, and are dropped
    n_, cc, st, _ = cv2.connectedComponentsWithStats((m > 0.35).astype(np.uint8), connectivity=8)
    for c_ in range(1, n_):
        if st[c_, cv2.CC_STAT_AREA] > 0.015 * H * W:
            m[cc == c_] = 0
    if (m > 0.5).mean() > 0.04:                              # many small spots adding up to > 4 %: bright surfaces, not lights
        m[:] = 0
    m = np.clip(cv2.dilate(cv2.GaussianBlur(m, (0, 0), 1.2), np.ones((3, 3), np.uint8)) * 1.3, 0, 1)
    cv2.imwrite(destino, (B * m[..., None] * 255 + 0.5).astype(np.uint8))


def derivados_ia(carpeta, editor, beauty=None):
    """CPU passes derived from an editor's AI passes: shadows (clay ÷ fill), specular highlights, ground reflections, shadows from shadow
    removal and emission in original colour."""
    import cv2
    import numpy as np
    arc, rel = _ultimo(carpeta, "ia_%s_arcilla" % editor), _ultimo(carpeta, "ia_%s_relleno" % editor)
    hechos = {}
    # SHADOWS: clay ÷ fill ratio in linear, NO blur (passes are already aligned to the beauty), normalised by its median over surfaces; the
    # sky and a 3 px band around it stay white. White = no shadow, black = shadow (multiply in comp). "v2" in the name: old blurred
    # versions are not reused.
    fs = os.path.join(carpeta, "ia_%s_sombras_v2.png" % editor)
    if arc and rel and not os.path.isfile(fs):
        lin = lambda a: np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        a = lin(cv2.imread(arc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
        r = lin(cv2.resize(cv2.imread(rel, cv2.IMREAD_GRAYSCALE), (a.shape[1], a.shape[0])).astype(np.float32) / 255.0)
        q = a / np.maximum(r, 1e-3)
        fuera = np.zeros(a.shape, bool)
        cielo = os.path.join(carpeta, "cripto_cielo.png") if os.path.isfile(os.path.join(carpeta, "cripto_cielo.png")) else _ultimo(carpeta, "mat_cielo")   # sky comes from Cryptomatte (CryptoMaterial "sky")
        if cielo:
            c = cv2.resize(cv2.imread(cielo, cv2.IMREAD_GRAYSCALE), (a.shape[1], a.shape[0])) > 127
            fuera = cv2.dilate(c.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        med = float(np.median(q[~fuera])) if (~fuera).any() else float(np.median(q))
        sm = np.clip((q / max(med, 1e-4) - 0.55) / 0.25, 0, 1)
        sm[fuera] = 1.0
        cv2.imwrite(fs, (sm * 255 + 0.5).astype(np.uint8))
    fe = os.path.join(carpeta, "ia_%s_especular_brillos.png" % editor)
    esp = _ultimo(carpeta, "ia_%s_especular" % editor)
    if editor == "fibo2" and esp and not os.path.isfile(fe):
        # Fibo specular HIGHLIGHTS isolated: white top-hat (r = 15 px) of linear luminance = what stands out from its immediate surroundings.
        # "Fibo − beauty" is noisy (Fibo shifts the overall tone); r = 7 is too small for large surfaces.
        lin = lambda a: np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        e = cv2.imread(esp, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        L = lin(0.0722 * e[..., 0] + 0.7152 * e[..., 1] + 0.2126 * e[..., 2]).astype(np.float32)
        th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        th = np.clip(th / max(np.percentile(th, 99.7), 1e-4), 0, 1)
        cv2.imwrite(fe, (th ** (1 / 2.2) * 255).astype(np.uint8))
    # Reflections are not only on the ground (sometimes there is none): the Klein or MOSS specular is split in two, on any surface:
    # HIGHLIGHTS = what is small and sharp and stands out from its surroundings (morphological opening, ~1/140 of the width) and
    # REFLECTIONS = what is wide and continuous (a wet path, a puddle, glass, a polished wall). highlights + reflections = specular.
    fr = os.path.join(carpeta, "ia_%s_reflejos.png" % editor)
    fb = os.path.join(carpeta, "ia_%s_brillos.png" % editor)
    if editor != "fibo2" and esp and (not os.path.isfile(fr) or os.path.getmtime(fr) < os.path.getmtime(esp)):
        E_ = cv2.imread(esp, cv2.IMREAD_COLOR).astype(np.float32)
        r_ = max(3, E_.shape[1] // 140)
        ancho = cv2.morphologyEx(E_, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1,) * 2))
        ancho = cv2.GaussianBlur(ancho, (0, 0), 1.0)
        cv2.imwrite(fr, np.clip(ancho, 0, 255).astype(np.uint8))
        cv2.imwrite(fb, np.clip(E_ - ancho, 0, 255).astype(np.uint8))
    fz = os.path.join(carpeta, "ia_%s_sombras_ia.png" % editor)
    ss = _ultimo(carpeta, "ia_%s_sin_sombras" % editor)
    if ss and beauty and (not os.path.isfile(fz) or os.path.getmtime(fz) < os.path.getmtime(ss)):
        Bq = cv2.imread(beauty, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        Hq, Wq = Bq.shape[:2]
        Kq = cv2.resize(cv2.imread(ss, cv2.IMREAD_COLOR), (Wq, Hq), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
        lin_ = lambda a: np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        r_ = cv2.GaussianBlur(lin_(Bq).mean(-1), (0, 0), 1.0) / np.maximum(cv2.GaussianBlur(lin_(Kq).mean(-1), (0, 0), 1.0), 1e-3)
        # lit = white and densest shadow = black (otherwise everything came out light grey)
        lo_, hi_ = float(np.percentile(r_, 2)), float(np.percentile(r_, 80))
        r_ = np.clip((r_ - lo_) / max(hi_ - lo_, 1e-3), 0, 1.2).astype(np.float32)
        g_ = cv2.cvtColor((Bq * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        r_ = _guiado(g_, r_, max(4, Wq // 150), 2e-3)            # remove texture noise, keep the shadow edge sharp
        for cf_ in ("cripto_cielo.png", "cielo_moge.png"):     # the sky has no shadows (Klein brightens it, which came out black)
            if os.path.isfile(os.path.join(carpeta, cf_)):
                cq = cv2.resize(cv2.imread(os.path.join(carpeta, cf_), cv2.IMREAD_GRAYSCALE), (Wq, Hq)).astype(np.float32) / 255.0
                r_ = np.maximum(r_, cv2.GaussianBlur(cq, (0, 0), 2))
                break
        cv2.imwrite(fz, (np.clip(r_, 0, 1) * 255 + 0.5).astype(np.uint8))
    fm = os.path.join(carpeta, "ia_%s_emision_color.png" % editor)
    em = _ultimo(carpeta, "ia_%s_emision" % editor)
    if em and beauty and (not os.path.isfile(fm) or os.path.getmtime(fm) < os.path.getmtime(em)):
        emision_color(beauty, em, fm)
    for n, f in (("sombras", fs), ("especular_brillos", fe), ("brillos", fb), ("reflejos", fr), ("emision_color", fm if editor != "moss" else ""), ("sombras_ia", fz if editor != "moss" else "")):
        if os.path.isfile(f):
            hechos[n] = f
    return hechos


def _beauty_de(carpeta):
    """Beauty of a passes folder (…/<batch>/pases_<engine>/<stem>) = that engine's output in the batch."""
    stem, motor, lote = os.path.basename(carpeta), os.path.basename(os.path.dirname(carpeta))[len("pases_"):], os.path.basename(os.path.dirname(os.path.dirname(carpeta)))
    return _salida(lote, motor, stem)


async def lanzar_pases_ia(d):
    lote, motor = str(d.get("lote") or ""), str(d.get("motor") or "")
    L = _leer_lote(lote)
    if not L:
        raise ValueError("batch not found")
    if not _fuente_pases(L, motor):
        raise ValueError("that engine is not in the batch")
    _reanudar(lote)                                          # a new launch: a previous cancel of this batch no longer applies
    pases = [p for p in (d.get("pases") or PASES_IA) if p in PASES_IA] or list(PASES_IA)
    editores = [m for m in (d.get("editores") or EDITORES_IA + (("moss",) if {"moss", "moss_prof"} & set(L["motores"]) else ())) if m in ("klein3", "fibo2", "krea2v2", "moss")]
    if not d.get("editores"):                                # by default each pass with the engine that does it well (Fibo: specular only)
        pares = [(m, p) for m in editores for p in pases if m in _editores_lote(p, L)]
    else:
        pares = [(m, p) for m in editores for p in pases]
    semilla = int(d.get("semilla") or 42)
    trabajos, avisos = 0, []
    import aiohttp
    async with aiohttp.ClientSession() as s:
        for m in editores:                                   # grouped by engine: one model swap per engine
            for b in _elegidos(L, d):
                beauty = _salida(lote, motor, b["stem"])
                if not beauty:
                    avisos.append("%s: no %s output yet" % (b["nombre"], MOTORES.get(motor, motor)))
                    continue
                img = "bgpases_%s_%s_%s.png" % (lote, motor, b["stem"])
                copia_pases(beauty, img)
                from PIL import Image
                with Image.open(beauty) as im:
                    wh = (im.size[0] // 16 * 16, im.size[1] // 16 * 16) if im.size[0] % 16 or im.size[1] % 16 else im.size
                    tam_b = im.size
                for p in [p for mm, p in pares if mm == m]:
                    if p in ("mate_plantas", "mate_objetos"):
                        try:
                            hay = await _presencia(img, _dir_pases(lote, motor, b["stem"]))
                        except Exception as e:
                            hay = {"plantas": True, "personajes": True}
                            print("[RenderPasses] bglook presencia:", e)
                        if not hay["plantas" if p == "mate_plantas" else "personajes"]:
                            avisos.append("%s: no %s found (Qwen): matte skipped" % (b["nombre"], "plants" if p == "mate_plantas" else "characters or props"))
                            continue
                    try:
                        # All passes match the beauty size: Fibo direct if the batch was made that way; otherwise the engine at its size plus a
                        # lanczos to the EXACT beauty size (no SeedVR2: it is a pass). MOSS uses MoGe depth control (creativity 0) so the pass
                        # stays aligned.
                        G = _prefijo(armar("moss_prof" if m == "moss" else m, PASES_IA[p][1], img, semilla, tam=wh if (m == "fibo2" and L.get("fibo_directo")) else None, creatividad=0.0),
                                     "renderpasses/%s/pases_%s/%s/ia_%s_%s" % (lote, motor, b["stem"], m, p))
                        a_resolucion(G, img, tam_b, semilla, limpio=True)
                    except ValueError as e:
                        avisos.append("%s · %s: %s" % (MOTORES[m], b["nombre"], e))
                        continue
                    async with s.post("http://127.0.0.1:%s/prompt" % comun.puerto(), json={"prompt": comun.modelos_os(G)}) as q:
                        r = await q.json()
                    if r.get("prompt_id"):
                        trabajos += 1
                    else:
                        avisos.append("%s · %s: ComfyUI rejected the graph (%s)" % (m, p, str(r.get("error") or r)[:200]))
    os.makedirs(_dir_pases(lote, motor), exist_ok=True)
    ruta = os.path.join(_dir_pases(lote, motor), "pases.json")
    try:
        with open(ruta, encoding="utf-8") as f:
            P = json.load(f)
    except (OSError, ValueError):
        P = {"motor": motor, "pases": []}
    P["ia"] = sorted({tuple(x) for x in P.get("ia", [])} | set(pares))
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(P, f, ensure_ascii=False)
    return {"trabajos": trabajos, "avisos": avisos}


@R.post("/renderpasses/pases_ia")
async def r_pases_ia(request):
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    try:
        return web.json_response(await lanzar_pases_ia(await request.json()))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)


def estado_pases(lote, motor):
    L = _leer_lote(lote)
    if not L or (not _fuente_pases(L, motor)):
        return None
    try:
        with open(os.path.join(_dir_pases(lote, motor), "pases.json"), encoding="utf-8") as f:
            P = json.load(f)
    except (OSError, ValueError):
        return {"lote": lote, "motor": motor, "lanzado": False, "bgs": [], "hechos": 0, "total": 0}
    bgs, hechos, total = [], 0, 0
    for b in L["bgs"]:
        c = _dir_pases(lote, motor, b["stem"])
        beauty = _salida(lote, motor, b["stem"])
        sal = {}
        plx = []                                             # parallax planes for the viewer
        moge = {}
        hdr_info = {}
        try:
            with open(os.path.join(c, "luz.json"), encoding="utf-8") as fh:
                luz = json.load(fh)
        except (OSError, ValueError):
            luz = {}
        for p in P["pases"] + (["cripto"] if "cripto" not in P["pases"] and rp_psd.es_psd(c) else []):   # a PSD Cryptomatte shows even if not asked for
            if p == "position":                              # not a GPU pass: derived from depth (derivados)
                continue
            total += 1
            if p == "capas":
                caps = sorted(glob.glob(os.path.join(c, "capa_[0-9]*.png")))
                if caps:
                    hechos += 1
                    from PIL import Image
                    try:   # layers at BG resolution (alpha snapped to the beauty, visible colour from the beauty, hidden parts via SeedVR2)
                        full = capas_hd(c, beauty) if beauty else []
                    except Exception as e:
                        print("[RenderPasses] bglook capas_hd:", e)
                        full = []
                    if len(full) == len(caps):
                        caps = full
                    for i, r in enumerate(caps):
                        try:   # Qwen-Image-Layered sometimes leaves layers unused: empty ones (alpha 0) are not shown
                            with Image.open(r) as im:
                                if im.mode == "RGBA" and im.getchannel("A").getextrema()[1] < 8:
                                    continue
                        except OSError:
                            pass
                        sal["Layer %d" % (i + 1)] = _url(r, "output")
            elif p == "capas_stable":
                caps = sorted(glob.glob(os.path.join(c, "scapa_[0-9]*.png")))
                if caps:
                    hechos += 1
                    try:   # at full resolution, like the Qwen layers
                        full = capas_hd(c, beauty, "scapa") if beauty else []
                    except Exception as e:
                        print("[RenderPasses] bglook capas_hd stable:", e)
                        full = []
                    if len(full) == len(caps):
                        caps = full
                for i, r in enumerate(caps):
                    sal["Layer %d · Stable-Layers%s" % (i + 1, " · background" if i == 0 else "")] = _url(r, "output")
            elif p == "objetos":
                try:
                    with open(os.path.join(c, "objetos.json"), encoding="utf-8") as f:
                        O = json.load(f)
                except (OSError, ValueError):
                    O = {"estado": "queued…", "objetos": []}
                if _terminado(O.get("estado")):
                    hechos += 1
                for o in O.get("objetos", []):
                    r = os.path.join(c, o["archivo"])
                    if os.path.isfile(r):
                        sal["Object · " + o["nombre"]] = _url(r, "output")
                if os.path.isfile(os.path.join(c, "fondo_limpio.png")):
                    sal["Clean plate (no objects)"] = _url(os.path.join(c, "fondo_limpio.png"), "output")
                if O.get("estado") != "listo":
                    sal["_objetos"] = O.get("estado")
            elif p == "hdr":
                try:
                    with open(os.path.join(c, "hdr.json"), encoding="utf-8") as f:
                        O = json.load(f)
                except (OSError, ValueError):
                    O = {"estado": "queued…"}
                if _terminado(O.get("estado")):
                    hechos += 1
                if os.path.isfile(os.path.join(c, "beauty_hdr.png")):
                    sal["beauty_hdr"] = _url(os.path.join(c, "beauty_hdr.png"), "output")
                if O.get("estado") != "listo":
                    sal["_hdr"] = O.get("estado")
                else:
                    hdr_info = O
            elif p == "moge":
                if os.path.isfile(os.path.join(c, "moge.json")):
                    hechos += 1
                    try:
                        with open(os.path.join(c, "moge.json"), encoding="utf-8") as f:
                            moge = json.load(f)
                    except (OSError, ValueError):
                        moge = {}
                    for k, f in (("moge_position", "position_moge.png"), ("moge_depth", "depth_moge.png")):
                        if os.path.isfile(os.path.join(c, f)):
                            sal[k] = _url(os.path.join(c, f), "output")
            elif p == "paralaje":
                try:
                    with open(os.path.join(c, "paralaje.json"), encoding="utf-8") as f:
                        O = json.load(f)
                except (OSError, ValueError):
                    O = {"estado": "queued…", "planos": []}
                if _terminado(O.get("estado")):
                    hechos += 1
                for o in O.get("planos", []):
                    r = os.path.join(c, o["archivo"])
                    if os.path.isfile(r):
                        sal["Parallax · plane %d · %s" % (o["n"], o["nombre"])] = _url(r, "output")
                        plx.append({"url": _url(r, "output"), "nombre": o["nombre"], "z": o.get("z")})
                if O.get("estado") != "listo":
                    sal["_paralaje"] = O.get("estado")
            elif p == "cripto":
                try:
                    with open(os.path.join(c, "cripto.json"), encoding="utf-8") as f:
                        O = json.load(f)
                except (OSError, ValueError):
                    O = {"estado": "queued…", "elementos": []}
                if O.get("fuente") in ("psd", "mascara"):    # from the artist's PSD or mask (rp_psd): "from PSD · N layers" on the card
                    sal["_cripto_fuente"] = O["fuente"]
                    sal["_cripto_capas"] = len(O.get("objetos") or [])
                    sal["_cripto_psd"] = O.get("psd") or ""
                if _terminado(O.get("estado")):
                    hechos += 1
                if os.path.isfile(os.path.join(c, "cryptomatte.png")):
                    sal["cripto"] = _url(os.path.join(c, "cryptomatte.png"), "output")
                if os.path.isfile(os.path.join(c, "cryptomatte_material.png")):
                    sal["cripto_material"] = _url(os.path.join(c, "cryptomatte_material.png"), "output")
                if O.get("estado") != "listo":
                    sal["_cripto"] = O.get("estado")
                elif O.get("elementos"):
                    sal["_cripto_nombres"] = ", ".join(O["elementos"])
            elif p == "materiales":
                try:   # Qwen said what is there (materiales.json); done when those mattes exist
                    with open(os.path.join(c, "materiales.json"), encoding="utf-8") as f:
                        presentes = json.load(f)["presentes"]
                except (OSError, ValueError, KeyError):
                    presentes = None
                encontrados = [(k, _ultimo(c, "mat_" + k)) for k in (presentes if presentes is not None else MATERIALES)]
                if presentes is not None and all(r for _, r in encontrados):
                    hechos += 1
                for k, r in encontrados:
                    if r:
                        try:
                            r = _mate_fino(r, beauty)
                        except Exception as e:
                            print("[RenderPasses] bglook mate fino:", e)
                        sal["Matte · " + MATERIALES[k][0]] = _url(r, "output")
            else:
                r = _fuente_depth(c) if p == "depth" else _ultimo(c, p)
                if r:
                    hechos += 1
                    sal[p] = _url(r, "output")
                elif p == "depth" and _depth_fallido(c):
                    hechos += 1
                    sal["_depth"] = "error: " + _depth_fallido(c)
        try:
            with open(os.path.join(c, "presencia.json"), encoding="utf-8") as fh:
                hay = json.load(fh)
        except (OSError, ValueError):
            hay = {}
        for m, p in P.get("ia", []):
            if p not in PASES_IA or (m not in _editores(p) and not (m == "moss" and p in MOSS_PASES)):   # leftovers of old engine/pass combos
                continue
            if (p == "mate_plantas" and hay.get("plantas") is False) or (p == "mate_objetos" and hay.get("personajes") is False):
                continue                                     # Qwen said there is none → that matte is not shown (if it existed, it was false)
            total += 1
            r = _ultimo(c, "ia_%s_%s" % (m, p))
            if r and beauty:
                try:
                    alinear(beauty, r)
                except Exception as e:                       # a failed alignment must not break the status
                    sal["_error"] = "alinear: %s" % str(e)[:150]
            if r:
                hechos += 1
                if p != "sin_sombras":                       # an internal step; only "Shadows (beauty ÷ shadow-free)" is shown
                    sal["AI %s · %s" % (MOTORES.get(m, m), PASES_IA[p][0])] = _url(r, "output")
        for m in sorted({m for m, _ in P.get("ia", []) if m != "fibo2"}):   # Fibo no longer makes AI passes (worse results)
            try:
                for n, r in derivados_ia(c, m, beauty).items():
                    sal["AI %s · %s" % (MOTORES.get(m, m), {"sombras": "Shadows (sharp)",
                                                             "especular_brillos": "Specular · highlights only", "reflejos": "Reflections", "brillos": "Highlights", "emision_color": "Emission · original colour", "sombras_ia": "Shadows (beauty ÷ shadow-free)"}[n])] = _url(r, "output")
            except Exception as e:
                sal["_error"] = str(e)[:200]
        if beauty and os.path.isdir(c):
            try:
                for n, r in derivados(c, beauty).items():
                    sal[n] = _url(r, "output")
                if "depth_afinado" in sal:
                    sal.pop("depth", None)                   # the refined depth replaces the raw Depth Anything one
            except Exception as e:                       # a failing CPU pass must not break the status
                sal["_error"] = str(e)[:200]
        pases_4k = {}                                        # 4K versions already done (4k/<pass>_4k.png)
        for k, u in sal.items():
            r = None if k.startswith("_") else _ruta_de_url(u)
            if r:
                f4 = os.path.join(os.path.dirname(r), "4k", re.sub(r"_\d{5}_$", "", os.path.splitext(os.path.basename(r))[0]) + "_4k.png")
                if os.path.isfile(f4):
                    pases_4k[k] = _url(f4, "output")
        if "moge_position" in sal:
            sal.pop("position", None)
        exr = {k: _url(os.path.join(c, f), "output") for k, f in (("raw_lighting", "raw_lighting.exr"), ("albedo", "albedo_lineal.exr"), ("cripto", "cryptomatte.exr"), ("cripto_material", "cryptomatte.exr"),
                                                                 ("position", "position.exr"), ("depth_afinado", "depth_z.exr"),
                                                                 ("moge_position", "position_moge.exr"), ("moge_depth", "depth_moge.exr"),
                                                                 ("beauty_hdr", "beauty_hdr.exr"))
               if k in sal and os.path.isfile(os.path.join(c, f))}
        hdr_pases = []
        for k, u in list(sal.items()):                       # the pass's LTX HDR, if already made
            r = None if k.startswith("_") or not PASES_LTX.search(k) else _ruta_de_url(u)
            if r and os.path.isfile(os.path.splitext(r)[0] + "_hdr_ltx.exr"):
                exr[k] = _url(os.path.splitext(r)[0] + "_hdr_ltx.exr", "output")
                hdr_pases.append(k)
        try:
            with open(os.path.join(c, "hdr_ltx.json"), encoding="utf-8") as f:
                e_ltx = json.load(f).get("estado", "")
            if e_ltx != "listo":
                sal["_hdr_ltx"] = e_ltx
        except (OSError, ValueError):
            pass
        if beauty and os.path.isfile(os.path.join(c, "beauty_hdr.exr")):
            for k, u in list(sal.items()):
                if k in hdr_pases:
                    continue
                r = None if k.startswith("_") or not PASES_HDR.search(k) else _ruta_de_url(u)
                if r:
                    try:
                        h_ = pase_hdr(r, c, beauty)
                        if h_:
                            exr[k] = _url(h_, "output")
                            hdr_pases.append(k)
                    except Exception as e:
                        sal["_error"] = "HDR of %s: %s" % (k, str(e)[:150])
        nuke = {}
        if beauty and os.path.isdir(c):
            try:
                nuke = {k: _url(v, "output") for k, v in paquete_nuke(c, beauty).items()}
                nuke["carpeta"] = c
            except Exception as e:
                sal["_error"] = "Nuke: %s" % str(e)[:200]
        bgs.append({"nombre": b["nombre"], "stem": b["stem"], "beauty": _url(beauty, "input" if motor == "original" else "output") if beauty else None, "original": _url(os.path.join(INPUT, b["input"]), "input"),
                    "pases": sal, "pases_4k": pases_4k, "exr": exr, "nuke": nuke, "paralaje": plx if _paralaje_listo(c) else [], "moge": moge, "luz": luz, "hdr": hdr_info, "hdr_pases": hdr_pases, "splat": _splat_de(c), "carpeta": c})
    return {"lote": lote, "motor": motor, "lanzado": True, "pases": P["pases"], "bgs": bgs, "hechos": hechos, "total": total,
            "salida": _dir_pases(lote, motor)}


_CAPAS_EN_CURSO = set()


def _capas_fuente_en_fondo(lote, carpeta, beauty):
    """Cryptomatte + Objects + per-layer mattes from the plate's saved PSD / mask, in the background (once, even if both passes ask)."""
    if carpeta in _CAPAS_EN_CURSO:
        return

    async def tarea():
        _CAPAS_EN_CURSO.add(carpeta)
        try:
            await asyncio.to_thread(rp_psd.capas_desde_fuente, carpeta, beauty, escribir_cryptomatte)
        except Exception:
            pass                                             # already in cripto.json / objetos.json
        finally:
            _CAPAS_EN_CURSO.discard(carpeta)
    _en_fondo(lote, tarea())


@R.post("/renderpasses/cripto_psd")
async def r_cripto_psd(request):
    """Cryptomatte from the artist's layered PSD (rp_psd): multipart {psd, lote, motor, bg} or JSON {lote, motor, bg, ruta} (local only)."""
    return await rp_psd.ruta(request, {"leer_lote": _leer_lote, "salida": _salida, "dir_pases": _dir_pases, "escribir": escribir_cryptomatte,
                                       "es_local": comun.es_local, "en_fondo": _en_fondo, "motores": MOTORES, "raiz": DIR_SAL})


@R.post("/renderpasses/cancelar")
async def r_cancelar(request):
    """{lote} → removes THAT batch's queued jobs (those saving to renderpasses/<batch>/ or reading its bglook_/bgpases_ copies) and
    interrupts the running one if it belongs to the batch. Other batches are untouched."""
    import aiohttp
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    d = await request.json()
    lote = str(d.get("lote") or "")
    if not _RX_LOTE.match(lote):
        return web.json_response({"error": "invalid batch"}, status=400)
    # First stop the batch's background tasks (Cryptomatte, objects, parallax, MOSS tiles, HDR, 4K, per-plate prompts) so they queue nothing
    # more; they write "cancelled" to their json at their next step. A new launch for this batch clears the flag.
    _CANCELADO[lote] = time.monotonic()
    marca = re.compile(r"(renderpasses/%s/|bglook_%s_|bgpases_%s_)" % (lote, lote, lote))
    base = "http://127.0.0.1:%s" % comun.puerto()
    es_del_lote = lambda g: bool(marca.search(json.dumps(g)))
    async with aiohttp.ClientSession() as s:
        async with s.get(base + "/queue") as q:
            Q = await q.json()
        borrar = [it[1] for it in Q.get("queue_pending", []) if es_del_lote(it[2])]
        corta = [it[1] for it in Q.get("queue_running", []) if es_del_lote(it[2])]
        Q2 = None
        async with s.get(base + "/queue") as q:
            Q2 = await q.json()
        quedan = [it[1] for it in Q2.get("queue_pending", []) if it[1] in borrar]       # those still pending on re-read
        if quedan:
            async with s.post(base + "/queue", json={"delete": quedan}) as q:
                await q.read()
        if any(it[1] in corta for it in Q2.get("queue_running", [])):
            async with s.post(base + "/interrupt", json={"prompt_id": corta[0]}) as q:
                await q.read()
        # short pause: after an interrupt, the next job of the batch may start right then
        await asyncio.sleep(1.5)
        # second sweep: a task may have queued a job between the first read and the flag taking effect (it checks every 2 s)
        async with s.get(base + "/queue") as q:
            Q3 = await q.json()
        tarde = [it[1] for it in Q3.get("queue_pending", []) if es_del_lote(it[2]) and it[1] not in borrar]
        if tarde:
            async with s.post(base + "/queue", json={"delete": tarde}) as q:
                await q.read()
    return web.json_response({"quitados": len(borrar) + len(tarde), "cortado": len(corta)})


@R.post("/renderpasses/pases")
async def r_pases(request):
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    try:
        return web.json_response(await lanzar_pases(await request.json()))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)


_RX_DATO_EXR = re.compile(r"(depth|normal|cavidad|niebla|sombra|oclusion|mate|cripto|position|suelo|objmask|mat_|alpha|cielo)", re.I)


@R.get("/renderpasses/exr")
async def r_exr(request):
    """Every pass can be exported as EXR: any PNG pass → float EXR (zip) next to it, once. Colour passes (beauty, clay, specular, SSS,
    emission, planes, layers…) go from sRGB to linear; data passes (depth, normals, mattes, cavity, shadows…) keep their 0–1 values.
    Alpha is kept. u = /view URL of the PNG."""
    import cv2
    import numpy as np
    r = _ruta_de_url(request.query.get("u", ""))
    if not r or not r.lower().endswith(".png"):
        return web.json_response({"error": "pass not found"}, status=404)
    destino = os.path.splitext(r)[0] + ".exr"

    def convierte():
        a = cv2.imread(r, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError("could not read the pass")
        a = a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)
        if a.ndim == 2:
            out = a
        else:
            rgb = cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2RGB)
            if not _RX_DATO_EXR.search(os.path.basename(r)):
                rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            out = np.dstack([rgb, a[..., 3]]) if a.shape[2] == 4 else rgb
        _exr(destino, out.astype(np.float32))
    if not os.path.isfile(destino) or os.path.getmtime(destino) < os.path.getmtime(r):
        await asyncio.to_thread(convierte)
    return web.FileResponse(destino, headers={"Content-Disposition": 'attachment; filename="%s"' % os.path.basename(destino)})


# ───────── Compo HDR: a node-based HDR compositing test on the page. The browser composites in WebGL2 with float textures (values above 1
# really show); the server sends each EXR as raw float16, all at the beauty HDR size, and writes the final EXR at full resolution with the
# same parameters. Nodes: Beauty HDR → (+ pass · gain · floor) × N → exposure → output.
def _exr_rgb(ruta, wh=None):
    import cv2
    import numpy as np
    _, a = _leer_exr(ruta)
    if a is None:
        raise ValueError("could not read %s" % os.path.basename(ruta))
    a = a[..., :3] if a.ndim == 3 and a.shape[2] >= 3 else np.repeat((a if a.ndim == 2 else a[..., :1]).reshape(a.shape[0], a.shape[1], 1), 3, -1)
    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=6e4, neginf=0.0)
    if wh and (a.shape[1], a.shape[0]) != tuple(wh):
        a = cv2.resize(a, tuple(wh), interpolation=cv2.INTER_AREA if wh[0] < a.shape[1] else cv2.INTER_LINEAR)
    return a


def _ruta_exr(u):
    from urllib.parse import urlparse, parse_qs
    q = parse_qs(urlparse(u).query)
    p = os.path.normpath(os.path.join(OUTPUT, (q.get("subfolder") or [""])[0], (q.get("filename") or [""])[0]))
    return p if p.lower().endswith(".exr") and comun.dentro(p, DIR_SAL) and os.path.isfile(p) else None


@R.get("/renderpasses/exr_bin")
async def r_exr_bin(request):
    """u = /view URL of an EXR; base = that of the beauty HDR (for the size); max = long side for the viewer → raw float16 RGB (X-W, X-H)."""
    import numpy as np
    r, rb = _ruta_exr(request.query.get("u", "")), _ruta_exr(request.query.get("base", "") or request.query.get("u", ""))
    if not r or not rb:
        return web.json_response({"error": "EXR not found"}, status=404)
    try:
        lado = max(256, min(4096, int(request.query.get("max") or 2048)))
    except ValueError:
        lado = 2048

    def lee():
        import OpenImageIO as oiio
        i = oiio.ImageInput.open(rb)
        sp = i.spec(); i.close()
        k = min(1.0, lado / float(max(sp.width, sp.height)))
        wh = (max(1, round(sp.width * k)), max(1, round(sp.height * k)))
        return wh, _exr_rgb(r, wh).astype(np.float16).tobytes()
    try:
        (w, h), datos = await asyncio.to_thread(lee)
    except Exception as e:
        return web.json_response({"error": str(e)[:200]}, status=500)
    return web.Response(body=datos, content_type="application/octet-stream", headers={"X-W": str(w), "X-H": str(h), "Cache-Control": "no-store"})


@R.post("/renderpasses/compo_hdr")
async def r_compo_hdr(request):
    """{base, capas: [{u, ganancia, piso, on}], ev} → comp_hdr.exr (ACEScg, float) next to the beauty HDR, at its full resolution."""
    import numpy as np
    d = await request.json()
    rb = _ruta_exr(str(d.get("base") or ""))
    if not rb:
        return web.json_response({"error": "beauty HDR not found"}, status=404)

    def compone():
        B = _exr_rgb(rb)
        wh = (B.shape[1], B.shape[0])
        out = B.copy()
        usadas = []
        for c in d.get("capas") or []:
            r = _ruta_exr(str(c.get("u") or ""))
            if not r or not c.get("on"):
                continue
            g, piso = float(c.get("ganancia", 1.0)), float(c.get("piso", 0.0))
            out += g * np.maximum(_exr_rgb(r, wh) - piso, 0.0)
            usadas.append({"pase": os.path.basename(r), "ganancia": g, "piso": piso})
        out *= 2.0 ** float(d.get("ev") or 0.0)
        dest = os.path.join(os.path.dirname(rb), "comp_hdr.exr")
        _exr(dest, out.astype(np.float32))
        with open(os.path.join(os.path.dirname(rb), "comp_hdr.json"), "w", encoding="utf-8") as f:
            json.dump({"base": os.path.basename(rb), "capas": usadas, "ev": float(d.get("ev") or 0.0)}, f, ensure_ascii=False, indent=1)
        return dest, float(np.percentile(out.max(-1), 99.9)), float((out.max(-1) > 1).mean() * 100)
    try:
        dest, p999, sobre = await asyncio.to_thread(compone)
    except Exception as e:
        return web.json_response({"error": str(e)[:200]}, status=500)
    return web.json_response({"ok": True, "url": _url(dest, "output"), "p999": round(p999, 2), "sobre_1": round(sobre, 2)})


@R.get("/renderpasses/paralaje_geo")
async def r_paralaje_geo(request):
    """Geometry of each parallax plane to project its image onto, like Project3D in Nuke. MoGe-2 distance (metres) where the plane is its
    own; elsewhere (covered by planes in front, or belonging to another plane) it is filled from its own edge (depth inpaint) so the ground
    behind an object continues the ground. u = URL of plano_1.png.
    Returns: JSON header (w, h, fx, fy, cx, cy, planos) + float32 [planes × h × w] distances."""
    import cv2
    import numpy as np
    r = _ruta_de_url(request.query.get("u", ""))
    c = os.path.dirname(r) if r else ""
    try:
        with open(os.path.join(c, "moge.json"), encoding="utf-8") as f:
            M = json.load(f)
        with open(os.path.join(c, "paralaje.json"), encoding="utf-8") as f:
            PJ = json.load(f)
    except (OSError, ValueError):
        return web.json_response({"error": "this BG has no MoGe-2 or parallax planes"}, status=404)

    def calcula():
        _, z = _leer_exr(os.path.join(c, "depth_moge.exr"))
        z = z[..., 0]
        H0, W0 = z.shape
        w = 400
        h = max(2, round(H0 * w / W0))
        Z = cv2.resize(z, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        planos = sorted(PJ["planos"], key=lambda p: p["n"])
        lab = np.full((h, w), len(planos) - 1, np.int32)
        for j, p in reversed(list(enumerate(planos[:-1]))):          # back to front: the front one wins
            A = cv2.imread(os.path.join(c, p["archivo"]), cv2.IMREAD_UNCHANGED)
            if A is None or A.ndim < 3 or A.shape[2] < 4:
                continue
            a = cv2.resize(A[..., 3], (w, h), interpolation=cv2.INTER_AREA) > 127
            lab[a] = j
        out = []
        lz = np.log(np.maximum(Z, 1e-3))
        for j in range(len(planos)):
            suyo = lab == j
            if suyo.sum() < 20:
                out.append(Z); continue
            # areas covered by front planes are filled from this plane's edge; areas behind do not matter (alpha 0) but are filled too
            m = (~suyo).astype(np.uint8)
            # Navier-Stokes, clipped to this plane's own range: OpenCV 5's Telea returns garbage (~1e37) on float32
            f = np.clip(cv2.inpaint(lz.astype(np.float32), m, 5, cv2.INPAINT_NS), float(lz[suyo].min()), float(lz[suyo].max()))
            # smooth: depth jumps inside one plane stretch the mesh into "hairs"; treat it as a continuous curved surface
            f = cv2.GaussianBlur(f, (0, 0), max(2.0, w / 90))
            out.append(np.exp(f).astype(np.float32))
        return w, h, np.stack(out)
    w, h, D = await asyncio.to_thread(calcula)
    cab = json.dumps({"w": w, "h": h, "fx": M["fx"], "fy": M["fy"], "cx": M["cx"], "cy": M["cy"], "fov_x": M["fov_x"],
                      "planos": [{"url": _url(os.path.join(c, p["archivo"]), "output"), "nombre": p["nombre"], "z": p.get("z")}
                                 for p in sorted(PJ["planos"], key=lambda p: p["n"])]}).encode()
    cuerpo = len(cab).to_bytes(4, "little") + cab + b"\0" * ((4 - (4 + len(cab)) % 4) % 4) + np.ascontiguousarray(D, np.float32).tobytes()
    return web.Response(body=cuerpo, content_type="application/octet-stream", headers={"Cache-Control": "no-store"})


# ───────── Layered Gaussian splat ─────────
# POST /renderpasses/splat {lote, motor, bgs?: [stems], densidad?, escala?} → for each plate whose parallax is finished (and has MoGe),
# rp_splat builds the layered 3D Gaussian splat and writes splat.ply into its pass folder (CPU, ~10 s per 2K plate).
# → {ok, hechos: [{bg, stem, url, splats}], saltados: [{bg, motivo}]}
def _splat_de(c):
    try:
        with open(os.path.join(c, "splat.json"), encoding="utf-8") as f:
            n = json.load(f).get("splats")
    except (OSError, ValueError):
        n = None
    r = os.path.join(c, "splat.ply")
    return {"url": _url(r, "output"), "splats": n} if os.path.isfile(r) else None


@R.post("/renderpasses/splat")
async def r_splat(request):
    from . import rp_splat
    try:
        d = await request.json()
        densidad, escala = float(d.get("densidad") or 0.625), float(d.get("escala") or 1.0)
    except (ValueError, TypeError):
        return web.json_response({"error": "bad request"}, status=400)
    lote, motor = str(d.get("lote") or ""), str(d.get("motor") or "")
    L = _leer_lote(lote) if _RX_LOTE.match(lote) else None
    if not L or not re.match(r"^[a-z0-9_]+$", motor) or not _fuente_pases(L, motor):
        return web.json_response({"error": "batch or engine not found"}, status=404)
    hechos, saltados = [], []
    for b in _elegidos(L, d):
        c = _dir_pases(lote, motor, b["stem"])
        if not _paralaje_listo(c):
            saltados.append({"bg": b["nombre"], "motivo": "parallax planes not finished"})
            continue
        if not (os.path.isfile(os.path.join(c, "depth_moge.exr")) and os.path.isfile(os.path.join(c, "moge.json"))):
            saltados.append({"bg": b["nombre"], "motivo": "needs the MoGe pass (metric depth and lens)"})
            continue
        try:
            ruta, info = await asyncio.to_thread(rp_splat.guardar_ply, c, densidad, escala)
            hechos.append({"bg": b["nombre"], "stem": b["stem"], "url": _url(ruta, "output"), "splats": info["splats"]})
        except Exception as e:
            saltados.append({"bg": b["nombre"], "motivo": str(e)[:200]})
    return web.json_response({"ok": bool(hechos), "hechos": hechos, "saltados": saltados})


# ───────── Live 3D splat (relightable) ─────────
# POST builds the plate's relightable splat scene from its passes (rp_splat3d) in the background; GET /renderpasses/splat3d/<batch>/<engine>/<bg>/
# opens the PBR Director (web/splat_director.html) and serves it, by relative path, the files of <pass folder>/splat3d/ (PLY, lens, floor,
# assets, glb/). The page polls estado.json until it reads "listo".
_SPLAT3D_EN_CURSO = set()


@R.post("/renderpasses/splat3d")
async def r_splat3d(request):
    """{lote, motor, bg (stem), rehacer?} → {ok, url, listo} (local requests only)."""
    if not comun.es_local(request):
        return web.json_response({"error": "local requests only"}, status=403)
    try:
        d = await request.json()
    except ValueError:
        return web.json_response({"error": "bad request"}, status=400)
    lote, motor, stem = (str(d.get(k) or "") for k in ("lote", "motor", "bg"))
    L = _leer_lote(lote) if _RX_LOTE.match(lote) else None
    if not L or not re.match(r"^[a-z0-9_]+$", motor) or not _fuente_pases(L, motor) or stem not in [b["stem"] for b in L.get("bgs", [])]:
        return web.json_response({"error": "plate not found in this batch"}, status=404)
    carpeta = _dir_pases(lote, motor, stem)
    from urllib.parse import quote
    url = "/renderpasses/splat3d/%s/%s/%s/" % (lote, motor, quote(stem))
    hecho = os.path.isfile(os.path.join(carpeta, "splat3d", "splat_pbr.ply"))
    if (hecho and not d.get("rehacer")) or carpeta in _SPLAT3D_EN_CURSO:
        return web.json_response({"ok": True, "url": url, "listo": hecho and carpeta not in _SPLAT3D_EN_CURSO})
    from . import rp_splat3d

    async def tarea():
        _SPLAT3D_EN_CURSO.add(carpeta)
        try:
            await asyncio.to_thread(rp_splat3d.construir, carpeta)
        except Exception:
            pass                                      # already in splat3d/estado.json
        finally:
            _SPLAT3D_EN_CURSO.discard(carpeta)
    _SPLAT3D_EN_CURSO.add(carpeta)
    _en_fondo(lote, tarea())
    return web.json_response({"ok": True, "url": url, "listo": False})


@R.get(r"/renderpasses/splat3d/{lote}/{motor}/{bg}/{ruta:.*}")
async def r_splat3d_archivo(request):
    lote, motor, stem, ruta = (request.match_info[k] for k in ("lote", "motor", "bg", "ruta"))
    if not _RX_LOTE.match(lote) or not re.fullmatch(r"[a-z0-9_]+", motor) or not re.fullmatch(r"[\w\-. ]+", stem) or stem in (".", ".."):
        return web.Response(status=404)
    if ruta in ("", "index.html"):
        return web.FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "splat_director.html"),
                                headers={"Cache-Control": "no-store"})
    base = os.path.realpath(os.path.join(_dir_pases(lote, motor, stem), "splat3d"))
    f = os.path.realpath(os.path.join(base, ruta))
    if not f.startswith(base + os.sep) or not os.path.isfile(f):
        return web.Response(status=404)
    return web.FileResponse(f, headers={"Cache-Control": "no-store"})


@R.get("/renderpasses/position_bin")
async def r_position_bin(request):
    """position.exr values (X Y Z per pixel, float32) reduced to <= 640 px wide, so the 3D viewer can place each pixel and re-shoot it with a
    moving camera (like PositionToPoints + ScanlineRender in Nuke). u = URL of the same BG's position.png."""
    import cv2
    import numpy as np
    r = _ruta_de_url(request.query.get("u", ""))
    exr = os.path.join(os.path.dirname(r), "position_moge.exr" if r and os.path.basename(r) == "position_moge.png" else "position.exr") if r else ""
    if not exr or not os.path.isfile(exr):
        return web.json_response({"error": "no position.exr for that BG"}, status=404)
    sp, a = await asyncio.to_thread(_leer_exr, exr)
    if a is None:
        return web.json_response({"error": "could not read position.exr"}, status=500)
    a = a[..., :3]
    h, w = a.shape[:2]
    if w > 640:
        a = cv2.resize(a, (640, max(2, round(h * 640 / w))), interpolation=cv2.INTER_NEAREST)   # no blending across edges
    h, w = a.shape[:2]
    fov = CAMARA["fov"]
    if exr.endswith("position_moge.exr"):                    # MoGe provides its own lens
        try:
            with open(os.path.join(os.path.dirname(exr), "moge.json"), encoding="utf-8") as f:
                fov = json.load(f).get("fov_x") or fov
        except (OSError, ValueError):
            pass
    return web.Response(body=np.ascontiguousarray(a, np.float32).tobytes(), content_type="application/octet-stream",
                        headers={"X-W": str(w), "X-H": str(h), "X-Fov": str(fov), "Cache-Control": "no-store"})


@R.get("/renderpasses/pases/estado")
async def r_pases_estado(request):
    lote, motor = request.query.get("lote", ""), request.query.get("motor", "")
    L = _leer_lote(lote) if _RX_LOTE.match(lote) else None
    for b in (L or {}).get("bgs", []):                       # Depth Anything failed for a plate → MoGe as its depth (queued once)
        dp = _dir_pases(lote, motor, b["stem"])
        if os.path.isfile(os.path.join(dp, DEPTH_JOB)):
            await _asegurar_depth(dp)
    E = estado_pases(lote, motor)
    return web.json_response(E) if E else web.json_response({"error": "batch or engine not found"}, status=404)



# ───────── Install check ─────────
# GET /renderpasses/requisitos → for every engine and pass: the model files that are missing (with the ComfyUI folder they go in) and the
# node classes that are missing. The lists are taken from the graph builders themselves (each one is built with placeholder inputs and
# walked), so they always match what is really queued; only "any of these" alternatives are declared by hand. The page greys out what
# cannot run. Optional extras (SeedVR2 for upscaling, the VLM description of the layers…) are reported as notes and never block.
_X = "rp_requisitos.png"                                     # placeholder image name: the builders do not read it
# Node classes that come from a custom node pack (everything else ships with ComfyUI 0.43+)
PAQUETE_NODO = {"Krea2EditModelPatch": "comfyui-krea2edit", "Krea2EditGroundedEncode": "comfyui-krea2edit",
                "DepthAnythingV2Preprocessor": "comfyui_controlnet_aux"}


def _uso_grafo(G):
    """(node classes, [(folders, model name)]) used by an API graph."""
    clases, modelos = set(), []
    for n in G.values():
        if not isinstance(n, dict) or "class_type" not in n:
            continue
        clases.add(n["class_type"])
        for k, v in (n.get("inputs") or {}).items():
            cs = comun.carpetas_de(n["class_type"], k) if isinstance(v, str) and v else ()
            if cs:
                modelos.append((cs, v))
    return clases, modelos


def _grupos_requisitos():
    """Requirement groups → {"nodos": set, "modelos": [(folders, [name, alternatives…])]}, read off the real graph builders."""
    def g(*grafos, alt=None):
        nodos, modelos = set(), []
        for G in grafos:
            c, m = _uso_grafo(G)
            nodos |= c
            for cs, v in m:
                par = (cs, list((alt or {}).get(v) or [v]))
                if par not in modelos:
                    modelos.append(par)
        return {"nodos": nodos, "modelos": modelos}
    moge = {m: list(MODELOS_MOGE) for m in MODELOS_MOGE}     # any MoGe-3 / MoGe-2 works
    km, fb, ES = _mod("klein_movil_srv"), _mod("fibo2_srv"), _mod("escalar_imagen_srv")
    t = (1024, 1024)
    return {
        "krea2": g(grafo_krea2("x", _X, t, 1)),
        "krea2v2": g(grafo_krea2v2("x", _X, t, 1)),
        "moss": g(grafo_moss("x", _X, t, 1, comprobar=False)),
        "moss_prof": g(grafo_moss_prof("x", _X, t, 1, mapa=_X, comprobar=False), alt=moge),
        "klein3": g(km._em().grafo_klein3("x", [_X], "x", t, 1, modelo=km.BASE_UNET)),
        # Fibo's builder picks the first file it finds among several (int8 or bf16; the edit text encoder in bf16 or fp8)
        "fibo2": {"nodos": {"UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeFibo2", "KSampler", "VAEDecode", "LoadImage", "SaveImage"},
                  "modelos": [(("diffusion_models",), list(fb.UNET_EDIT)), (("text_encoders",), list(fb.CLIP_EDITAR)), (("vae",), [fb.VAE])]},
        "depth_da": g(grafo_depth(_X, "x")),
        "moge": g(grafo_moge(_X, "x"), alt=moge),
        "normals": g(grafo_marigold("normals", _X, "x")),
        "albedo": g(grafo_marigold("albedo", _X, "x")),
        "capas": g(grafo_capas(_X, "x", texto="x")),
        "capas_stable": g(grafo_capas_stable(_X, "x")),
        "vlm": g(grafo_prompt_bg(_X)),
        "sam3": g(grafo_sam3(_X, "x", ["a", "b"]), grafo_sam3_cajas(_X, "x", [(0, 0, 10, 10)], 100, 100)),
        "seedvr2": g(ES.grafo(_X, 2.0, 1, "x")),
    }


_DEPTH = ("depth_da", "moge")                                 # depth: Depth Anything V2 (preferred) OR MoGe (fallback)
# item → (required groups, optional groups); a tuple inside "required" means "any one of these"
COMPOSICION_MOTORES = {"krea2": (["krea2"], []), "krea2v2": (["krea2v2"], []), "moss": (["moss"], []), "moss_prof": (["moss_prof"], []),
                       "klein3": (["klein3"], []), "fibo2": (["fibo2"], ["seedvr2"])}
COMPOSICION_PASES = {"depth": ([_DEPTH], []), "position": ([_DEPTH], []), "moge": (["moge"], []), "hdr": (["ltx"], []),
                     "normals": (["normals"], []), "albedo": (["albedo"], []), "objetos": (["vlm", "sam3", "klein3"], []),
                     "capas_stable": (["capas_stable"], ["seedvr2"]), "capas": (["capas"], ["vlm", "seedvr2"]), "cripto": (["vlm", "sam3"], []),
                     "paralaje": ([_DEPTH, "klein3"], ["moge", "vlm", "sam3"]), "materiales": (["vlm", "sam3"], [])}
COMPOSICION_EXTRA = {"escalar4k": (["seedvr2"], []), "prompt_qwen": (["vlm"], []), "traduccion": (["traduccion"], []), "crear": (["crear"], [])}
NOTA_GRUPO = {"seedvr2": "SeedVR2 (upscaling above ×1.5 and the 4K switch)", "vlm": "the local Qwen VLM (scene description)",
              "moge": "MoGe (distances in meters)", "sam3": "SAM 3 (whole elements per plane, with Cryptomatte)",
              "moss_prof": "MOSS · depth (also runs this pass in MOSS batches)", "ltx": LTX_NO}


def _ruta_carpeta(c):
    try:
        import folder_paths
        r = folder_paths.get_folder_paths(c)
        return r[0] if r else None
    except Exception:
        return None


def requisitos(listar=None, clases=None, modulos=None):
    """Install check. listar(folder) → model list (default: ComfyUI's); clases → available node classes (default: nodes.NODE_CLASS_MAPPINGS;
    if that cannot be read, nodes are not checked); modulos(name) → whether a Python package can be imported."""
    import importlib.util
    listas = {}

    def lista(c):
        if c not in listas:
            listas[c] = list((listar or comun.lista_modelos)(c))
        return listas[c]
    if clases is None:
        try:
            import nodes
            clases = set(nodes.NODE_CLASS_MAPPINGS)
        except Exception:
            clases = None
    modulos = modulos or (lambda n: importlib.util.find_spec(n) is not None)
    grupos = _grupos_requisitos()

    def evalua(nombre):
        if nombre == "ltx":
            ok = _mod("ltx_efectos_srv") is not None
            return {"ok": ok, "faltan_modelos": [], "faltan_nodos": [], "faltan_paquetes": [], "nota": "" if ok else LTX_NO}
        if nombre == "crear":                               # the Krea 2 Turbo or any Krea 2 fine-tune, see modelos_crear
            try:
                modelos_crear(lista)
                return {"ok": True, "faltan_modelos": [], "faltan_nodos": [], "faltan_paquetes": []}
            except ValueError:
                dm, te = lista("diffusion_models"), lista("text_encoders")
                fm = [{"carpeta": c, "archivo": alt[0], "alternativas": list(alt[1:]), "ruta": _ruta_carpeta(c)}
                      for c, alt in (("diffusion_models", KREA2_T2I_UNET[1:2] + KREA2_T2I_UNET[:1]), ("text_encoders", KREA2_T2I_CLIP), ("vae", (KREA2_VAE,)))
                      if not any(comun.hay_modelo((c,), a, lista) for a in alt)
                      and not (c == "diffusion_models" and any(re.search(r"krea[-_ ]?2", os.path.basename(x), re.I) for x in dm))]
                return {"ok": not fm, "faltan_modelos": fm, "faltan_nodos": [], "faltan_paquetes": []}
        if nombre == "traduccion":
            faltan = [p for p in ("transformers", "sentencepiece") if not modulos(p)]
            return {"ok": not faltan, "faltan_modelos": [], "faltan_nodos": [], "faltan_paquetes": faltan,
                    "nota": "pip install " + " ".join(faltan) if faltan else "MarianMT downloads its weights from Hugging Face on first use"}
        R = grupos[nombre]
        fm = [{"carpeta": cs[0], "archivo": alt[0], "alternativas": alt[1:], "ruta": _ruta_carpeta(cs[0])}
              for cs, alt in R["modelos"] if not any(comun.hay_modelo(cs, a, lista) for a in alt)]
        fn = sorted(c for c in R["nodos"] if clases is not None and c not in clases)
        return {"ok": not fm and not fn, "faltan_modelos": fm, "faltan_nodos": fn, "faltan_paquetes": []}
    G = {k: evalua(k) for k in list(grupos) + ["ltx", "traduccion", "crear"]}

    def item(necesita, opcional):
        r = {"ok": True, "faltan_modelos": [], "faltan_nodos": [], "faltan_paquetes": [], "notas": []}
        for n in necesita:
            if isinstance(n, tuple):                         # any one of: fine if one works; if none does, everything they lack
                ok = [x for x in n if G[x]["ok"]]
                if ok:
                    if n == _DEPTH and ok[0] != n[0]:
                        r["notas"].append("Depth Anything V2 (comfyui_controlnet_aux) is not available: depth comes from MoGe")
                    continue
                partes = [G[x] for x in n]
                if n == _DEPTH:
                    r["notas"].append("Needs Depth Anything V2 (comfyui_controlnet_aux) or MoGe")
            else:
                partes = [G[n]]
            for p in partes:
                if not p["ok"]:
                    r["ok"] = False
                    for k in ("faltan_modelos", "faltan_nodos", "faltan_paquetes"):
                        r[k] += [x for x in p[k] if x not in r[k]]
                    if p.get("nota"):
                        r["notas"].append(p["nota"])
        for n in opcional:
            if not G[n]["ok"]:
                r["notas"].append("Optional, missing: " + NOTA_GRUPO.get(n, n))
        r["instalar"] = sorted({PAQUETE_NODO.get(c, "ComfyUI 0.43 or newer") for c in r["faltan_nodos"]})
        return r
    motores = {k: item(*v) for k, v in COMPOSICION_MOTORES.items()}
    pases = {k: item(*v) for k, v in COMPOSICION_PASES.items()}
    for p in PASES_IA:                                       # AI passes: the editor that does each one well; MOSS and the VLM are extras
        pases[p] = item(list(_editores(p)), (["moss_prof"] if p in MOSS_PASES else []) + (["vlm"] if p in ("mate_plantas", "mate_objetos") else []))
    extra = {k: item(*v) for k, v in COMPOSICION_EXTRA.items()}
    no_motores = [k for k, v in motores.items() if not v["ok"]]
    no_pases = [k for k, v in pases.items() if not v["ok"] and k not in ("hdr", "position", "materiales")]   # HDR is hidden when missing;
    # position and materials have no card of their own
    familias = {"pases_gpu": [p for p in PASES_GPU if p != "hdr" and not pases[p]["ok"]], "pases_ia": [p for p in PASES_IA if not pases[p]["ok"]],
                "escalar4k": extra["escalar4k"]["ok"], "vlm": G["vlm"]["ok"], "moge": G["moge"]["ok"], "traduccion": extra["traduccion"]["ok"],
                "crear": extra["crear"]["ok"]}
    return {"ok": not no_motores and not no_pases, "motores": motores, "pases": pases, "extra": extra, "familias": familias,
            "capacidades": capacidades(), "nodos_comprobados": clases is not None,
            "resumen": {"motores_no": no_motores, "pases_no": no_pases, "total_no": len(no_motores) + len(no_pases)}}


def capacidades():
    """Optional features the page shows only when this installation has them."""
    return {"hdr_ltx": _mod("ltx_efectos_srv") is not None}


@R.get("/renderpasses/requisitos")
async def r_requisitos(request):
    try:
        return web.json_response(await asyncio.to_thread(requisitos))
    except Exception as e:
        return web.json_response({"error": "install check failed: %s" % e}, status=500)
