# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Shared server helpers:
#   - dentro(path, base): correct "is this file inside this folder" check (commonpath; a plain startswith would let a
#     sibling folder with the same prefix, e.g. output_old, through).
#   - CERROJO_CFG + escribir_spot(): every project-file write goes through the same lock and is atomic
#     (temp file + os.replace), so a full save never clobbers a half-written partial update.
#   - puerto(): ComfyUI's real port (never the Host header's, which may differ behind a proxy/VPN).
#   - es_local(request): whether the request comes from this same machine or from another machine on the network.
import json
import os
import shutil
import threading
import time

CERROJO_CFG = threading.Lock()
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def dentro(ruta, base):
    """True if `ruta` (resolved or relative) lies inside folder `base`."""
    try:
        return os.path.commonpath([os.path.abspath(ruta), os.path.abspath(base)]) == os.path.abspath(base)
    except ValueError:   # different drive on Windows
        return False


def resolver(base, rel):
    """Join base + rel (with / or \\) and return the absolute path only if it stays inside base; otherwise None."""
    ruta = os.path.normpath(os.path.join(base, str(rel or "").replace("\\", "/").replace("/", os.sep)))
    return ruta if dentro(ruta, base) else None


def leer_json(ruta, defecto=None):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return defecto


def escribir_json(ruta, datos):
    """Atomic write: temp file next to the target + os.replace (with retries: Windows sometimes keeps it open)."""
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    tmp = "%s.tmp_%d" % (ruta, threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=1)
    for _ in range(8):
        try:
            os.replace(tmp, ruta)
            return
        except PermissionError:
            time.sleep(0.06)
    shutil.copyfile(tmp, ruta)
    os.remove(tmp)


def escribir_spot(ruta, datos):
    """Save the project file (spot.json) under the shared lock with an atomic write."""
    with CERROJO_CFG:
        escribir_json(ruta, datos)


def puerto():
    """The port ComfyUI listens on (8000 if not known yet)."""
    try:
        from server import PromptServer
        return int(getattr(PromptServer.instance, "port", 0) or 8000)
    except Exception:
        return 8000


def es_local(request):
    """True if the request comes from this same machine (loopback)."""
    try:
        return (request.remote or "") in _LOOPBACK
    except Exception:
        return False


def nombre_limpio(s, largo=80):
    """File name without separators or paths: only letters, digits, hyphen, underscore and space."""
    import re
    return re.sub(r"[^\w\- ]+", "", str(s or "")).strip()[:largo]



def modelo_existente(carpeta, candidatos):
    """The first of `candidatos` that exists (as ComfyUI lists it, including subfolder); otherwise the first one."""
    L = lista_modelos(carpeta)
    for n in candidatos:
        hit = nombre_listado(n, L)
        if hit:
            return hit
    return candidatos[0]


# ───────── Model file names on any OS ─────────
# Graphs name models with a fixed string ("StableLayers\\stable_layers_comfy.safetensors"), but ComfyUI lists files with the OS separator
# and validates the EXACT string, so a Windows name fails on Linux/macOS (and a model kept in a subfolder fails everywhere). Every graph
# goes through modelos_os() right before it is posted to /prompt: each model-name input is rewritten to the string ComfyUI lists.
# Input name → models folder; some classes use a generic input name ("model_name") for their own folder.
CARPETA_ENTRADA = {"unet_name": ("diffusion_models",), "clip_name": ("text_encoders",), "clip_name1": ("text_encoders",),
                   "clip_name2": ("text_encoders",), "clip_name3": ("text_encoders",), "clip_name4": ("text_encoders",),
                   "vae_name": ("vae",), "lora_name": ("loras",), "ckpt_name": ("checkpoints",), "control_net_name": ("controlnet",),
                   "style_model_name": ("style_models",), "conditioning_name": ("conditioning", "embeddings")}
CARPETA_CLASE = {"LoadMoGeModel": {"model_name": ("geometry_estimation",)},
                 "UpscaleModelLoader": {"model_name": ("upscale_models",)},
                 "CLIPVisionLoader": {"clip_name": ("clip_vision",)},
                 "DepthAnythingV2Preprocessor": {"ckpt_name": ()}}   # comfyui_controlnet_aux downloads its own weights


def carpetas_de(clase, entrada):
    """ComfyUI model folders an input of a node class reads from (() = not a ComfyUI model folder)."""
    propio = CARPETA_CLASE.get(clase, {})
    if entrada in propio:
        return propio[entrada]
    return CARPETA_ENTRADA.get(entrada, ())


def lista_modelos(carpeta):
    """folder_paths.get_filename_list(carpeta), or [] if the folder is unknown in this ComfyUI."""
    try:
        import folder_paths
        return list(folder_paths.get_filename_list(carpeta))
    except Exception:
        return []


def _norm(s):
    return str(s).replace("\\", "/").strip("/")


def nombre_listado(nombre, lista):
    """The string ComfyUI lists for model `nombre`: exact, then same relative path with any separator, then same path ignoring case,
    then same file name in any subfolder. None if it is not there."""
    if not nombre:
        return None
    if nombre in lista:
        return nombre
    n = _norm(nombre)
    por_ruta = {_norm(x): x for x in lista}
    if n in por_ruta:
        return por_ruta[n]
    nl = n.lower()
    hit = next((x for k, x in por_ruta.items() if k.lower() == nl), None)
    if hit:
        return hit
    base = nl.split("/")[-1]
    return next((x for k, x in sorted(por_ruta.items()) if k.lower().split("/")[-1] == base), None)


def hay_modelo(carpetas, nombre, listar=None):
    """True if `nombre` is in any of `carpetas` (a folder name or a tuple of them). listar(carpeta) → list (default: ComfyUI's)."""
    listar = listar or lista_modelos
    for c in ((carpetas,) if isinstance(carpetas, str) else carpetas):
        if nombre_listado(nombre, listar(c)):
            return True
    return False


def modelos_os(G, listar=None):
    """Rewrites every model-name input of an API graph to the exact string ComfyUI lists (OS separator, real subfolder). Names that are
    not found are left as they are, so ComfyUI still reports them. Returns the same graph (modified in place)."""
    listar = listar or lista_modelos
    listas = {}
    for n in (G or {}).values():
        if not isinstance(n, dict) or not isinstance(n.get("inputs"), dict):
            continue
        for k, v in list(n["inputs"].items()):
            if not isinstance(v, str) or not v:
                continue
            for c in carpetas_de(n.get("class_type"), k):
                if c not in listas:
                    listas[c] = listar(c)
                hit = nombre_listado(v, listas[c])
                if hit:
                    n["inputs"][k] = hit
                    break
    return G


def ffmpeg():
    """Path to ffmpeg on any OS: on the PATH (shutil.which), else the binary bundled with imageio-ffmpeg if installed; None if neither."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
