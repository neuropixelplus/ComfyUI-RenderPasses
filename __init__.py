# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Turns flat 2D background plates into relit beauties and compositing passes (depth, normals, albedo, layers,
# parallax planes, position, Cryptomatte, AI passes) packed as a multilayer EXR + Nuke script.
# The app lives at /extensions/<this folder>/renderpasses.html and is opened from the "RenderPasses" button in ComfyUI.
# License: AGPL-3.0-or-later (see LICENSE).
import asyncio

from aiohttp import web
from server import PromptServer

from . import rp_server  # noqa: F401  (registers the /renderpasses/* routes)
from .rp_moge_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import rp_splat_node  # layered Gaussian splat → core SPLAT
from . import rp_lote_node  # KSampler with a live preview of the whole batch

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **rp_splat_node.NODE_CLASS_MAPPINGS, **rp_lote_node.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **rp_splat_node.NODE_DISPLAY_NAME_MAPPINGS, **rp_lote_node.NODE_DISPLAY_NAME_MAPPINGS}

WEB_DIRECTORY = "web"

_R = PromptServer.instance.routes


@_R.post("/renderpasses/traducir")
async def _traducir(request):
    """{texto, de?, a?} → {texto, de, a, ms}: Spanish ⇄ English with MarianMT on CPU (used for the master prompt)."""
    from . import rp_translate
    try:
        d = await request.json()
        texto = str(d.get("texto") or "")
        if len(texto) > 60000:
            return web.json_response({"error": "Text too long (60,000 characters max)"}, status=400)
        r = await asyncio.to_thread(rp_translate.traducir, texto, d.get("de") or "auto", d.get("a"))
        return web.json_response(r)
    except Exception as e:
        return web.json_response({"error": "Could not translate: %s" % e}, status=500)


@_R.get("/renderpasses/traducir")
async def _traducir_estado(request):
    return web.json_response({"ok": True})


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
