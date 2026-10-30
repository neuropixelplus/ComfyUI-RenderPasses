# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Node: the layered Gaussian splat of a plate (rp_splat) as ComfyUI's core SPLAT type, so it plugs straight into RenderSplat,
# TransformSplat, MergeSplat, SplatToFile3D / Save Splat. It also gives the plate's own camera (camera_info) so RenderSplat starts
# exactly on the original framing, and can write splat.ply into the pass folder.
import os

import folder_paths

from . import rp_splat

CAT = "RenderPasses"


def carpeta_de(texto):
    """Pass folder of a plate from: an absolute path inside output/, a path relative to output/, or "batch/engine/plate"
    (e.g. 20261029_212216/original/futuro_1_00001 → output/renderpasses/20261029_212216/pases_original/futuro_1_00001)."""
    base = os.path.abspath(folder_paths.get_output_directory())
    t = (texto or "").strip().strip('"').replace("\\", "/").rstrip("/")
    if not t:
        raise ValueError("Give the plate's pass folder (it holds plano_N.png, paralaje.json and moge.json)")
    candidatas = [t] if os.path.isabs(t) else [os.path.join(base, t)]
    partes = t.split("/")
    if not os.path.isabs(t) and len(partes) == 3:
        lote, motor, plato = partes
        candidatas.append(os.path.join(base, "renderpasses", lote, motor if motor.startswith("pases_") else "pases_" + motor, plato))
    for c in candidatas:
        c = os.path.abspath(c)
        if os.path.commonpath([c, base]) != base:
            raise ValueError("The pass folder must be inside ComfyUI's output folder")
        if os.path.isfile(os.path.join(c, "paralaje.json")):
            return c
    raise ValueError("No parallax planes (paralaje.json) in %s: run the Parallax and MoGe passes on that plate first" % t)


class RenderPassesLayeredSplat:
    """Parallax planes + MoGe metric depth of one plate → a layered 3D Gaussian splat (one gaussian per pixel of each plane, the
    repainted hidden areas included, stacked in depth)."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pass_folder": ("STRING", {"default": "renderpasses/<batch>/pases_original/<plate>",
                                       "tooltip": "The plate's pass folder (inside output/), or batch/engine/plate. It needs the finished Parallax "
                                                  "planes (plano_N.png, paralaje.json) and MoGe (depth_moge.exr, moge.json)."}),
            "density": ("FLOAT", {"default": 0.625, "min": 0.1, "max": 1.0, "step": 0.025,
                                  "tooltip": "Splats across, as a fraction of the plate width: 0.625 of a 2048-px plate = 1280 across, ~2 M splats."}),
            "scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.05,
                                "tooltip": "Multiplier on each splat's footprint: lower = crisper, higher = softer and more closed."}),
            "include_hidden_fills": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Also place the areas Klein repainted behind the front planes, so a camera move uncovers "
                                                            "them instead of holes."}),
            "orientation": (["surface", "camera"], {"default": "surface",
                                                    "tooltip": "surface: each splat lies on the surface (ground at grazing angles stays closed). "
                                                               "camera: flat discs facing the original camera."}),
            "save_ply": ("BOOLEAN", {"default": True, "tooltip": "Also write splat.ply (standard 3DGS layout) into the pass folder."}),
        }}
    RETURN_TYPES = ("SPLAT", "LOAD3D_CAMERA", "INT", "STRING")
    RETURN_NAMES = ("splat", "camera_info", "count", "ply_path")
    FUNCTION = "ejecuta"
    CATEGORY = CAT
    DESCRIPTION = ("Layered Gaussian splat from RenderPasses' parallax planes and MoGe depth. camera_info is the plate's own camera: feed it to "
                   "Render Splat at the plate's aspect ratio to start on the original framing.")

    def ejecuta(self, pass_folder, density, scale, include_hidden_fills, orientation, save_ply):
        c = carpeta_de(pass_folder)
        G = rp_splat.construir(c, density, scale, include_hidden_fills, superficie=orientation == "surface")
        ruta = ""
        if save_ply:
            ruta, _ = rp_splat.guardar_ply(c, G=G)
        return {"ui": {"text": ["%d splats · %s" % (G["info"]["splats"], ruta or c)]},
                "result": (rp_splat.a_splat(G), rp_splat.camara(G), int(G["info"]["splats"]), ruta)}


NODE_CLASS_MAPPINGS = {"RenderPassesLayeredSplat": RenderPassesLayeredSplat}
NODE_DISPLAY_NAME_MAPPINGS = {"RenderPassesLayeredSplat": "RenderPasses · Layered Gaussian splat"}
