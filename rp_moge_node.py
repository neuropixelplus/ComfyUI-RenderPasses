# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# MoGe-2 (Microsoft, Wang et al., CVPR 2025) predicts, from a single image, a per-pixel X Y Z point map IN METRES plus the camera
# (field of view): a real P pass, without guessing scale or lens as a P derived from relative Depth Anything depth would.
# This node writes what ComfyUI's "Run MoGe Inference" node outputs to float EXR (no 8-bit PNG round trip), using Nuke's camera
# convention (X right, Y up, looking down −Z).
import json
import os

import folder_paths
import numpy as np

CAT = "RenderPasses"


def _exr(ruta, a, canales=None):
    import OpenImageIO as oiio
    a = np.ascontiguousarray(a[..., None] if a.ndim == 2 else a, dtype=np.float32)   # OpenImageIO fails on 2D arrays
    h, w, c = a.shape
    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    spec.channelnames = tuple(canales or (("Y",) if c == 1 else ("R", "G", "B", "A")[:c]))
    spec.attribute("compression", "zip")
    o = oiio.ImageOutput.create(ruta)
    if not o or not o.open(ruta, spec) or not o.write_image(a):
        raise ValueError("could not write %s" % os.path.basename(ruta))
    o.close()


class RenderPassesMoGeEXR:
    """MOGE_GEOMETRY → into the folder (inside output/): position_moge.exr (P in metres, Nuke camera), depth_moge.exr (distance in metres
    along the view axis; sky pushed far away), normales_moge.exr, cielo_moge.png (sky mask), moge.json (lens: field of view, focal and
    centre) and PNG previews (depth_moge.png 16-bit, white = near; position_moge.png)."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "moge_geometry": ("MOGE_GEOMETRY",),
            "carpeta": ("STRING", {"default": "np_moge", "tooltip": "Folder inside output/ where the EXR files are written."}),
        }}
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("fov_x",)
    FUNCTION = "ejecuta"
    CATEGORY = CAT
    OUTPUT_NODE = True

    def ejecuta(self, moge_geometry, carpeta):
        import cv2
        base = os.path.abspath(folder_paths.get_output_directory())
        dst = os.path.abspath(os.path.join(base, carpeta.strip().replace("/", os.sep)))
        if os.path.commonpath([dst, base]) != base:
            raise ValueError("The folder must be inside output/")
        os.makedirs(dst, exist_ok=True)
        pts = moge_geometry["points"][0].float().cpu().numpy()            # OpenCV: x right, y down, z forward
        H, W = pts.shape[:2]
        mask = moge_geometry["mask"][0].cpu().numpy().astype(bool) if "mask" in moge_geometry else np.ones((H, W), bool)
        ok = mask & np.isfinite(pts).all(-1) & (pts[..., 2] > 0)
        z = np.where(ok, pts[..., 2], np.nan)
        lejos = float(np.nanpercentile(z, 99.5) * 3.0) if ok.any() else 100.0   # sky and invalid pixels: far away, behind everything
        K = moge_geometry["intrinsics"][0].float().cpu().numpy() if "intrinsics" in moge_geometry else None
        fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])) if K is not None else (0.5 / np.tan(np.radians(30)), 0.5 / np.tan(np.radians(30)) * W / H, .5, .5)
        if fx > 4:                                                          # in case intrinsics come in pixels instead of normalised
            fx, fy, cx, cy = fx / W, fy / H, cx / W, cy / H
        # where MoGe found no surface (sky), use that pixel's ray at the far distance so P has no holes
        u = (np.arange(W, dtype=np.float32) + 0.5) / W
        v = (np.arange(H, dtype=np.float32) + 0.5) / H
        rx, ry = (u[None, :] - cx) / fx, (v[:, None] - cy) / fy
        rayo = np.stack([np.broadcast_to(rx, (H, W)), np.broadcast_to(ry, (H, W)), np.ones((H, W), np.float32)], -1) * lejos
        P = np.where(ok[..., None], np.nan_to_num(pts), rayo).astype(np.float32)
        Pn = np.stack([P[..., 0], -P[..., 1], -P[..., 2]], -1)            # → Nuke: Y up, looking down −Z
        Z = P[..., 2]
        _exr(os.path.join(dst, "position_moge.exr"), Pn)
        _exr(os.path.join(dst, "depth_moge.exr"), Z)
        if "normal" in moge_geometry:
            N = moge_geometry["normal"][0].float().cpu().numpy()
            _exr(os.path.join(dst, "normales_moge.exr"), np.stack([N[..., 0], -N[..., 1], -N[..., 2]], -1))
        cv2.imwrite(os.path.join(dst, "cielo_moge.png"), ((~ok) * 255).astype(np.uint8))
        disp = 1.0 / np.maximum(Z, 1e-6)                                   # preview: white = near (like the usual depth pass)
        lo, hi = np.percentile(disp, 0.5), np.percentile(disp, 99.5)
        cv2.imwrite(os.path.join(dst, "depth_moge.png"), (np.clip((disp - lo) / max(hi - lo, 1e-9), 0, 1) * 65535).astype(np.uint16))
        ref = Pn[ok] if ok.sum() > 100 else Pn.reshape(-1, 3)
        a, b = np.percentile(ref, 2, 0), np.percentile(ref, 98, 0)
        vis = np.clip((Pn - a) / np.maximum(b - a, 1e-6), 0, 1)
        vis[~ok] *= 0.3
        cv2.imwrite(os.path.join(dst, "position_moge.png"), cv2.cvtColor((vis * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        fov_x, fov_y = float(np.degrees(2 * np.arctan(0.5 / fx))), float(np.degrees(2 * np.arctan(0.5 / fy)))
        info = {"fov_x": round(fov_x, 2), "fov_y": round(fov_y, 2), "fx": fx, "fy": fy, "cx": cx, "cy": cy, "W": W, "H": H,
                "cerca_m": round(float(np.nanpercentile(z, 0.5)), 3) if ok.any() else None, "lejos_m": round(lejos, 2),
                "cielo": round(float((~ok).mean()), 3), "focal_35mm": round(18.0 / np.tan(np.radians(fov_x) / 2), 1)}
        with open(os.path.join(dst, "moge.json"), "w", encoding="utf-8") as f:
            json.dump(info, f)
        return {"ui": {"text": [json.dumps(info)]}, "result": (fov_x,)}


NODE_CLASS_MAPPINGS = {"RenderPassesMoGeEXR": RenderPassesMoGeEXR}
NODE_DISPLAY_NAME_MAPPINGS = {"RenderPassesMoGeEXR": "RenderPasses · MoGe to EXR (P, depth, lens)"}
