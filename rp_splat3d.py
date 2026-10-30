# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Live 3D splat (relightable): with the passes of one plate (parallax planes + MoGe depth and lens + albedo + normals + Cryptomatte) builds
# a relightable Gaussian-splatting scene and opens it in the PBR Director (web/splat_director.html: key / fill / rim lights, 3D assets with
# shadows on the floor, free camera, camera director and WebM recording).
# Writes to <plate pass folder>/splat3d/:
#   · splat_pbr.ply: one gaussian per pixel of each plane (rp_splat, the layered splat) plus, per gaussian, albedo (Albedo pass, linear),
#     normal (Normals pass, OpenCV camera), rough / metal / emis from the material of its Cryptomatte layer (with a PSD or colour mask: its
#     layers and materials; otherwise neutral values) and capa (layer index into splat_pbr.json);
#   · splat_pbr.json (layers and materials), moge.json (the lens), ubicaciones.json (the floor: plane fitted to the ground, in metres) and,
#     if present, assets.json ([{k, url "glb/…", name, giro, vis}] with the GLBs in splat3d/glb/; their positions live in ubicaciones.json[k]
#     as {x, y_suelo, z, alto});
#   · estado.json (the status the page polls).
import json
import os
import re
import threading

from . import rp_splat

DIR = "splat3d"
_SUELO = re.compile(r"(calle|street|suelo|ground|piso|floor|road|camino|path|plaza|acera|sidewalk|vereda|pasto|grass|arena|sand)", re.I)
# material by keyword in the layer name or its material: (roughness, metal, emission)
_MATERIALES = ((re.compile(r"(cielo|sky)", re.I), (1.0, 0.0, 1.0)), (re.compile(r"(nube|cloud)", re.I), (1.0, 0.0, 0.6)),
               (re.compile(r"(luz|light|neon|neón|farol|lamp|halo|glow|emis|letrero|sign)", re.I), (0.4, 0.0, 3.0)),
               (re.compile(r"(metal|tren|tram|train|robot|hierro|iron|steel|acero)", re.I), (0.3, 0.6, 0.0)),
               (re.compile(r"(mojad|wet|agua|water|vidrio|glass|charco|puddle)", re.I), (0.22, 0.0, 0.0)),
               (re.compile(r"(plástic|plastic)", re.I), (0.35, 0.0, 0.0)))


def _material(nombre, mat=""):
    for rx, v in _MATERIALES:
        if rx.search(mat or "") or rx.search(nombre or ""):
            return v
    return (0.8, 0.0, 0.0)


def _guarda(carpeta, **kw):
    os.makedirs(carpeta, exist_ok=True)
    f = os.path.join(carpeta, "estado.json")
    tmp = "%s.tmp_%d" % (f, threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(kw, fh, ensure_ascii=False)
    os.replace(tmp, f)


def _ultimo(carpeta, prefijo):
    fs = sorted(f for f in os.listdir(carpeta) if f.startswith(prefijo) and f.endswith(".png"))
    return os.path.join(carpeta, fs[-1]) if fs else None


def _capas(carpeta, W, H):
    """Per-pixel label from the plate's Cryptomatte: (labels H×W, [names], [materials]) or None."""
    import cv2
    import numpy as np
    try:
        with open(os.path.join(carpeta, "cripto.json"), encoding="utf-8") as f:
            C = json.load(f)
    except (OSError, ValueError):
        return None
    capas = C.get("capas") or []
    if not capas:
        return None
    nombres, mats, cob = [], [], []
    mat_de = {}
    try:   # material of each layer: with a colour mask, its legend; with a PSD, the top group (CryptoMaterial)
        with open(os.path.join(carpeta, "mascara_leyenda.json"), encoding="utf-8") as f:
            for c in json.load(f):
                mat_de[c.get("nombre")] = c.get("material") or c.get("grupo") or ""
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    for c in capas:
        a = cv2.imread(os.path.join(carpeta, c["archivo"]), cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        cob.append(cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA))
        nombres.append(c["nombre"])
        mats.append(mat_de.get(c["nombre"], c["nombre"].split("/")[0]))
    if not cob:
        return None
    return np.argmax(np.stack(cob), 0).astype(np.int32), nombres, mats


def _piso(carpeta, M, et, nombres, W, H):
    """Ground plane in metres (OpenCV camera): RANSAC on the MoGe points of the ground layers; without layers, the bottom third."""
    import cv2
    import numpy as np
    Z = rp_splat._leer_exr(os.path.join(carpeta, "depth_moge.exr"))[..., 0]
    Z = cv2.resize(Z, (W, H), interpolation=cv2.INTER_NEAREST)
    u = (np.arange(W) + .5) / W
    v = (np.arange(H) + .5) / H
    RX, RY = ((u - M["cx"]) / M["fx"])[None, :].repeat(H, 0), ((v - M["cy"]) / M["fy"])[:, None].repeat(W, 1)
    zona = None
    if et is not None:
        ks = [k for k, n in enumerate(nombres) if _SUELO.search(n)]
        if ks:
            zona = np.isin(et, ks)
    if zona is None or zona.sum() < 500:
        zona = np.zeros((H, W), bool)
        zona[int(H * .72):] = True
    zona &= np.isfinite(Z) & (Z > 0) & (Z < np.nanpercentile(Z[zona & np.isfinite(Z)], 90) if zona.any() else 40)
    zona = cv2.erode(zona.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    P = np.stack([RX[zona] * Z[zona], RY[zona] * Z[zona], Z[zona]], 1)[:: max(1, int(zona.sum()) // 60000)]
    if len(P) < 50:
        return None
    rng, mejor = np.random.default_rng(0), None
    tol = 0.01 * float(np.median(P[:, 2]))
    for _ in range(300):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        dentro = np.abs(P @ n - n @ s[0]) < tol
        if mejor is None or dentro.sum() > mejor.sum():
            mejor = dentro
    Q = P[mejor]
    c = Q.mean(0)
    n = np.linalg.eigh(np.cov((Q - c).T))[1][:, 0]
    n = n if n[1] < 0 else -n
    return {"normal": [float(x) for x in n], "d": float(-n @ c), "ajuste": round(float(mejor.mean()), 3)}


ANCHO_SPLAT = 900     # gaussians across the frame: ~2.4 M with 4 parallax planes (a 4K plate at 0.625 gave 8.5 M and the browser crawls)


def construir(carpeta, densidad=None):
    """Blocking (run it in a thread). carpeta = the plate's pass folder. → path of splat3d/."""
    import cv2
    import numpy as np
    sal = os.path.join(carpeta, DIR)
    os.makedirs(os.path.join(sal, "glb"), exist_ok=True)
    try:
        _guarda(sal, estado="Checking the passes…")
        try:
            with open(os.path.join(carpeta, "paralaje.json"), encoding="utf-8") as f:
                if json.load(f).get("estado") != "listo":
                    raise ValueError
        except (OSError, ValueError):
            raise ValueError("this plate has no finished parallax planes: extract the Parallax pass first")
        for f in ("moge.json", "depth_moge.exr"):
            if not os.path.isfile(os.path.join(carpeta, f)):
                raise ValueError("this plate is missing the MoGe pass (%s)" % f)
        with open(os.path.join(carpeta, "moge.json"), encoding="utf-8") as f:
            M = json.load(f)
        W, H = int(M.get("W") or 2048), int(M.get("H") or 1152)
        _guarda(sal, estado="Building the gaussians from the parallax planes…")
        G = rp_splat.construir(carpeta, densidad=densidad or min(0.625, ANCHO_SPLAT / float(W)))
        _guarda(sal, estado="Material per gaussian (albedo, normals, Cryptomatte)…")
        P = G["pos"]
        u = np.clip(((P[:, 0] / P[:, 2]) * M["fx"] + M["cx"]) * W, 0, W - 1).astype(int)
        v = np.clip(((P[:, 1] / P[:, 2]) * M["fy"] + M["cy"]) * H, 0, H - 1).astype(int)
        fa = _ultimo(carpeta, "albedo")
        if fa:
            alb = cv2.cvtColor(cv2.resize(cv2.imread(fa), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) / 255.0
            alb = np.where(alb <= 0.04045, alb / 12.92, ((alb + 0.055) / 1.055) ** 2.4)[v, u]
        else:
            alb = np.clip(G["rgb"], 0, 1) ** 2.2
        fn = _ultimo(carpeta, "normals")
        if fn:
            nrm = cv2.cvtColor(cv2.resize(cv2.imread(fn), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) / 127.5 - 1
            nrm = (nrm * np.array([1, -1, -1]))[v, u]          # Y up, Z towards the camera → OpenCV
            nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        else:
            nrm = -P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-6)
        C = _capas(carpeta, W, H)
        if C:
            et, nombres, mats = C
            capa = et[v, u]
            tabla = np.array([_material(n, m) for n, m in zip(nombres, mats)], np.float32)
        else:
            et, nombres, mats = None, ["scene"], [""]
            capa = np.zeros(len(P), np.int32)
            tabla = np.array([(0.8, 0.0, 0.0)], np.float32)
        campos = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3", "albedo_r", "albedo_g", "albedo_b", "rough", "metal", "emis", "capa")
        E = np.empty(len(P), dtype=[(x, "<f4") for x in campos])
        for i, x in enumerate("xyz"):
            E[x] = P[:, i]
        E["nx"], E["ny"], E["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
        dc = (G["rgb"] - 0.5) / rp_splat.C0
        for i in range(3):
            E["f_dc_%d" % i] = dc[:, i]
            E["scale_%d" % i] = np.log(np.maximum(G["escala"][:, i], 1e-8))
        op = np.clip(G["opacidad"], 1e-6, 1 - 1e-6)
        E["opacity"] = np.log(op / (1 - op))
        for i in range(4):
            E["rot_%d" % i] = G["rot"][:, i]
        E["albedo_r"], E["albedo_g"], E["albedo_b"] = alb[:, 0], alb[:, 1], alb[:, 2]
        E["rough"], E["metal"], E["emis"] = tabla[capa, 0], tabla[capa, 1], tabla[capa, 2]
        E["capa"] = capa
        cab = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(P) + "".join("property float %s\n" % x for x in campos) + "end_header\n"
        tmp = os.path.join(sal, "splat_pbr.ply.tmp")
        with open(tmp, "wb") as f:
            f.write(cab.encode())
            f.write(E.tobytes())
        os.replace(tmp, os.path.join(sal, "splat_pbr.ply"))
        with open(os.path.join(sal, "splat_pbr.json"), "w", encoding="utf-8") as f:
            json.dump({"gaussianas": int(len(P)), "capas": nombres, "materiales": {n: list(map(float, t)) for n, t in zip(nombres, tabla)},
                       "camara": "OpenCV (X right, Y down, Z forward) at the origin"}, f, ensure_ascii=False, indent=1)
        with open(os.path.join(sal, "moge.json"), "w", encoding="utf-8") as f:
            json.dump(M, f)
        if not os.path.isfile(os.path.join(sal, "assets.json")):     # no 3D assets yet: an empty list the user can fill
            with open(os.path.join(sal, "assets.json"), "w", encoding="utf-8") as f:
                json.dump([], f)
        _guarda(sal, estado="Measuring the floor…")
        U = {}
        try:
            with open(os.path.join(sal, "ubicaciones.json"), encoding="utf-8") as f:
                U = json.load(f)                                  # positions of assets already placed are kept
        except (OSError, ValueError):
            pass
        piso = _piso(carpeta, M, et, nombres, W, H)
        if piso:
            U["piso"] = piso
        else:
            U.setdefault("piso", {"normal": [0.0, -1.0, 0.0], "d": 1.6})
        with open(os.path.join(sal, "ubicaciones.json"), "w", encoding="utf-8") as f:
            json.dump(U, f, ensure_ascii=False, indent=1)
        _guarda(sal, estado="listo", gaussianas=int(len(P)), capas=len(nombres), piso=bool(piso))
        return sal
    except Exception as e:
        print("[RenderPasses] live 3D splat:", e)
        _guarda(sal, estado="error: %s" % e)
        raise
