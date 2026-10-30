# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Layered Gaussian splat: the parallax planes of ONE flat plate turned into a real 3D Gaussian splat.
#
# What goes in (all already in the plate's pass folder):
#   · plano_N.png (RGBA): the depth planes, front to back; the last one is the full background plate with everything in front of it
#     removed and repainted by Klein;
#   · depth_moge.exr + moge.json: MoGe metric depth (metres along the view axis) and the lens (normalised fx, fy, cx, cy).
# How it is built:
#   · every pixel of every plane becomes one gaussian at its 3D position (pinhole camera at the origin, OpenCV / 3DGS frame:
#     X right, Y down, Z forward), with the plane's colour and its alpha as opacity;
#   · depth: where the plane is the visible one, MoGe's own distance (a 2 px band along its outline takes the depth of the nearest
#     interior pixel, so soft edges do not float halfway between foreground and background); where the plane is HIDDEN by planes in
#     front (the repainted areas), the depth is inpainted from the plane itself and everything behind it, smoothed, and never put in
#     front of what hides it. So the planes stack in depth and a camera move uncovers repainted content instead of holes;
#   · size: the pixel footprint at that distance (z · pixel angle) times an overlap factor, as a flat disc facing the camera, so the
#     surface stays closed without the rubber-sheet stretching a projected mesh shows at depth jumps;
#   · count: the planes are resampled (area average, premultiplied) to a working size: density 0.625 of a 2048-px plate (1280 splats across) is ~2.1 M splats.
# Output: the core SPLAT type (comfy_api Types.SPLAT, render-ready: linear scales, wxyz quaternions, opacity 0–1, SH DC colour), so it
# plugs into RenderSplat, TransformSplat, MergeSplat, SplatToFile3D…; and splat.ply (standard 3DGS layout) in the pass folder.
import json
import os

import numpy as np

C0 = 0.28209479177387814          # SH band-0 constant (DC coefficient ↔ base RGB)
ARCHIVO = "splat.ply"
SOLAPE = 0.62                      # gaussian sigma in pixel footprints: 0.62 closes the surface without blurring it
DELGADO = 0.25                     # thickness of each surfel (fraction of its smaller footprint)
SUELTO = 2.5                       # ...and longer than 2.5 when the two sides disagree (a loose pixel between two depths)
LIMITE = 40.0                      # a step longer than 40 pixel sizes is a depth jump, not a slanted surface
FONDO = 1.35                       # the background plate has nothing behind it: wider overlap so no dark pinholes between splats
MARGEN = 0.05                      # overscan on each side (fraction of the width), mirrored from the plate edge


def _leer_exr(ruta):
    import OpenImageIO as oiio
    i = oiio.ImageInput.open(ruta)
    if not i:
        raise ValueError("could not read %s" % os.path.basename(ruta))
    a = i.read_image(oiio.FLOAT)
    i.close()
    return a


def _plano(ruta, w, h):
    """Plane PNG → (rgb 0–1, alpha 0–1) at w×h, resampled premultiplied so edges do not pick up dark fringes."""
    import cv2
    A = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if A is None:
        raise ValueError("could not read %s" % os.path.basename(ruta))
    A = A.astype(np.float32) / (65535.0 if A.dtype == np.uint16 else 255.0)
    if A.ndim == 2:
        A = np.dstack([A, A, A])
    rgb = A[..., 2::-1] if A.shape[2] >= 3 else np.repeat(A[..., :1], 3, -1)
    a = A[..., 3] if A.shape[2] >= 4 else np.ones(A.shape[:2], np.float32)
    pm = np.dstack([rgb * a[..., None], a])
    if pm.shape[1] != w or pm.shape[0] != h:
        pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_AREA)
    a = np.clip(pm[..., 3], 0, 1)
    rgb = np.clip(pm[..., :3] / np.maximum(a, 1e-6)[..., None], 0, 1)
    return rgb, a


def _del_mas_cercano(valor, fuente):
    """For every pixel, `valor` at the nearest pixel where `fuente` is True."""
    import cv2
    _, idx = cv2.distanceTransformWithLabels((~fuente).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    return valor[fuente][np.clip(idx - 1, 0, None)]


def _relleno_z(logz, fuente, w, h):
    """Depth (metres) for areas outside `fuente`: log-depth inpainted from `fuente` at low resolution and smoothed (a continuous
    surface, like the ground continuing behind an object)."""
    import cv2
    wl = min(w, 480)
    hl = max(2, round(h * wl / w))
    lz = cv2.resize(logz, (wl, hl), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize((~fuente).astype(np.uint8), (wl, hl), interpolation=cv2.INTER_NEAREST)
    # Navier-Stokes, not Telea: OpenCV 5.0's Telea returns garbage on float32 (values far outside the input, exp() of them overflows to
    # ~1e37 m and eigh() then fails). Clipped to the known range so a bad inpaint can never put a splat at infinity.
    known = lz[m == 0]
    f = cv2.inpaint(lz, m, 5, cv2.INPAINT_NS)
    if known.size:
        f = np.clip(f, float(known.min()), float(known.max()))
    f = cv2.GaussianBlur(f, (0, 0), max(2.0, wl / 90))
    return np.exp(cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR))



def _mat_a_quat(R):
    """Rotation matrices (N,3,3) -> quaternions (N,4) wxyz (largest-component branch for stability)."""
    n = R.shape[0]
    t = np.stack([1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2],
                  1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2], 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]], 1)
    i = t.argmax(1)
    s = 2.0 * np.sqrt(np.maximum(t[np.arange(n), i], 1e-12))
    d = lambda a, b: R[:, a[0], a[1]] - R[:, b[0], b[1]]
    u = lambda a, b: R[:, a[0], a[1]] + R[:, b[0], b[1]]
    q = np.empty((n, 4), np.float32)
    for c, (w_, x_, y_, z_) in enumerate((
            (s / 4, d((2, 1), (1, 2)) / s, d((0, 2), (2, 0)) / s, d((1, 0), (0, 1)) / s),
            (d((2, 1), (1, 2)) / s, s / 4, u((0, 1), (1, 0)) / s, u((0, 2), (2, 0)) / s),
            (d((0, 2), (2, 0)) / s, u((0, 1), (1, 0)) / s, s / 4, u((1, 2), (2, 1)) / s),
            (d((1, 0), (0, 1)) / s, u((0, 2), (2, 0)) / s, u((1, 2), (2, 1)) / s, s / 4))):
        k = i == c
        q[k] = np.stack([w_[k], x_[k], y_[k], z_[k]], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _gaussianas(Zp, RGB, A, sel, RX, RY, pie, ks, superficie=True):
    """One gaussian per selected pixel of a depth map. Each one covers its pixel's footprint ON THE SURFACE (a surfel spanned by the
    3D steps to the neighbouring pixels, so ground seen at a grazing angle stays closed when the camera moves), taking the step on the
    side with the smaller depth change (at a depth jump the pixel belongs to its own side: nothing is stretched across the gap). Where
    even that step is a jump (a one-pixel-wide sliver) it falls back to a disc facing the camera."""
    z = Zp[sel].astype(np.float32)
    rx, ry = RX[sel], RY[sel]
    P = np.stack([RX * Zp, RY * Zp, Zp], -1).astype(np.float32)
    d = np.stack([rx, ry, np.ones_like(z)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    b = z * pie                                                          # pixel size on the plane z = const
    ejes = []
    for eje, c in ((1, 0), (0, 1)):                                      # image x (columns), image y (rows)
        e = np.zeros(3, np.float32)
        e[c] = 1
        cara = e[None, :] - d * d[:, c:c + 1]
        cara = cara / np.linalg.norm(cara, axis=-1, keepdims=True) * (b * d[:, 2])[:, None]   # camera-facing footprint
        if not superficie:
            ejes.append(cara)
            continue
        ad = np.diff(P, axis=eje, append=np.take(P, [-1], axis=eje))[sel]     # forward step (0 at the last row/column)
        at = np.diff(P, axis=eje, prepend=np.take(P, [0], axis=eje))[sel]     # backward step (0 at the first)
        cero_ad, cero_at = np.abs(ad).sum(-1) == 0, np.abs(at).sum(-1) == 0
        usa_ad = ((np.abs(ad[:, 2]) <= np.abs(at[:, 2])) & ~cero_ad) | cero_at
        T = np.where(usa_ad[:, None], ad, at)
        # a slanted surface (ground at a grazing angle) steps the same way on both sides; a sliver caught between two depths does not
        largo = np.linalg.norm(T, axis=-1)
        parejo = np.abs(ad[:, 2] - at[:, 2]) <= 0.35 * np.maximum(np.abs(ad[:, 2]), np.abs(at[:, 2]))
        malo = (largo > LIMITE * b) | ((largo > SUELTO * b) & ~parejo)  # still a jump: sliver -> camera-facing disc
        ejes.append(np.where(malo[:, None], cara, T))
    Tu, Tv = ejes
    n = np.cross(Tu, Tv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(nn > 1e-12, n / np.maximum(nn, 1e-12), -d)
    grosor = DELGADO * np.minimum(np.linalg.norm(Tu, axis=-1), np.linalg.norm(Tv, axis=-1))
    S = ks * ks * (Tu[:, :, None] * Tu[:, None, :] + Tv[:, :, None] * Tv[:, None, :]) + \
        (ks * grosor)[:, None, None] ** 2 * (n[:, :, None] * n[:, None, :])
    val, vec = np.linalg.eigh(S.astype(np.float64))
    vec[np.linalg.det(vec) < 0, :, 0] *= -1                              # proper rotation (det +1)
    return {"pos": np.stack([rx * z, ry * z, z], -1), "escala": np.sqrt(np.maximum(val, 1e-14)).astype(np.float32),
            "rot": _mat_a_quat(vec.astype(np.float32)), "opacidad": A[sel], "rgb": RGB[sel]}


def _respaldo(zp, rgb, w):
    """Back plate for the background plane's own depth jumps (a dome against the sky): nothing was repainted behind them, so the near
    side of each jump gets a second layer at the far side's depth with the far side's colour inpainted. -> (where, depth, rgb) or None."""
    import cv2
    R = max(4, w // 70)
    zmax = cv2.dilate(zp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1,) * 2))
    cerca = zp < 0.8 * zmax
    if cerca.sum() < 20:
        return None
    zb = np.maximum(cv2.GaussianBlur(zmax, (0, 0), R / 2), zp * 1.05)
    # colour of the FAR side only (the sky behind the dome, not the dome): nearest pixel that is clearly farther than its surroundings
    lejos = zp > 1.25 * cv2.erode(zp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1,) * 2))
    lejos &= ~cv2.dilate(cerca.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # skip the anti-aliased halo of the edge
    if lejos.sum() < 20:
        return None
    col = np.dstack([_del_mas_cercano(rgb[..., c], lejos) for c in range(3)])
    col = cv2.GaussianBlur(col, (0, 0), max(1.5, R / 3)).astype(np.float32)
    return cerca, zb.astype(np.float32), col


def construir(carpeta, densidad=0.625, escala=1.0, rellenos=True, superficie=True, progreso=None):
    """Plate pass folder → dict of numpy arrays: pos (N,3) m, escala (N,3) linear sigma, rot (N,4) wxyz, opacidad (N,), rgb (N,3) sRGB
    0–1, plano (N,) plane index; plus "info" (counts per plane, lens, working size)."""
    import cv2
    with open(os.path.join(carpeta, "moge.json"), encoding="utf-8") as f:
        M = json.load(f)
    with open(os.path.join(carpeta, "paralaje.json"), encoding="utf-8") as f:
        PJ = json.load(f)
    if PJ.get("estado") != "listo" or not PJ.get("planos"):
        raise ValueError("the parallax planes of this plate are not finished")
    planos = sorted(PJ["planos"], key=lambda p: p["n"])
    ref = cv2.imread(os.path.join(carpeta, planos[-1]["archivo"]), cv2.IMREAD_UNCHANGED)
    if ref is None:
        raise ValueError("missing %s" % planos[-1]["archivo"])
    H0, W0 = ref.shape[:2]
    densidad = float(min(max(densidad, 0.05), 1.0))
    w = max(64, round(W0 * densidad))
    h = max(36, round(H0 * w / W0))
    z = _leer_exr(os.path.join(carpeta, "depth_moge.exr"))[..., 0]
    Z = cv2.resize(z, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)   # nearest: no depth halfway across an edge
    Z = np.where(np.isfinite(Z) & (Z > 1e-3), Z, np.nanmax(np.where(np.isfinite(Z), Z, 0)) or 100.0).astype(np.float32)
    logz = np.log(Z)
    fx, fy, cx, cy = float(M["fx"]), float(M["fy"]), float(M["cx"]), float(M["cy"])
    rx = ((np.arange(w, dtype=np.float32) + 0.5) / w - cx) / fx          # ray at z = 1
    ry = ((np.arange(h, dtype=np.float32) + 0.5) / h - cy) / fy
    RX, RY = np.broadcast_to(rx[None, :], (h, w)), np.broadcast_to(ry[:, None], (h, w))
    pie = 1.0 / (0.5 * (fx * w + fy * h))                                # pixel angle (radians at the centre)

    k = len(planos)
    P = []
    for i, p in enumerate(planos):
        P.append(_plano(os.path.join(carpeta, p["archivo"]), w, h))
        if progreso:
            progreso(i + 1, 2 * k)
    lab = np.full((h, w), k - 1, np.int32)                               # which plane is the visible one at each pixel
    for j in reversed(range(k - 1)):                                     # back to front: the front one wins
        lab[P[j][1] > 0.5] = j
    borde = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    r_occ = max(2, w // 200)
    disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_occ + 1,) * 2)
    m = round(MARGEN * w)                                                # overscan: the plate continues a little past its frame
    pad = lambda x: cv2.copyMakeBorder(x, m, m, m, m, cv2.BORDER_REFLECT_101) if m else x
    rxp = ((np.arange(-m, w + m, dtype=np.float32) + 0.5) / w - cx) / fx
    ryp = ((np.arange(-m, h + m, dtype=np.float32) + 0.5) / h - cy) / fy
    RXp, RYp = np.broadcast_to(rxp[None, :], (h + 2 * m, w + 2 * m)), np.broadcast_to(ryp[:, None], (h + 2 * m, w + 2 * m))
    ks = SOLAPE * float(escala)

    out = {x: [] for x in ("pos", "escala", "rot", "opacidad", "rgb", "plano")}

    def suma(g, j):
        for x in ("pos", "escala", "rot", "opacidad", "rgb"):
            out[x].append(g[x])
        out["plano"].append(np.full(g["opacidad"].shape, j, np.int16))

    info = []
    for j, p in enumerate(planos):
        rgb, a = P[j]
        propio = lab == j
        if propio.sum() < 20:
            propio = a > 0.5
        interior = cv2.erode(propio.astype(np.uint8), borde).astype(bool)
        if interior.sum() < 20:
            interior = propio
        zj = _del_mas_cercano(Z, interior) if interior.any() else Z.copy()   # own depth inside; soft rim takes the nearest interior depth
        zj = np.where(interior, Z, zj)
        tapado = lab < j                                                 # hidden behind front planes (repainted)
        if tapado.any():
            zf = _relleno_z(logz, lab >= j, w, h)
            # never in front of what hides it: behind the farthest nearby occluder depth, smoothed (a jagged occluder such as a cart's
            # rails would otherwise make a jagged fill that tears open when the camera moves)
            occ = cv2.GaussianBlur(cv2.dilate(Z, disco), (0, 0), r_occ)
            zj = np.where(tapado, np.maximum(zf, occ * 1.04), zj)
        sel = a > 0.02
        if not rellenos:
            sel &= ~tapado
        zp, rgbp, ap = pad(zj), pad(rgb), pad(np.where(sel, a, 0).astype(np.float32))
        selp = ap > 0.02
        kj = ks * (FONDO if j == k - 1 else 1.0)
        suma(_gaussianas(zp, rgbp, ap, selp, RXp, RYp, pie, kj, superficie), j)
        n_atras = 0
        if j == k - 1 and rellenos:                                      # nothing lies behind the background plate: back its depth jumps
            atras = _respaldo(zp, rgbp, w)
            if atras is not None:
                cerca, zb, rgbb = atras
                c2 = (slice(None, None, 2),) * 2                     # half resolution: it is only ever seen in slivers
                suma(_gaussianas(zb[c2], rgbb[c2], np.ones_like(zb[c2]), cerca[c2], RXp[c2], RYp[c2], 2 * pie, kj, superficie), j)
                n_atras = int(cerca[c2].sum())
        info.append({"n": p["n"], "nombre": p.get("nombre"), "splats": int(selp.sum()) + n_atras, "ocultos": int((sel & tapado).sum()),
                     "respaldo": n_atras})
        if progreso:
            progreso(k + j + 1, 2 * k)
    G = {x: np.ascontiguousarray(np.concatenate(v).astype(np.int16 if x == "plano" else np.float32)) for x, v in out.items()}
    G["info"] = {"splats": int(G["pos"].shape[0]), "planos": info, "w": w, "h": h, "fov_x": M.get("fov_x"), "fov_y": M.get("fov_y"),
                 "densidad": densidad, "escala": float(escala), "rellenos": bool(rellenos), "superficie": bool(superficie), "margen": m,
                 "z_ref": float(planos[min(1, k - 1)].get("z") or np.median(Z))}
    return G


def a_splat(G):
    """dict from construir → core SPLAT (batch of 1)."""
    import torch
    from comfy_api.latest import Types
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))
    sh = (G["rgb"] - 0.5) / C0
    return Types.SPLAT(t(G["pos"])[None], t(G["escala"])[None], t(G["rot"])[None], t(G["opacidad"]).reshape(1, -1, 1), t(sh)[None, :, None, :])


def camara(G):
    """camera_info (Load3D convention: right-handed, Y up, looking down −Z) of the plate's own camera, aimed at the mid plane so an
    orbit pivots inside the scene. RenderSplat measures fov over the shorter side: render at the plate's aspect ratio."""
    i = G["info"]
    zr = float(i.get("z_ref") or 10.0)
    return {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "target": {"x": 0.0, "y": 0.0, "z": -zr},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}, "fov": float(i.get("fov_y") or 45.0), "cameraType": "perspective", "zoom": 1.0}


def _ply_propio(G):
    """Standard 3DGS binary PLY (SH degree 0): x y z nx ny nz f_dc_0..2 opacity (logit) scale_0..2 (log) rot_0..3 (wxyz)."""
    n = G["pos"].shape[0]
    nombres = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
               "rot_0", "rot_1", "rot_2", "rot_3"]
    E = np.empty(n, dtype=[(x, "<f4") for x in nombres])
    for i, x in enumerate("xyz"):
        E[x] = G["pos"][:, i]
    E["nx"] = E["ny"] = E["nz"] = 0
    dc = (G["rgb"] - 0.5) / C0
    for i in range(3):
        E["f_dc_%d" % i] = dc[:, i]
        E["scale_%d" % i] = np.log(np.maximum(G["escala"][:, i], 1e-8))
    op = np.clip(G["opacidad"], 1e-6, 1 - 1e-6)
    E["opacity"] = np.log(op / (1 - op))
    for i in range(4):
        E["rot_%d" % i] = G["rot"][:, i]
    cab = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join("property float %s\n" % x for x in nombres) + "end_header\n"
    return cab.encode("ascii") + E.tobytes()


def ply_bytes(G):
    """Core's own 3DGS PLY writer when running inside ComfyUI; the same standard layout written here otherwise."""
    try:
        from comfy_extras.nodes_gaussian_splat import _gaussian_ply_bytes
        import torch
    except Exception:
        return _ply_propio(G)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))
    return _gaussian_ply_bytes(t(G["pos"]), t(G["escala"]), t(G["rot"]), t(G["opacidad"]).reshape(-1, 1), t((G["rgb"] - 0.5) / C0)[:, None, :])


def guardar_ply(carpeta, densidad=0.625, escala=1.0, rellenos=True, G=None):
    """Builds the splat of a plate and writes splat.ply (+ splat.json with the counts) into its pass folder. → (path, info)."""
    G = G or construir(carpeta, densidad, escala, rellenos)
    ruta = os.path.join(carpeta, ARCHIVO)
    tmp = ruta + ".tmp"
    with open(tmp, "wb") as f:
        f.write(ply_bytes(G))
    os.replace(tmp, ruta)
    info = dict(G["info"], camara=camara(G))
    with open(os.path.join(carpeta, "splat.json"), "w", encoding="utf-8") as f:
        json.dump(info, f)
    return ruta, info
