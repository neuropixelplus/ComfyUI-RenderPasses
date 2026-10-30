# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Cryptomatte from the artist's layered PSD (the production path).
# In production the background already exists as a layered PSD (each prop, building or plant on its own layer), so the layers ARE the
# object mattes: no VLM, no segmentation, the artist's own antialiased edges.
#   · Every visible leaf pixel layer becomes one CryptoObject, named with its group path ("Buildings/Church dome"). Hidden layers, adjustment
#     layers, clipped layers (they only shade their base) and, by default, text layers are skipped.
#   · Framing: production PSDs usually have overscan (the canvas is larger than the camera frame). The plate is found inside the PSD's
#     merged image (SIFT + a scale-and-shift fit) and the layers are cut to that frame. If it cannot be found, the canvas is scaled to the
#     beauty (with a warning when the aspect ratio differs by more than 1 %).
#   · Atmosphere and effect layers (fog, haze, VFX, cast shadows, non-normal blend modes, opacity < 30 %) are set aside: no ID, no occlusion.
#   · Coverage is composited FRONT TO BACK in the PSD stacking order: layer i gets alpha_i × Π(1 − alpha_j) over the layers above it, so
#     every pixel's coverage is split by visibility and antialiased edges keep their partial values (Cryptomatte ranks 0–3).
#   · A bottom layer that covers the whole canvas (the usual "Background") only gets what nothing else covers; when nothing covers a pixel,
#     it goes to "background". Coverage sums to 1 everywhere.
#   · CryptoAsset = the layer's parent group path, CryptoMaterial = its top-level group.
# Writes into the plate's pass folder: cryptomatte.exr (with the shared writer, rp_server.escribir_cryptomatte), cryptomatte.png /
# cryptomatte_material.png previews, cripto.json (status, with fuente "psd"), psd_capas/NN_<layer>.png (per-layer visible coverage) and
# cripto_cielo.png when a layer is called sky / cielo. psd-tools is imported lazily: without it, the route answers with a clear message.
import json
import os
import re
import threading

EXT_PSD = (".psd", ".psb")
EXT_MASCARA = (".png", ".tif", ".tiff")
MAX_PSD = 1200 * 1024 * 1024        # production PSDs can weigh ~1 GB
ESTADO = "cripto.json"
_CIELO = re.compile(r"(^|[\W_])(sky|skies|cielo)($|[\W_\d])", re.I)   # also "l010_Cielo_NORM" (\b does not split on "_")
_OMITIR = ("adjustment", "group", "psdimage")
# Atmosphere and effect layers (fog, haze, VFX glows, cast shadows) are not objects: they neither own pixels nor hide what is under them.
# Recognised by name, by a blend mode other than normal, or by a very low opacity.
_EFECTO = re.compile(r"(fog|niebla|neblina|bruma|haze|mist|vfx|glow|shadow|sombra|atmos)|(^|[\W_])fx($|[\W_\d])", re.I)
_MEZCLA_OBJETO = ("NORMAL", "PASS_THROUGH", "DISSOLVE")


def falta_psd_tools():
    """None if psd-tools can be imported; otherwise the message the page shows."""
    try:
        import psd_tools  # noqa: F401
        return None
    except Exception:
        return "psd-tools is not installed in ComfyUI's Python: run  python -m pip install psd-tools  and restart ComfyUI"


def es_psd(carpeta):
    """Does this pass folder hold a Cryptomatte made from the artist's layers, a PSD or a mask (finished or in progress)?"""
    try:
        with open(os.path.join(carpeta, ESTADO), encoding="utf-8") as f:
            return json.load(f).get("fuente") in ("psd", "mascara")
    except (OSError, ValueError, AttributeError):
        return False


def guardar_estado(carpeta, **kw):
    """Atomic write of cripto.json (the page polls it)."""
    kw.setdefault("fuente", "psd")
    os.makedirs(carpeta, exist_ok=True)
    f = os.path.join(carpeta, ESTADO)
    tmp = "%s.tmp_%d" % (f, threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(kw, fh, ensure_ascii=False)
    os.replace(tmp, f)


# ───────── reading the PSD ─────────
def hojas(psd, texto=False):
    """Visible leaf layers, bottom → top: [(layer, "Group/Sub/Layer", [group names])]."""
    out = []

    def rec(grupo, ruta):
        for capa in grupo:                                   # psd-tools iterates bottom → top
            if not capa.visible:
                continue
            nombre = (capa.name or "Layer").strip().replace("/", "∕") or "Layer"
            if capa.is_group():
                rec(capa, ruta + [nombre])
                continue
            k = str(getattr(capa, "kind", "pixel"))
            if k in _OMITIR or (k == "type" and not texto):
                continue
            if getattr(capa, "clipping", None) if hasattr(capa, "clipping") else getattr(capa, "clipping_layer", False):
                continue                                     # a clipped layer only shades its base: it belongs to it
            out.append((capa, "/".join(ruta + [nombre]), list(ruta)))
    rec(psd, [])
    return out


def _al_destino(a, left, top, sx, sy, W, H, fondo=0.0):
    """Places a crop (float 0–1) that sits at (left, top) on the PSD canvas onto the W×H plate: pre-reduced with INTER_AREA when shrinking
    (no aliasing), then one sub-pixel affine warp. Outside the crop = fondo."""
    import cv2
    import numpy as np
    h, w = a.shape[:2]
    a = a.astype(np.float32)
    if sx < 1 or sy < 1:
        w2, h2 = max(1, round(w * sx)), max(1, round(h * sy))
        if (w2, h2) != (w, h):
            a = cv2.resize(a, (w2, h2), interpolation=cv2.INTER_AREA)
        fx, fy = w * sx / w2, h * sy / h2
    else:
        fx, fy = sx, sy
    M = np.float32([[fx, 0, left * sx + 0.5 * fx - 0.5], [0, fy, top * sy + 0.5 * fy - 0.5]])
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(fondo))


def es_efecto(capa, nombre):
    """Why this layer is an atmosphere/effect layer (a short reason), or "" if it is an object."""
    hoja = nombre.split("/")[-1]
    if _EFECTO.search(hoja):
        return "named as an effect"
    modo = getattr(getattr(capa, "blend_mode", None), "name", "NORMAL")
    if modo not in _MEZCLA_OBJETO:
        return "blend mode %s" % modo.lower().replace("_", " ")
    if _opacidad(capa) < 0.3:
        return "opacity %d %%" % round(_opacidad(capa) * 100)
    return ""


def _opacidad(capa):
    """Layer opacity × fill × every parent group's opacity (0–1)."""
    relleno = getattr(capa, "fill_opacity", None)
    o = (getattr(capa, "opacity", 255) or 0) / 255.0 * (255 if relleno is None else relleno) / 255.0
    p = getattr(capa, "parent", None)
    while p is not None and getattr(p, "kind", "") != "psdimage":
        o *= (getattr(p, "opacity", 255) or 0) / 255.0
        p = getattr(p, "parent", None)
    return o


def encuadre(psd, beauty):
    """Where the plate sits on the PSD canvas: (x0, y0, s, inliers) with plate = (canvas − (x0, y0)) × s, or None when it cannot be found.
    Matches SIFT features of the beauty (BGR) against the PSD's merged image, both reduced to ≤ 2000 px, and fits scale + shift (RANSAC);
    a rotated or weak fit is rejected."""
    import cv2
    import numpy as np
    ref = None
    try:
        ref = psd.topil()                                    # the merged image stored in the file ("maximize compatibility")
    except Exception:
        ref = None
    if ref is None:
        try:
            ref = psd.composite()
        except Exception:
            return None
    ref = cv2.cvtColor(np.asarray(ref.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if beauty.dtype != np.uint8:
        beauty = (np.clip(beauty / max(float(beauty.max()), 1.0), 0, 1) * 255).astype(np.uint8)
    if beauty.ndim == 3:
        beauty = cv2.cvtColor(beauty[..., :3], cv2.COLOR_BGR2GRAY)
    r = min(1.0, 2000.0 / max(ref.shape))
    q = min(1.0, 2000.0 / max(beauty.shape))
    a = cv2.resize(ref, None, fx=r, fy=r, interpolation=cv2.INTER_AREA) if r < 1 else ref
    b = cv2.resize(beauty, None, fx=q, fy=q, interpolation=cv2.INTER_AREA) if q < 1 else beauty
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None
    pares = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    buenos = [p[0] for p in pares if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(buenos) < 25:
        return None
    pb = np.float32([kb[m.queryIdx].pt for m in buenos])
    pa = np.float32([ka[m.trainIdx].pt for m in buenos])
    M, dentro = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000, confidence=0.999)
    if M is None or int(dentro.sum()) < 25:
        return None
    k, giro = float(np.hypot(M[0, 0], M[1, 0])), float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    if abs(giro) > 0.5 or k <= 0:
        return None
    # the fit maps pixel indices of the reduced images (a = k·b + t); in continuous full-size coordinates: canvas = (k·q/r)·plate + x0
    x0 = (M[0, 2] + 0.5 - 0.5 * k) / r
    y0 = (M[1, 2] + 0.5 - 0.5 * k) / r
    return x0, y0, r / (k * q), int(dentro.sum())


def alfa_capa(capa, sx, sy, W, H, x0=0.0, y0=0.0):
    """The layer's alpha on the plate (W×H float32): transparency × layer mask × opacity. Non-pixel layers (shapes, fills) are rendered.
    (x0, y0) = the plate's top-left corner on the PSD canvas."""
    import numpy as np
    a = None
    if getattr(capa, "has_pixels", lambda: False)() and str(capa.kind) in ("pixel", "smartobject", "type"):
        try:
            a = capa.numpy("shape")
        except Exception:
            a = None
        if a is None:
            try:
                a = capa.numpy("alpha")
            except Exception:
                a = None
        if a is None and capa.width and capa.height:
            a = np.ones((capa.height, capa.width), np.float32)       # no transparency channel: opaque over its box
        l, t = capa.left, capa.top
    if a is None:                                                     # shape, fill, gradient… → let psd-tools render it
        try:
            im = capa.composite()
        except Exception:
            im = None
        if im is None:
            return None
        a = np.asarray(im.getchannel("A") if "A" in im.getbands() else im.convert("L").point(lambda _: 255), np.float32) / 255.0
        l, t = capa.left, capa.top
    a = np.asarray(a, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.max() > 1.0:
        a = a / 255.0
    A = _al_destino(a, l - x0, t - y0, sx, sy, W, H)
    m = getattr(capa, "mask", None) if getattr(capa, "has_mask", lambda: False)() else None
    if m is not None and not getattr(m, "disabled", False):
        try:
            mi = m.topil()
            fondo = (getattr(m, "background_color", 0) or 0) / 255.0
            if mi is not None and m.width and m.height:
                M = _al_destino(np.asarray(mi.convert("L"), np.float32) / 255.0, m.left - x0, m.top - y0, sx, sy, W, H, fondo)
            else:
                M = np.full((H, W), fondo, np.float32)
            A = A * M
        except Exception as e:
            print("[RenderPasses] PSD mask of %s skipped: %s" % (capa.name, e))
    return np.clip(A * _opacidad(capa), 0, 1).astype(np.float32)


def coberturas(ruta_psd, W, H, texto=False, avance=None, beauty=None, efectos=False):
    """→ (names, parent groups, top groups, coverage (n, H, W) summing to 1, warnings, (psd W, psd H), frame on the canvas or None).
    beauty (BGR array of the plate) lets the plate be found inside a PSD with overscan (see encuadre)."""
    import numpy as np
    from psd_tools import PSDImage
    psd = PSDImage.open(ruta_psd)
    Wp, Hp = psd.width, psd.height
    avisos = []
    E = None
    if beauty is not None:
        if avance:
            avance("Finding the plate's frame on the PSD canvas…")
        try:
            E = encuadre(psd, beauty)
        except Exception as e:
            print("[RenderPasses] PSD framing skipped:", e)
    difiere = abs((Wp / Hp) / (W / H) - 1) > 0.01
    if E:
        x0, y0, s, _ = E
        sx = sy = s
        if not difiere and abs(x0) < 2 and abs(y0) < 2 and abs(s * Wp / W - 1) < 0.002:
            E = None                                         # the plate is the whole canvas
    if not E:
        x0 = y0 = 0.0
        sx, sy = W / Wp, H / Hp
        if difiere:
            avisos.append("The PSD is %d×%d and the plate %d×%d: the aspect ratio differs by %.1f %% and the plate was not found in the PSD, "
                          "so the PSD was stretched to fit" % (Wp, Hp, W, H, abs((Wp / Hp) / (W / H) - 1) * 100))
    else:
        avisos.append("The plate is a %d×%d frame of the %d×%d PSD canvas (at %d, %d): only that frame was used"
                      % (round(W / sx), round(H / sy), Wp, Hp, round(x0), round(y0)))
    L = hojas(psd, texto)
    if not L:
        raise ValueError("the PSD has no visible pixel layers")
    alfas, apartadas = [], []
    for i, (capa, nombre, grupos) in enumerate(L):
        if avance:
            avance("Reading layer %d of %d: %s…" % (i + 1, len(L), nombre))
        motivo = "" if efectos else es_efecto(capa, nombre)
        if motivo:
            apartadas.append("%s (%s)" % (nombre.split("/")[-1], motivo))
            continue
        a = alfa_capa(capa, sx, sy, W, H, x0, y0)
        if a is None or a.max() <= 1e-4:
            avisos.append("“%s” is empty or outside the frame and was skipped" % nombre)
            continue
        alfas.append((nombre, grupos, a))
    if apartadas:
        avisos.append("%d atmosphere/effect layers are not objects and do not hide what is under them: %s" % (len(apartadas), ", ".join(apartadas)))
    if not alfas:
        raise ValueError("every visible layer of the PSD is empty")
    if avance:
        avance("Compositing %d layers front to back…" % len(alfas))
    resto = np.ones((H, W), np.float32)
    nombres, padres, sup, C = [], [], [], []
    for k, (nombre, grupos, a) in enumerate(reversed(alfas)):          # top → bottom
        c = a * resto
        resto *= (1.0 - a)
        es_fondo = k == len(alfas) - 1 and len(alfas) > 1 and float((a > 0.996).mean()) > 0.995
        if es_fondo:
            avisos.append("“%s” covers the whole canvas: it only gets what no other layer covers" % nombre)
        if float(c.sum()) < 0.5:
            avisos.append("“%s” is completely hidden behind the layers above it and was skipped" % nombre)
            continue
        base, n = nombre, 2
        while nombre in nombres:
            nombre, n = "%s %d" % (base, n), n + 1
        nombres.append(nombre); padres.append("/".join(grupos) or nombre); sup.append(grupos[0] if grupos else nombre); C.append(c)
    if float(resto.max()) > 0.01:                                       # transparent everywhere: "background", so coverage sums to 1
        nombres.append("background"); padres.append("background"); sup.append("background"); C.append(resto)
    S = np.stack(C).astype(np.float32)
    S /= np.maximum(S.sum(0), 1e-6)[None]
    marco = [round(x0, 2), round(y0, 2), round(W / sx, 2), round(H / sy, 2)] if E else None
    return nombres, padres, sup, S, avisos, (Wp, Hp), marco


def fuente(carpeta):
    """The layer source saved for this plate: ("psd", path) · ("mascara", mask path, legend path) · None."""
    for e in EXT_PSD:
        f = os.path.join(carpeta, "psd_fuente" + e)
        if os.path.isfile(f):
            return ("psd", f)
    m, l = os.path.join(carpeta, "mascara_fuente.png"), os.path.join(carpeta, "mascara_leyenda.json")
    if os.path.isfile(m) and os.path.isfile(l):
        return ("mascara", m, l)
    return None


def _guarda(carpeta, archivo, **kw):
    os.makedirs(carpeta, exist_ok=True)
    f = os.path.join(carpeta, archivo)
    tmp = "%s.tmp_%d" % (f, threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(kw, fh, ensure_ascii=False)
    os.replace(tmp, f)


# ───────── flat-colour mask + legend ─────────
def coberturas_mascara(ruta_mascara, ruta_leyenda, W, H):
    """Flat-colour ID mask + legend [{name, color "#rrggbb", group?, material?}] → (names, groups, materials, coverage (n, H, W) summing
    to 1, warnings). Each mask pixel goes to the nearest legend colour AT THE MASK'S RESOLUTION, then the labels are resized to the plate
    with nearest neighbour (blurring or scaling the colours first mixes two colours at a border and invents a third). Only a compressed mask
    (JPG/webp: many more colours than the legend) gets a median first. Edges are softened by ~1 px so the Cryptomatte keeps antialiasing."""
    import cv2
    import numpy as np
    with open(ruta_leyenda, encoding="utf-8") as f:
        L = json.load(f)
    if isinstance(L, dict):
        L = L.get("capas") or L.get("layers") or [{"nombre": k, "color": v} for k, v in L.items()]
    L = [dict(c, nombre=c.get("nombre") or c.get("name")) for c in L if isinstance(c, dict)]
    L = [c for c in L if c.get("nombre") and re.fullmatch(r"#?[0-9a-fA-F]{6}", str(c.get("color", "")))]
    if not L:
        raise ValueError('the legend has no layers ([{"name": …, "color": "#rrggbb"}])')
    M = cv2.imread(ruta_mascara, cv2.IMREAD_COLOR)
    if M is None:
        raise ValueError("cannot read the mask")
    avisos = []
    if abs((M.shape[1] / M.shape[0]) / (W / H) - 1) > 0.01:
        avisos.append("The mask is %d×%d and the plate %d×%d: it was stretched to the frame" % (M.shape[1], M.shape[0], W, H))
    if len(np.unique(M.reshape(-1, 3)[:: max(1, M.size // 600000)], axis=0)) > 8 * len(L):
        M = cv2.medianBlur(M, 3)
    pal = np.array([[int(str(c["color"]).lstrip("#")[i:i + 2], 16) for i in (4, 2, 0)] for c in L], np.float32)   # BGR
    d = np.stack([((M.astype(np.float32) - q) ** 2).sum(-1) for q in pal])
    lejos = np.sqrt(d.min(0)) > 60
    et = cv2.resize(d.argmin(0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    if lejos.mean() > 0.02:
        avisos.append("%.1f %% of the mask matches no legend colour: it went to the nearest one" % (100 * lejos.mean()))
    S = np.stack([cv2.GaussianBlur((et == k).astype(np.float32), (0, 0), 0.7) for k in range(len(L))])
    S /= np.maximum(S.sum(0), 1e-6)[None]
    nombres, grupos, mats, C = [], [], [], []
    for c, sc in zip(L, S):
        if float(sc.sum()) < 0.5:
            avisos.append("“%s” is not in the mask" % c["nombre"])
            continue
        nombres.append(c["nombre"]); C.append(sc)
        grupos.append(str(c.get("grupo") or c.get("group") or c["nombre"]))
        mats.append(str(c.get("material") or c.get("grupo") or c.get("group") or c["nombre"]))
    return nombres, grupos, mats, np.stack(C).astype(np.float32), avisos


def _limpio(s):
    return re.sub(r"[^\w\- ]+", "_", s).strip()[:60] or "layer"


def capas_desde_fuente(carpeta, beauty, escribir):
    """Whole job from the plate's saved source (blocking: run it in a thread). escribir = rp_server.escribir_cryptomatte. Writes the
    Cryptomatte (cryptomatte.exr + previews + cripto.json), the "Objects" (obj_N.png = the beauty cut by each layer's coverage, RGBA, +
    objetos.json, like the AI objects), one matte per layer in psd_capas/ and cripto_cielo.png when a layer is called sky."""
    import cv2
    import numpy as np
    F = fuente(carpeta)
    if not F:
        raise ValueError("this plate has no PSD or mask")
    nombre_f = os.path.basename(F[1])

    def avance(t):
        _guarda(carpeta, ESTADO, estado=t, fuente=F[0], psd=nombre_f, elementos=[])
        _guarda(carpeta, "objetos.json", estado=t, fuente=F[0], objetos=[])
    try:
        avance("Reading the %s…" % ("PSD" if F[0] == "psd" else "mask"))
        Bh = cv2.imread(beauty, cv2.IMREAD_UNCHANGED)
        if Bh is None:
            raise ValueError("cannot read the plate's beauty")
        H, W = Bh.shape[:2]
        if F[0] == "psd":
            nombres, padres, sup, S, avisos, lienzo, marco = coberturas(F[1], W, H, False, avance, Bh)
        else:
            nombres, padres, sup, S, avisos = coberturas_mascara(F[1], F[2], W, H)
            lienzo, marco = None, None
        avance("Writing the EXR…")
        objetos = [(n, None, p, m) for n, p, m in zip(nombres, padres, sup)]
        sal = escribir(beauty, objetos, os.path.join(carpeta, "cryptomatte.exr"),
                       {"CryptoObject": os.path.join(carpeta, "cryptomatte.png"),
                        "CryptoMaterial": os.path.join(carpeta, "cryptomatte_material.png")}, cobertura=S)
        d = os.path.join(carpeta, "psd_capas")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):                                         # mattes of an earlier source
            if f.lower().endswith(".png"):
                os.remove(os.path.join(d, f))
        for f in os.listdir(carpeta):                                   # earlier objects (the AI ones)
            if re.fullmatch(r"obj_\d+\.png", f):
                os.remove(os.path.join(carpeta, f))
        B8 = cv2.imread(beauty, cv2.IMREAD_COLOR)
        capas, objs = [], []
        for i, (n, c) in enumerate(zip(nombres, S), 1):
            a = (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)
            f = "%02d_%s.png" % (i, _limpio(n.replace("/", " - ")))
            cv2.imwrite(os.path.join(d, f), a)
            capas.append({"nombre": n, "archivo": "psd_capas/" + f, "area": round(float(c.mean()), 4)})
            if n != "background" and not _CIELO.search(n.split("/")[-1]):
                cv2.imwrite(os.path.join(carpeta, "obj_%d.png" % (len(objs) + 1)), np.dstack([B8, a]))
                objs.append({"n": len(objs) + 1, "nombre": n, "archivo": "obj_%d.png" % (len(objs) + 1)})
        cielo = [c for n, c in zip(nombres, S) if _CIELO.search(n.split("/")[-1])]
        if cielo:                                                       # the sky for the shadow and parallax passes
            cv2.imwrite(os.path.join(carpeta, "cripto_cielo.png"), (np.clip(np.sum(cielo, 0), 0, 1) * 255 + 0.5).astype(np.uint8))
        _guarda(carpeta, "objetos.json", estado="listo", fuente=F[0], objetos=objs)
        _guarda(carpeta, ESTADO, estado="listo", fuente=F[0], psd=nombre_f, lienzo=lienzo, marco=marco, elementos=sal.get("CryptoAsset", []),
                objetos=sal.get("CryptoObject", []), materiales=sal.get("CryptoMaterial", []), capas=capas, avisos=avisos)
        return sal
    except Exception as e:
        print("[RenderPasses] layers from PSD / mask:", e)
        _guarda(carpeta, ESTADO, estado="error: %s" % e, fuente=F[0], psd=nombre_f, elementos=[])
        _guarda(carpeta, "objetos.json", estado="error: %s" % e, fuente=F[0], objetos=[])
        raise


def cripto_desde_psd(ruta_psd, carpeta, beauty, escribir, texto=False, nombre_psd=None):
    """Kept for callers of the first version: the PSD must already be saved as the plate's psd_fuente.psd/.psb."""
    return capas_desde_fuente(carpeta, beauty, escribir)


# ───────── route: POST /renderpasses/cripto_psd ─────────
async def ruta(request, ctx):
    """The plate's layer source → Cryptomatte with the layer names, Objects and per-layer mattes (in the background).
    Multipart {psd (.psd/.psb, or a flat-colour .png/.tif mask), leyenda (.json, with a mask), lote, motor, bg}; or JSON {lote, motor, bg,
    ruta, leyenda (path or list)} with local paths (local requests only); or JSON {lote, motor, bg, quitar: true} to drop the source and go
    back to the AI Cryptomatte / Objects. The source is saved in the plate's pass folder (psd_fuente.psd/.psb, or mascara_fuente.png +
    mascara_leyenda.json): "Extract passes" with Cryptomatte or Objects then rebuilds them from it. → {ok, fuente, estado, carpeta}.
    ctx: leer_lote, salida, dir_pases, escribir, es_local, en_fondo, motores, raiz (folder for temporary uploads)."""
    import asyncio
    import shutil
    import uuid
    from aiohttp import web

    def mal(t, s=400):
        return web.json_response({"error": t}, status=s)
    campos, tmp, tmp_ley, origen, leyenda = {}, None, None, None, None
    try:
        if (request.content_type or "").startswith("multipart/"):
            lector = await request.multipart()
            while True:
                parte = await lector.next()
                if parte is None:
                    break
                if parte.name in ("psd", "archivo", "leyenda") and parte.filename:
                    os.makedirs(ctx["raiz"], exist_ok=True)
                    destino = os.path.join(ctx["raiz"], "_capas_%s.tmp" % uuid.uuid4().hex)
                    n = 0
                    with open(destino, "wb") as f:
                        while True:
                            trozo = await parte.read_chunk(1 << 20)
                            if not trozo:
                                break
                            n += len(trozo)
                            if n > MAX_PSD:
                                raise OverflowError
                            f.write(trozo)
                    if parte.name == "leyenda":
                        tmp_ley = destino
                    else:
                        tmp, origen = destino, os.path.basename(parte.filename)
                elif parte.name in ("lote", "motor", "bg"):
                    campos[parte.name] = (await parte.text()).strip()[:200]
        else:
            if not ctx["es_local"](request):
                return mal("local paths are only accepted from this machine; upload the file instead", 403)
            d = await request.json()
            campos = {k: str(d.get(k) or "").strip() for k in ("lote", "motor", "bg")}
            campos["quitar"] = bool(d.get("quitar"))
            origen = str(d.get("ruta") or "").strip().strip('"') or None
            leyenda = d.get("leyenda")
        lote, motor, stem = campos.get("lote", ""), campos.get("motor", ""), campos.get("bg", "")
        L = ctx["leer_lote"](lote)
        if not L:
            return mal("batch not found", 404)
        if not re.fullmatch(r"[a-z0-9_]+", motor) or (motor not in ctx["motores"] and motor != "original" and not motor.endswith("_4k")):
            return mal("unknown engine")
        if stem not in [b["stem"] for b in L.get("bgs", [])]:
            return mal("that plate is not in the batch", 404)
        beauty = ctx["salida"](lote, motor, stem)
        if not beauty or not os.path.isfile(beauty):
            return mal("that plate has no beauty with this engine yet", 404)
        carpeta = ctx["dir_pases"](lote, motor, stem)
        os.makedirs(carpeta, exist_ok=True)
        viejas = [os.path.join(carpeta, "psd_fuente" + e) for e in EXT_PSD] + [os.path.join(carpeta, x) for x in ("mascara_fuente.png", "mascara_leyenda.json")]
        if campos.get("quitar"):
            for f in viejas:
                if os.path.isfile(f):
                    os.remove(f)
            return web.json_response({"ok": True, "fuente": None})
        if not origen:
            return mal("no PSD or mask received (field “psd”)")
        src = tmp or origen
        if not os.path.isfile(src):
            return mal("file not found: %s" % origen[:200])
        if not tmp and os.path.getsize(src) > MAX_PSD:
            return mal("the file is larger than 1.2 GB", 413)
        ext = os.path.splitext(origen)[1].lower()
        if ext in EXT_PSD:
            falta = falta_psd_tools()
            if falta:
                return mal(falta, 501)
            with open(src, "rb") as f:
                if f.read(4) != b"8BPS":
                    return mal("this is not a Photoshop file (bad signature)")
            for f in viejas:
                if os.path.isfile(f):
                    os.remove(f)
            destino = os.path.join(carpeta, "psd_fuente" + ext)
            if tmp:
                shutil.move(tmp, destino)
                tmp = None
            else:
                await asyncio.to_thread(shutil.copyfile, origen, destino)
            tipo = "psd"
        elif ext in EXT_MASCARA:
            if tmp_ley:
                with open(tmp_ley, encoding="utf-8") as f:
                    ley = json.load(f)
            elif isinstance(leyenda, (list, dict)):
                ley = leyenda
            elif isinstance(leyenda, str) and os.path.isfile(leyenda.strip('"')):
                with open(leyenda.strip('"'), encoding="utf-8") as f:
                    ley = json.load(f)
            else:
                return mal('a mask needs its legend (.json: [{"name": …, "color": "#rrggbb"}])')
            import cv2
            M = cv2.imread(src, cv2.IMREAD_COLOR)
            if M is None:
                return mal("cannot read the mask")
            for f in viejas:
                if os.path.isfile(f):
                    os.remove(f)
            cv2.imwrite(os.path.join(carpeta, "mascara_fuente.png"), M)
            with open(os.path.join(carpeta, "mascara_leyenda.json"), "w", encoding="utf-8") as f:
                json.dump(ley, f, ensure_ascii=False, indent=1)
            tipo = "mascara"
        else:
            return mal("upload a .psd / .psb, or a .png mask together with its .json legend")
        nombre = os.path.basename(origen)
        guardar_estado(carpeta, estado="queued…", fuente=tipo, psd=nombre, elementos=[])

        async def trabajo():
            try:
                await asyncio.to_thread(capas_desde_fuente, carpeta, beauty, ctx["escribir"])
            except Exception:
                pass                                                    # already in cripto.json
        ctx["en_fondo"](lote, trabajo())
        return web.json_response({"ok": True, "fuente": tipo, "estado": "queued…", "carpeta": carpeta})
    except OverflowError:
        return mal("the file is larger than 1.2 GB", 413)
    except (ValueError, OSError) as e:
        return mal(str(e))
    finally:
        for f in (tmp, tmp_ley):
            if f and os.path.isfile(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
