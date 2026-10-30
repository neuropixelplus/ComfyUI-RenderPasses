# RenderPasses for ComfyUI — © 2026 NeuroPixel+ — AGPL-3.0-or-later
# -*- coding: utf-8 -*-
# Spanish ⇄ English prompt translator (served by POST /renderpasses/traducir, see __init__.py).
#   traducir(texto, de="es"|"en"|"auto", a="en"|"es") → {texto, de, a, ms}
# Engine: MarianMT (Helsinki-NLP/opus-mt-es-en and opus-mt-en-es) on CPU ONLY (the GPU is kept for renders); loaded on first
# request and kept in memory; inference runs in a locked thread, sentence by sentence, in batches, with an LRU cache.
# Prompt structure is protected: <Picture 1>/<Subject 2>/<Video 1>, [Shot 1]/[Spanish], section labels (subject_definitions:,
# summary:…), the alignment line "For the target video, at 0.00 seconds…", snake_case tokens, timecodes (00:02.500, 0.00s) and
# dialogue <d>…</d> (spoken lines are NEVER translated). Line breaks are preserved.
# The core is plain functions (traducir(), detectar()) that can be tested without ComfyUI.
import os, re, threading, time
from collections import OrderedDict

MODELOS = {("es", "en"): "Helsinki-NLP/opus-mt-es-en", ("en", "es"): "Helsinki-NLP/opus-mt-en-es"}
_cerrojo = threading.Lock()
_cargados = {}
_cache = OrderedDict()
_cache_cerrojo = threading.Lock()
CACHE_MAX = 3000
LOTE = 12

# ---------------- structure protection ----------------
A, Z = "", ""                       # internal sentinels (private-use characters: never appear in a prompt)
CENT = re.compile(A + r"(\d+)" + Z)
PH = "XQZ%dQZX"                                 # placeholder MarianMT keeps verbatim (⟦0⟧, {0}, @@0@@, Ω0 get lost)
PH_RE = re.compile(r"XQZ(\d+)QZX")
SECCIONES = r"subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music"
PROTEGER = [
    re.compile(r"<d\b[^>]*>.*?</d>", re.S | re.I),                                   # spoken dialogue: never translated
    re.compile(r"^[ \t]*For the target video\b.*$", re.M),                             # alignment line
    re.compile(r"«[^«»\n]{1,120}»|“[^“”\n]{1,120}”|\"[^\"\n]{1,80}\""),             # quoted on-screen text is not translated
    re.compile(r"\((?:aligns with|appears in|from)\b[^()\n]*\)", re.I),                # (aligns with 0.00s) (appears in [Shot 3])
    re.compile(r"\([^()\n]{1,60}:\s*-?\d+(?:\.\d+)?\)"),                               # ComfyUI weights (word:1.2)
    re.compile(r"<[^<>\n]{1,60}>"),                                                    # <Picture 1> <Subject 2> <lora:x:1>
    re.compile(r"\[[^\[\]\n]{1,40}\]"),                                                # [Shot 1] [Spanish]
    re.compile(r"^[ \t]*(?:%s|[a-z]+(?:_[a-z]+)+)[ \t]*:" % SECCIONES, re.M),          # section labels
    re.compile(r"\b[a-zA-Z]+(?:_[a-zA-Z0-9]+)+\b"),                                    # fully_preserved, snake_case_names
    re.compile(r"\b\d{1,2}:\d{2}(?:[.,]\d+)?\b"),                                      # 00:02.500
    re.compile(r"\b\d+(?:\.\d+)?s\b"),                                                 # 0.00s
    re.compile(r"https?://\S+|\bembedding:\S+"),
]


def _proteger(texto):
    guardados = []

    def tapa(m):
        guardados.append(m.group(0))
        return "%s%d%s" % (A, len(guardados) - 1, Z)
    for rx in PROTEGER:
        texto = rx.sub(tapa, texto)
    return texto, guardados


def _destapar(texto, guardados):
    return CENT.sub(lambda m: guardados[int(m.group(1))], texto)


# ---------------- cheap language detection ----------------
_ES = set("el la los las de del que y en un una unos unas por con para es son está están se su sus lo como más pero muy cuando "
          "donde mientras hacia sobre entre sin ya le les al este esta ese esa hay tiene hace".split())
_EN = set("the and of to is are in with on an his her its it for from at this that as by into while toward towards over under "
          "through he she they their was were has have be been very when where".split())


def detectar(texto):
    t, _ = _proteger(texto)
    t = CENT.sub(" ", t).lower()
    pal = re.findall(r"[a-záéíóúüñ]+", t)
    es = sum(1 for p in pal if p in _ES) + 2 * len(re.findall(r"[áéíóúñ¿¡]", t))
    en = sum(1 for p in pal if p in _EN)
    if es == en:
        return "en" if re.search(r"\b(the|and|is)\b", t) else "es"
    return "es" if es > en else "en"


# ---------------- engine ----------------
def _cargar(de, a):
    clave = (de, a)
    if clave in _cargados:
        return _cargados[clave]
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from transformers import MarianMTModel, MarianTokenizer
    nombre = MODELOS[clave]
    try:
        tok = MarianTokenizer.from_pretrained(nombre, local_files_only=True)
        mod = MarianMTModel.from_pretrained(nombre, local_files_only=True)
    except Exception:
        cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")   # in case ComfyUI changed HF_HOME
        try:
            tok = MarianTokenizer.from_pretrained(nombre, cache_dir=cache, local_files_only=True)
            mod = MarianMTModel.from_pretrained(nombre, cache_dir=cache, local_files_only=True)
        except Exception:
            tok = MarianTokenizer.from_pretrained(nombre, cache_dir=cache)
            mod = MarianMTModel.from_pretrained(nombre, cache_dir=cache)
    import torch
    mod = mod.to(device="cpu", dtype=torch.float32).eval()          # never the GPU: it is kept for renders
    _cargados[clave] = (tok, mod)
    return _cargados[clave]


def _generar(de, a, frases):
    """Translate a list of sentences (already with XQZnQZX placeholders). Call with the lock held."""
    import torch
    tok, mod = _cargar(de, a)
    salida = [None] * len(frases)
    orden = sorted(range(len(frases)), key=lambda i: len(frases[i]))         # similar-length batches = less padding
    with torch.inference_mode():
        for k in range(0, len(orden), LOTE):
            idx = orden[k:k + LOTE]
            b = tok([frases[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512)
            o = mod.generate(**b, num_beams=2, max_new_tokens=512)
            for i, t in zip(idx, tok.batch_decode(o, skip_special_tokens=True)):
                salida[i] = t.strip()
    return salida


# ---------------- splitting ----------------
_LETRAS = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}")
_C = "%s\\d+%s" % (A, Z)
# structural prefix at line start: bullet, then "<Picture 1> (aligns with 0.00s): fully_preserved - " or "[Shot 1] "
_PREFIJO = re.compile(r"^(\s*(?:[-*•]\s+|\d+[.)]\s+)?(?:(?:%s\s*)+[:\-–—]\s*(?:%s\s*[:\-–—]?\s*)*)?)" % (_C, _C))
_FRASES = re.compile(r"(?<=[.!?…:;])\s+(?=[\"“¿¡(A-ZÁÉÍÓÚÑ0-9%s])" % A)


def _corta_largas(fr, maximo=700):
    if len(fr) <= maximo:
        return [fr]
    out, cur = [], ""
    for p in re.split(r"(?<=[,;])\s+", fr):
        if cur and len(cur) + len(p) > maximo:
            out.append(cur); cur = p
        else:
            cur = (cur + " " + p) if cur else p
    if cur:
        out.append(cur)
    return out


def _es_traducible(c):
    return bool(_LETRAS.search(CENT.sub("", c)))


def _piezas(linea, guardados):
    """Split a line (with sentinels) into [(text, translatable)] keeping the original whitespace."""
    m = _PREFIJO.match(linea)
    pre = m.group(1) if m else ""
    if not pre.strip(" \t-*•"):
        # [Shot 1] at line start stays outside the sentence (a leading <Subject 1> does NOT: it is the sentence subject)
        m2 = re.match(r"^(\s*(?:%s\s*)+)" % _C, linea)
        if m2 and all(guardados[int(i)].startswith("[") for i in CENT.findall(m2.group(1))):
            pre = m2.group(1)
    resto = linea[len(pre):]
    cola = re.search(r"\s*$", resto).group(0)
    if cola:
        resto = resto[:len(resto) - len(cola)]
    out = [(pre, False)] if pre else []

    def mete(fr):
        for j, c in enumerate(_corta_largas(fr)):
            if j:
                out.append((" ", False))
            out.append((c, _es_traducible(c)))
    pos = 0
    for mm in _FRASES.finditer(resto):
        mete(resto[pos:mm.start()])
        out.append((mm.group(0), False))
        pos = mm.end()
    if resto[pos:]:
        mete(resto[pos:])
    if cola:
        out.append((cola, False))
    return out


def _a_marcadores(frase):
    """Global sentinels → sentence-local XQZ0QZX, XQZ1QZX… (so the cache works across different prompts)."""
    mapa = []

    def f(m):
        mapa.append(m.group(0))
        return PH % (len(mapa) - 1)
    return CENT.sub(f, frase), mapa


def _de_marcadores(trad, mapa):
    vistos = [int(x) for x in PH_RE.findall(trad)]
    if sorted(vistos) != list(range(len(mapa))):
        return None                                   # the model dropped or duplicated a placeholder
    return PH_RE.sub(lambda m: mapa[int(m.group(1))], trad)


def _cache_get(k):
    with _cache_cerrojo:
        v = _cache.get(k)
        if v is not None:
            _cache.move_to_end(k)
        return v


def _cache_put(k, v):
    with _cache_cerrojo:
        _cache[k] = v
        _cache.move_to_end(k)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


def _traducir_frases(de, a, frases):
    """Sentences with placeholders → translations, using the cache and a single model pass for the rest."""
    res, falta = {}, []
    for f in dict.fromkeys(frases):
        v = _cache_get((de, a, f))
        if v is None:
            falta.append(f)
        else:
            res[f] = v
    if falta:
        with _cerrojo:
            trad = _generar(de, a, falta)
        for f, t in zip(falta, trad):
            res[f] = t
            _cache_put((de, a, f), t)
    return [res[f] for f in frases]


def _rescatar(de, a, txt):
    """The model lost a placeholder: translate the chunks between placeholders separately (less fluent but exact)."""
    trozos = re.split(r"(%s)" % _C, txt)
    pedir = [p.strip() for p in trozos if p.strip() and not CENT.fullmatch(p) and _es_traducible(p)]
    hechos = dict(zip(pedir, _traducir_frases(de, a, pedir))) if pedir else {}
    r = ""
    for p in trozos:
        if CENT.fullmatch(p) or p.strip() not in hechos:
            r += p
        else:
            r += p[:len(p) - len(p.lstrip())] + PH_RE.sub("", hechos[p.strip()]) + p[len(p.rstrip()):]
    return r


def traducir(texto, de="auto", a=None):
    """Core: returns {texto, de, a, ms}. Blocking (ComfyUI calls it via asyncio.to_thread)."""
    t0 = time.time()
    texto = str(texto or "").replace("\r\n", "\n")
    if de not in ("es", "en"):
        de = detectar(texto)
    if a not in ("es", "en"):
        a = "en" if de == "es" else "es"
    if de == a or not texto.strip():
        return {"texto": texto, "de": de, "a": a, "ms": 0, "igual": True}
    tapado, guardados = _proteger(texto)
    estructura = [_piezas(l, guardados) for l in tapado.split("\n")]
    mapas = {}
    for piezas in estructura:
        for txt, trad in piezas:
            if trad and txt not in mapas:
                mapas[txt] = _a_marcadores(txt)
    lista = list(mapas)
    hechas = _traducir_frases(de, a, [mapas[t][0] for t in lista]) if lista else []
    final, rescates = {}, 0
    for txt, t in zip(lista, hechas):
        r = _de_marcadores(t, mapas[txt][1])
        if r is None:
            r = _rescatar(de, a, txt); rescates += 1
        final[txt] = r
    salida = "\n".join("".join(final[txt] if trad else txt for txt, trad in piezas) for piezas in estructura)
    return {"texto": _destapar(salida, guardados), "de": de, "a": a, "ms": int((time.time() - t0) * 1000), "rescates": rescates}
