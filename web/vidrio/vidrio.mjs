// © 2026 NeuroPixel+ · Vidrio — what CSS alone cannot do. No dependencies.
//   iniciar(raiz)   adds the lens filter (#vLente, Chromium only) and wires up sliders and segmented controls, present or added later.
//   pista(el)       the glass drop stretches to the button with aria-pressed="true" (and marks it on click).
//   rango(el)       paints the filled part of a .v-rango (--p) and its <output>, if any.
//   avance(el, n, total)  moves the rail of the progress widget.
//   aviso(texto)    the floating toast at the bottom.
//   aplicar(a)      applies appearance settings: theme, accent, glass, blur, corners, edge, scene and lens.
//                   The accent recomputes its derivatives so text stays readable.
// Shipped as .mjs (not .js): ComfyUI loads any .js under web/ as an extension.

function mapaLente(n = 128, b = 0.18) {   // R = x push, G = y push (128 = none): inward along an edge band
  const c = document.createElement("canvas"); c.width = c.height = n; const x = c.getContext("2d"), d = x.createImageData(n, n);
  const f = (t) => (t < b ? (b - t) / b : t > 1 - b ? -(t - (1 - b)) / b : 0);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const dx = f(i / (n - 1)), dy = f(j / (n - 1)), k = (i + j * n) * 4;
    d.data[k] = 128 + Math.round(127 * dx * Math.abs(dx)); d.data[k + 1] = 128 + Math.round(127 * dy * Math.abs(dy)); d.data[k + 2] = 128; d.data[k + 3] = 255; }
  x.putImageData(d, 0, 0); return c.toDataURL("image/png"); }

export function esChromium() { return !!(window.chrome || navigator.userAgentData?.brands?.some((b) => /Chrom/.test(b.brand))); }

export function ponLente() {
  if (document.getElementById("vLenteSvg")) return;
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg"); s.id = "vLenteSvg"; s.setAttribute("aria-hidden", "true");
  s.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;pointer-events:none";
  s.innerHTML = `<filter id="vLente" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">
    <feImage href="${mapaLente()}" x="0" y="0" width="100%" height="100%" preserveAspectRatio="none" result="m"/>
    <feDisplacementMap in="SourceGraphic" in2="m" scale="48" xChannelSelector="R" yChannelSelector="G"/></filter>`;
  document.body.appendChild(s);
  if (esChromium() && !matchMedia("(prefers-reduced-transparency: reduce)").matches) document.documentElement.classList.add("v-con-lente"); }

export function rango(el) {
  const a = +el.min || 0, b = el.max === "" ? 100 : +el.max, v = +el.value;
  el.style.setProperty("--p", (b > a ? ((v - a) / (b - a)) * 100 : 0).toFixed(1) + "%");
  const o = el.parentElement && el.parentElement.querySelector(`output[for="${el.id}"]`); if (o && !o.dataset.propio) o.textContent = el.value; }

export function pista(el) {
  let g = el.querySelector(":scope>.v-gota"); if (!g) { g = document.createElement("i"); g.className = "v-gota"; el.prepend(g); }
  const mueve = () => { const b = el.querySelector(':scope>button[aria-pressed="true"]'); if (!b) { g.style.width = "0"; return; }
    g.style.width = b.offsetWidth + "px"; g.style.transform = `translateX(${b.offsetLeft}px)`; };
  if (!el.dataset.vPista) { el.dataset.vPista = "1";
    el.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b || b.parentElement !== el || b.disabled) return;
      el.querySelectorAll(":scope>button").forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false")); mueve();
      el.dispatchEvent(new CustomEvent("v-elige", { detail: { valor: b.value || b.dataset.valor || b.textContent.trim() }, bubbles: true })); });
    new ResizeObserver(mueve).observe(el); }
  mueve(); return mueve; }

export function avance(el, n, total) {
  const p = total > 0 ? Math.max(0, Math.min(100, (n / total) * 100)) : 0; const r = el.querySelector(".v-riel"); if (r) r.style.setProperty("--p", p.toFixed(1) + "%"); }

let tAviso = 0;
export function aviso(texto, ms = 2600) {
  let a = document.querySelector(".v-aviso"); if (!a) { a = document.createElement("div"); a.className = "v-aviso"; a.setAttribute("role", "status"); document.body.appendChild(a); }
  a.textContent = texto; requestAnimationFrame(() => a.classList.add("on")); clearTimeout(tAviso); tAviso = setTimeout(() => a.classList.remove("on"), ms); }

export function iniciar(raiz = document) {
  ponLente();
  const engancha = (n) => { n.querySelectorAll?.(".v-rango").forEach(rango); n.querySelectorAll?.(".v-pista").forEach(pista); };
  engancha(raiz);
  raiz.addEventListener("input", (e) => { if (e.target.classList?.contains("v-rango")) rango(e.target); }, true);
  new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => n.nodeType === 1 && engancha(n)))).observe(raiz === document ? document.body : raiz, { childList: true, subtree: true }); }

// ---------- appearance ----------
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
export const luz = (h) => { const [r, g, b] = hex(h); return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b); };
export const contraste = (a, b) => { const x = luz(a), y = luz(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
export function derivados(acento, tema) {   // derived from the accent so everything stays readable
  const enA = contraste(acento, "#17190a") >= contraste(acento, "#ffffff") ? "#17190a" : "#ffffff";
  const fondo = tema === "claro" ? "#eceef4" : "#000000", hacia = tema === "claro" ? 0 : 255;
  let txt = acento;
  for (let t = 0.05; contraste(txt, fondo) < 4.5 && t <= 1; t += 0.05) txt = "#" + hex(acento).map((c) => Math.round(c + (hacia - c) * t).toString(16).padStart(2, "0")).join("");
  return { "en-acento": enA, "acento-texto": txt };
}
let escenaUrl = "";   // standalone: no server-provided scene; the page sets --escena itself
export async function aplicar(a, raiz = document.documentElement) {
  if (!a) return;
  if (a.tema) raiz.dataset.theme = a.tema;
  const st = raiz.style, tema = a.tema || raiz.dataset.theme || "negro";
  if (a.acento) { st.setProperty("--acento", a.acento); const d = derivados(a.acento, tema); for (const k in d) st.setProperty("--" + k, d[k]); }
  if (a.vidrio != null) st.setProperty("--vidrio-a", tema === "claro" ? `rgba(255,255,255,${a.vidrio})` : `rgba(6,6,10,${a.vidrio})`);
  if (a.desenfoque != null) { st.setProperty("--desenfoque-a", `blur(${a.desenfoque}px) saturate(150%)`); st.setProperty("--desenfoque-b", `blur(${Math.round(a.desenfoque * 0.8)}px) saturate(170%)`); }
  if (a.esquinas != null) { st.setProperty("--r-placa", a.esquinas + "px"); st.setProperty("--r-hoja", a.esquinas + 4 + "px"); st.setProperty("--r-campo", Math.max(6, a.esquinas - 10) + "px"); }
  if (a.canto != null) st.setProperty("--canto", tema === "claro" ? `rgba(255,255,255,${Math.min(1, a.canto + 0.38)})` : `rgba(255,255,255,${a.canto})`);
  if (a.lente != null) { if (a.lente) ponLente(); else raiz.classList.remove("v-con-lente"); }
  const f = document.querySelector(".v-fondo");
  if (a.escena && f) {
    f.classList.toggle("render", a.escena === "ultimo"); f.classList.toggle("nada", a.escena === "ninguna");
    if (a.escena === "ultimo") {
      if (escenaUrl) f.style.setProperty("--escena", `url("${escenaUrl}")`); else f.classList.remove("render");
    } else f.style.removeProperty("--escena");
  }
}
// Standalone package: there is no settings server, so these are no-ops (kept for API compatibility).
export async function cargarAjustes() { return null; }
export async function guardarAjustes() { return null; }

export default { iniciar, ponLente, rango, pista, avance, aviso, esChromium, aplicar, cargarAjustes, guardarAjustes, derivados, contraste };
