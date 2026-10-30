# RenderPasses for ComfyUI

**Turn flat 2D background plates into compositing-ready render passes, using only open-weights models running locally in ComfyUI.**

Animation and VFX teams get background plates as flat images: a painting, a generated frame or an approved BG. A flat image has no depth, normals, albedo, mattes or layers, so the compositor can't relight it, add fog, defocus it or move a camera through it. RenderPasses rebuilds those passes from the single image and packs them into one multilayer EXR with a ready-to-open Nuke script.

It works on a **whole folder of BGs at once**, in two stages. You can bring your own plates or **create** them first: the **Create** button writes new plates from a prompt with Krea 2 (ComfyUI's official Krea 2 Turbo recipe), and they load like uploaded images.

1. **Look (relight / finish).** One *master prompt* is applied to every BG with the engine you choose: Krea 2, Klein 3, Fibo 2, Krea 2 V2, MOSS, or MOSS · depth + color. You can set the output resolution (native, 2K base or original), add an optional 4K upscale with SeedVR2, and turn on **Keep original color**, which takes the low frequencies (color, light) from the original and the high-frequency detail from the AI. Contact sheets compare the engines side by side.
2. **Extract passes.** From the chosen beauty, RenderPasses generates the passes below and writes them per BG, plus `pases_nuke.exr` (multilayer) and a `pases.nk` script that rebuilds the beauty (`albedo × raw lighting`) with Cryptomatte and ZDefocus wired in.

| Pass | How it's made |
|---|---|
| Depth (16-bit), refined depth | Depth Anything V2, then edge-aware refinement |
| Normals, albedo | Marigold v2 (LoRAs on Qwen-Image-Edit) |
| Raw lighting | beauty ÷ albedo |
| Position (P) + camera lens | MoGe-2 (metric point map, FOV, focal length) → EXR via the included `RenderPassesMoGeEXR` node |
| Layers (RGBA) | Qwen-Image-Layered or Stable-Layers |
| Parallax planes | depth slices with the hidden areas behind each plane repainted (Klein) → interactive 2.5D preview |
| Objects / Cryptomatte (auto, experimental) | a local VLM lists the elements, SAM 3 masks them → Cryptomatte (object + material) in the EXR |
| Layers from Photoshop (production) | the artist's layered PSD of the plate (or a colour mask + legend): each layer becomes one Cryptomatte ID with its name and clean edges, plus Objects and per-layer mattes |
| Fog, cavity | computed from depth and normals (CPU) |
| AI light passes | clay (same light), fill/GI, occlusion, shadows, specular, emission, character/prop and plant mattes, made by image editors (Klein 3, Fibo 2 or MOSS with depth control) and aligned back to the plate |

Each pass can be upscaled to 4K individually or all at once. Image passes go through SeedVR2. Data passes (depth, normals, mattes) use clean resampling so no texture gets invented.

**New here? Read the [step-by-step guide with screenshots](docs/GUIDE.md).**

## Install

Requires **ComfyUI 0.43 or newer**. Most of the nodes RenderPasses uses ship with ComfyUI: MoGe, SAM 3, SeedVR2, Fibo 2, MOSS, Marigold, TextGenerate and Qwen-Image-Layered.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/neuropixelplus/ComfyUI-RenderPasses
pip install -r ComfyUI-RenderPasses/requirements.txt
```

Optional custom node packs:

- [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit), for the Krea 2 / Krea 2 V2 engines
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux), for the Depth Anything V2 pass. Without it, depth comes from MoGe (see *Depth* below).

Restart ComfyUI. A **RenderPasses** button appears in the top bar (also under *Extensions*). It opens the app in a new tab.

### Install check

The app checks your installation when it opens (`GET /renderpasses/requisitos`). It builds each engine's and each pass's graph, then compares the models and nodes those graphs use with what ComfyUI lists. A pill at the top shows the result. It is green when everything is installed. Otherwise it reads "*n* engines/passes unavailable — see missing models", and clicking it lists every missing file with the `models/…` folder it goes in. Missing node classes are listed with the pack to install. Engines and pass cards that can't run are unchecked and greyed out, with the missing files in a small note and in the tooltip. Optional extras never block anything and only add a note. These include SeedVR2 for upscaling, the VLM description of the layers, and MOSS for the AI passes. Beauty HDR is hidden when the LTX module isn't installed.

Files can sit in subfolders of their models folder (`loras/StableLayers/…`). The app finds them by relative path or by file name.

### Models

Install only the ones for the engines and passes you plan to use. The install check shows which ones are missing.

| Used for | Files (ComfyUI models folder) | License |
|---|---|---|
| Klein 3 engine, AI passes, parallax and object fill | `diffusion_models/flux3_image_klein_int8_convrot.safetensors`, `text_encoders/qwen3vl_4b_int8_convrot.safetensors`, `vae/flux2-vae.safetensors` | FLUX Kommunity (non-commercial) |
| Fibo 2 engine | `diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors` (or `_bf16`), `text_encoders/qwen3vl_4b_bf16.safetensors` (or `_fp8_scaled`; int8 fails with images), `vae/flux2-vae.safetensors` | CC BY-NC |
| Create (new plates from a prompt) | `diffusion_models/krea2_turbo_fp8_scaled.safetensors` (or `krea2_turbo_int8_convrot`; any Krea 2 fine-tune also works, at 12 steps), `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` (or `_int8_convrot` / `_bf16`), `vae/qwen_image_vae.safetensors` ([Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)) | Krea 2 Community |
| Krea 2 / Krea 2 V2 | `diffusion_models/krea2_turbo_int8_convrot.safetensors`, `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`, `loras/krea2_identity_edit_v1_2.safetensors` (Krea 2), `loras/krea2_general_edit_v2_r128.safetensors` (Krea 2 V2), `vae/qwen_image_vae.safetensors` | Krea 2 Community; V2 LoRA personal use |
| MOSS / MOSS · depth + color | `diffusion_models/moss_image_sft_int8_convrot.safetensors`, `text_encoders/qwen3.5_9b_w4a8_gptq.safetensors`, `vae/flux2-vae_small.safetensors`, `loras/moss_image_edit_lora_rank_64_bf16.safetensors`, `loras/moss_image_color_depth_control.safetensors` ([Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image)) | Apache-2.0 (check the adapter's license) |
| Normals / albedo | `diffusion_models/qwen_image_edit_2509_int8_convrot.safetensors` + `marigold_v2_{normals,albedo}` LoRA, conditioning and VAE | see Marigold |
| Layers | `diffusion_models/qwen_image_layered_fp8mixed.safetensors`, `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`, `vae/qwen_image_layered_vae.safetensors`; Stable-Layers LoRA `loras/StableLayers/stable_layers_comfy.safetensors` | Apache-2.0; Stability Community |
| Position / lens, MOSS · depth, depth fallback | `geometry_estimation/moge_3_vitg_fp16.safetensors` (preferred) or `moge_2_vitl_normal_fp16.safetensors` | MIT |
| Depth | `depth_anything_v2_vitl.pth` (downloaded by comfyui_controlnet_aux); without that pack, MoGe | Apache-2.0 |
| Beauty HDR (optional) | needs an LTX 2.5 SDR → HDR effect module that is **not bundled**; without it the Beauty HDR card and the "HDR with LTX" buttons are hidden | — |
| Objects / Cryptomatte, per-plate prompt | `checkpoints/sam3.1_multiplex_fp16.safetensors` + the local Qwen VLM `text_encoders/qwen3.8_27b_w4a8.safetensors` for `TextGenerate` | SAM license; Qwen license |
| 4K upscale | `diffusion_models/seedvr2_3b_int8_convrot.safetensors`, `vae/seedvr2_ema_vae_fp16.safetensors` | Apache-2.0 |

**Model licenses are your responsibility.** Several engines are non-commercial. The app shows each engine's license when you launch a batch.

## Use

1. Paste the path to a folder of BGs, or drag and drop images. Pick the ones to work on. No plates yet? Click the ✦ **Create** button next to the upload icon, describe the plate, pick the aspect ratio (16:9, 21:9, 3:2, 1:1, 9:16) and how many (1, 2 or 4), and press **Create**. The plates appear as they finish and are selected for the batch. The graph is in `workflows/renderpasses_create_krea2.json`; the API is `POST /renderpasses/crear {prompt, n, formato, semilla}`, then `GET /renderpasses/crear?id=…` until `hecho`.
2. Write or keep the **master prompt**. You can write it in English or Spanish; Spanish is translated locally with MarianMT.
3. Tick the engines and the resolution, then press **Render**. Results and contact sheets appear per batch and are saved under `ComfyUI/output/renderpasses/<batch>/`.
4. Pick the beauty you like, tick passes in **Extract passes** and run. Download `pases_nuke.exr` + `pases.nk`, open them in Nuke and composite.

The graph of every engine is saved to `user/default/workflows/RenderPasses/`, so you can open it in ComfyUI and inspect or tweak the nodes.

**Cancel.** The **Cancel** buttons in ② and ③ remove the batch's queued ComfyUI jobs and stop the running one. They also stop the batch's background tasks: Cryptomatte, objects, parallax, materials, MOSS 4K/8K tiles, HDR, the 4K upscale and the per-plate prompts. Each task stops at its next step, queues nothing more and shows **cancelled**. Launching anything new for that batch clears the cancel. Tasks started after that run normally.

### Layers from Photoshop: Cryptomatte, Objects and mattes (production)

The auto Cryptomatte guesses the objects with a local VLM and SAM 3. It's fine for a quick look, but its edges aren't production-clean. In production the background usually already exists as a layered PSD, with each prop, building or plant on its own layer. Those layers are the object mattes.

On a plate in ③, click **Layers from PSD / mask…** and pick the PSD (`.psd` or `.psb`, up to 1.2 GB) that matches that plate. Instead of a PSD you can also give a flat-colour ID mask (`.png`) together with its legend (`.json`: `[{"name": "Church dome", "color": "#d94f2b", "group": "Buildings", "material": "stone"}]`; select both files at once). The upload shows its progress. The Cryptomatte, the **Objects** (`obj_N.png`, the beauty cut out by each layer) and one matte per layer are then built in the background, and the plate's Cryptomatte card reads **from PSD · N layers**. The source stays with the plate: **Extract passes** with Cryptomatte or Objects rebuilds them from it, and the parallax planes follow its elements. **Back to AI** forgets it. `psd-tools` must be installed (it's in `requirements.txt`). If it's missing, the button explains how to install it.

- **IDs.** Every visible leaf layer becomes one CryptoObject, named with its group path, e.g. `Buildings/Church dome`. CryptoAsset is the layer's parent group and CryptoMaterial its top-level group. Hidden layers, adjustment layers, text layers and clipped layers are skipped. A clipped layer only shades its base, so it belongs to that base.
- **Edges.** Coverage comes from each layer's alpha × layer mask × opacity, including the opacity of its groups. It is composited front to back in the PSD stacking order: a layer gets `alpha × (1 − alpha of everything above it)`. Each pixel's coverage is split by visibility, as in a rendered Cryptomatte, and antialiased edges keep their partial values (up to four IDs per pixel). Coverage always sums to 1.
- **Background.** A bottom layer that covers the whole canvas only gets what no other layer covers. Pixels that no layer covers go to `background`.
- **Frame (overscan).** Production PSDs are usually larger than the camera frame. RenderPasses finds the plate inside the PSD's merged image (SIFT features and a scale-and-shift fit) and cuts every layer to that frame, so the mattes line up with the beauty pixel for pixel. The frame is saved in `cripto.json` (`marco`). If the plate cannot be found, the canvas is scaled to the beauty and, when the aspect ratio differs by more than 1 %, a warning says so.
- **Atmosphere and effects.** Fog, haze, VFX glows and cast shadows are not objects. Layers named that way (fog, niebla, haze, mist, vfx, fx, glow, shadow, sombra…), layers with a blend mode other than normal and layers under 30 % opacity get no ID and do not hide what is under them. `cripto.json` lists each one with the reason.
- **Output.** The PSD's IDs replace the auto Cryptomatte in `cryptomatte.exr`, which uses the same encoder: MurmurHash3 IDs and the manifest. They also go into the multi-layer EXR and the Nuke script's Cryptomatte node. The plate's pass folder also gets one matte per layer in `psd_capas/`, the PSD itself (`psd_fuente.psd`) and `cripto_cielo.png` when a layer is called *sky*. The shadow and parallax passes read that sky.
- **Mask + legend.** Each mask pixel goes to the nearest legend colour at the mask's own resolution, then the labels are resized to the plate, so no third colour appears at a border. Edges are softened by about one pixel for antialiasing. CryptoAsset is the legend's `group` and CryptoMaterial its `material`.
- **API.** Call `POST /renderpasses/cripto_psd` as multipart with fields `psd` (the PSD or the mask), `leyenda` (with a mask), `lote`, `motor` and `bg` (the plate's stem). From the same machine, JSON `{lote, motor, bg, ruta, leyenda}` with local paths also works, and `{lote, motor, bg, quitar: true}` goes back to the AI. Progress and the result are written to `cripto.json` and `objetos.json` (`fuente: "psd"` or `"mascara"`).

### Depth

Depth Anything V2 (comfyui_controlnet_aux) is still the preferred depth when it works. If the pack isn't installed, the depth pass runs MoGe instead. MoGe uses core ComfyUI nodes and writes `depth_moge.png`. If Depth Anything's job fails or disappears from the queue, MoGe is queued once as a backup. Parallax and the CPU passes (refined depth, fog, position, cavity) then use the MoGe depth instead of waiting for a depth that will never come. If neither source can work, parallax stops with a clear error.

**Troubleshooting.** If the depth pass fails with `No operator found for memory_efficient_attention_forward`, your `xformers` build has no CUDA kernels and Depth Anything V2 picks it up anyway. Depth then comes from MoGe automatically. To get Depth Anything back, install an `xformers` build that matches your torch/CUDA, or uninstall it (`pip uninstall xformers`) so it falls back to PyTorch attention.

### Layered Gaussian splat

Once a plate has its **Parallax** planes and **MoGe** depth, press **3D splat** in the plate's header in ③. The server turns that one flat image into a real 3D Gaussian splat and writes `splat.ply` (standard 3DGS layout) into the plate's pass folder. Open it in ComfyUI's 3D viewer or in any 3DGS viewer. For scripts: `POST /renderpasses/splat {lote, motor, bgs?}`.

In a graph, use the **RenderPasses · Layered Gaussian splat** node (`RenderPassesLayeredSplat`). Its output is ComfyUI's core `SPLAT` type, so it plugs into Render Splat, Transform Splat, Merge Splat and Create 3D File → Save Splat. Its `camera_info` output is the plate's own camera: feed it to Render Splat at the plate's aspect ratio and the render starts on the original framing. Example: `workflows/renderpasses_splat.json` (node → Render Splat → Save Image).

How it's built:

- Every pixel of every parallax plane becomes one gaussian at its 3D position: MoGe's metric depth through MoGe's lens. Colour comes from the plane, opacity from its alpha.
- The areas Klein repainted behind the front planes are placed too. Their depth is inpainted from the plane and everything behind it, then smoothed, and it is never in front of what hides it. So the planes stack in depth, and a camera move uncovers repainted content instead of holes.
- Each gaussian is a surfel that covers its pixel's footprint on the surface, so ground seen at a grazing angle stays closed. Its edges come from the neighbour on its own side of a depth jump, so nothing stretches across the gap (no rubber sheet). A loose pixel caught between two depths becomes a small disc facing the camera instead.
- The background plate gets an extra back layer behind its own depth jumps (a dome against the sky), coloured from the far side, plus a 5 % mirrored overscan past the frame edges.
- **density** sets the count: 0.625 of a 2048-px plate is 1280 splats across, about 2.1 M splats (a 145 MB `.ply`). It takes about 10 s on the CPU.

Limits: it's built from one image, so what no plane holds (beyond the overscan, or behind the background plate) is not invented. Leftovers that Klein didn't remove from the background plate show up as faint doubles when the camera moves.

### Live 3D splat (relightable)

Press **Live 3D splat** next to **3D splat** in the plate's header in ③ (it needs the **Parallax** planes and **MoGe**; **Albedo**, **Normals** and a **Cryptomatte** make it better). The server builds a relightable splat scene from the plate's passes, about 1 M gaussians in 10–20 s on the CPU, and opens the **PBR Director** in a new tab. It runs live in the browser with WebGL (three.js), so nothing is rendered in ComfyUI.

- Each gaussian carries the plate's colour plus its albedo (Albedo pass, linear), normal (Normals pass) and a roughness / metal / emission preset picked from the name of its Cryptomatte layer (sky, glass, metal, neon…). A PSD or colour-mask Cryptomatte gives the best layers.
- The ground plane is fitted with RANSAC on MoGe's metric points of the ground layers (street, floor, grass…), or of the bottom of the frame when there are none.
- In the Director: free camera (right-click + WASD flies, F focuses), three point lights (key / fill / rim) that relight the scene on a ring you drag, a mix between the original light and the PBR relight, views per pass (beauty, albedo, normal, roughness, metal, emission, layers), layer toggles, camera keys and presets, and WebM recording. Lights can follow a loaded music track.
- 3D assets: put `.glb` files in the plate's `splat3d/glb/` folder, list them in `splat3d/assets.json` (`[{"k", "url": "glb/…", "name", "giro", "vis"}]`) and their place in `splat3d/ubicaciones.json` under the same `k` (`{"x", "z", "alto"}` in metres). They stand on the measured floor, take the same lights and cast shadows on it.

Shift+click rebuilds the scene. Everything is written to `splat3d/` in the plate's pass folder (`splat_pbr.ply` with extra PBR properties, `splat_pbr.json`, `moge.json`, `ubicaciones.json`). For scripts: `POST /renderpasses/splat3d {lote, motor, bg, rehacer?}`, then open `/renderpasses/splat3d/<lote>/<motor>/<bg>/` when `estado.json` there reads `listo`.

### Windows, Linux and macOS

- Model names are written with `/` in the graphs. Right before each graph is queued, every model input is rewritten to the exact string ComfyUI lists on that OS. This covers `unet_name`, `clip_name`, `vae_name`, `lora_name`, `ckpt_name`, MoGe's `model_name` and the rest. ComfyUI uses `\` on Windows and `/` elsewhere, and the file may sit in any subfolder.
- File paths are built with `os.path`. No Windows tools are called.
- ffmpeg is only needed by the optional LTX HDR module. The app looks for it on the `PATH` (`shutil.which`) and then in `imageio-ffmpeg` if that's installed.
- The Spanish ⇄ English translator needs `transformers` and `sentencepiece` (in `requirements.txt`). MarianMT downloads its weights from Hugging Face on first use.

## How it's built

RenderPasses is a ComfyUI extension made of three parts:

- **A server module** (`rp_server.py`) that adds `/renderpasses/*` routes. It builds plain ComfyUI API graphs, mostly from core nodes and official templates, and queues them on ComfyUI's own queue. Jobs are grouped by model so models aren't swapped for every image.
- **A web app** (`web/renderpasses.html`) served by ComfyUI.
- **Custom nodes**: `RenderPassesMoGeEXR` writes MoGe-2's metric point map, depth and normals to float EXR in Nuke's camera convention. `RenderPassesLayeredSplat` (`rp_splat.py`) builds the layered Gaussian splat as core `SPLAT`.

CPU passes (raw lighting, fog, cavity, position from depth), the Cryptomatte encoder (MurmurHash3 IDs, OpenImageIO), the PSD layer reader (`rp_psd.py`, psd-tools) and the Nuke script writer are plain Python and NumPy.

## License

**AGPL-3.0-or-later.** © 2026 NeuroPixel+. You can use, study, change and share RenderPasses. If you distribute it, or let people use a modified version over a network (for example as a hosted web service), you must offer them the complete source of your version under the same license. The AGPL is compatible with ComfyUI's GPL-3.0.

The names **RenderPasses** and **NeuroPixel+** and their logos are not licensed under the AGPL: see [TRADEMARKS.md](TRADEMARKS.md).

The models RenderPasses runs have their own licenses (see *Models* above); several are non-commercial.

---

### En español

RenderPasses convierte una carpeta de fondos 2D planos en pases de composición: depth, normales, albedo, capas, planos de paralaje, position, Cryptomatte y pases de luz con IA. Todo sale en un EXR multicapa con su script de Nuke, y usa solo modelos de pesos abiertos dentro de ComfyUI. Si no tienes fondos, el botón ✦ **Create** los crea desde un prompt con Krea 2. Antes de sacar los pases puede reiluminar o mejorar los fondos con un prompt maestro y el motor que elijas. El Cryptomatte de producción sale de las capas del PSD del artista, aunque el PSD tenga overscan. Para instalarlo, clona el repo en `ComfyUI/custom_nodes`, instala `requirements.txt` y reinicia ComfyUI; el botón **RenderPasses** abre la app. El prompt se puede escribir en español.
