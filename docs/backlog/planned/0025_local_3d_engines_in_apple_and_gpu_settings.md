# Planned: Local 3D engines in the apple and gpu settings

## Metadata

- Created: 2026-09-29
- Status: Planned
- Completed: N/A
- Priority: P2 (operator ruling 2026-09-29: "ideally it should be everywhere, but like
  abstractvision it is certain that it's not compatible with gpu for the moment, so most probably
  a planned backlog item")
- Area: packaging, device portability, AbstractCore integration

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_scene3d_local_first_glb_contract.md) (local-first;
  "CUDA-heavy research stacks are intentionally not first-class defaults", line 31),
  [ADR 0002](../../adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
  (TripoSR is the validated default; validation means a checked proof run),
  [ADR 0003](../../adr/0003_trellis2_uses_official_upstream_assets_only.md) (TRELLIS.2 official
  assets, gated DINOv3 companion),
  [ADR 0004](../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
  (Step1X geometry only, local compatibility patches, `float32` on `mps`),
  [ADR 0005](../../adr/0005_hunyuan3d21_license_gated_official_shape_backend.md) (Hunyuan3D-2.1
  license gate: EU, UK and South Korea excluded; runtime acknowledgement required).
- ADR impact: May need a new ADR. The rule this item applies ("an AbstractCore install setting
  carries a local engine only when that engine has a recorded run on that setting's hardware")
  is cross-task policy. If it is kept after this item closes, record it as an ADR rather than
  leaving it in backlog prose. Installing the Hunyuan3D runtime inside `abstractcore[apple]` /
  `abstractcore[gpu]` does not change ADR 0005: the licence gate stays a runtime acknowledgement,
  not an install-time choice.

---

## Main goals

- An AbstractCore user who installs `abstractcore[apple]` gets the local 3D engines that run on
  Apple silicon today, with no separate `abstract3d[...]` step.
- An AbstractCore user who installs `abstractcore[gpu]` (NVIDIA on Linux) gets the local 3D
  engines that have a recorded NVIDIA run. No engine goes into a setting without one.
- Each setting installs only what runs on it, and adding abstract3d never downgrades or
  duplicates another AbstractCore dependency.

## Secondary goals

- One capability table (`docs/models.md` + `model_catalog.py`) that says, per engine and task,
  whether it runs on `apple`, on `gpu` (NVIDIA/Linux) and on Windows, with measured time and peak
  memory where it runs. Memory figures are labelled as engine measurements, not model
  requirements.
- Windows gets an explicit status for each engine ("runs on CUDA", "CPU only", or "not
  supported"). Today the README lumps it in with Linux as "implemented, not validated".

---

## Context / problem

AbstractCore has exactly three install settings: light (`pip install abstractcore`),
`abstractcore[apple]` and `abstractcore[gpu]`. On today's code (checked 2026-09-29):

- AbstractCore's base install depends on `abstract3d>=0.1.0; python_version >= '3.10'`
  (`abstractcore/pyproject.toml:98`). That gives the plugin and the remote/composed contract but
  no local engine: abstract3d's base install is only `abstractvision>=0.3.27`
  (`abstract3d/pyproject.toml:33-35`).
- Neither `abstractcore[apple]` (`abstractcore/pyproject.toml:144-172`) nor `abstractcore[gpu]`
  (`:174-194`) pulls `abstract3d[apple]` or `abstract3d[gpu]`. So **no local 3D engine is in any
  AbstractCore setting.** The root `abstractframework` profile pins `abstract3d==0.3.1` in its
  base (`pyproject.toml:69`) and reaches AbstractCore's settings only through
  `abstractgateway[apple|gpu]`, so it also has no local 3D engine.
- Commit `b074378` (2026-09-29) made the AbstractCore plugin's install hint say exactly that
  (`src/abstract3d/integrations/abstractcore_plugin.py:8-18`): "The local {engine} engine is not
  available with AbstractCore's install settings (light, apple, gpu)".
- abstract3d 0.3.1 has the extras `triposr`, `trellis2`, `step1x`, `hunyuan3d`, `t23d` (empty
  compatibility alias, `pyproject.toml:127-128`), `mesh`, `apple`, `gpu`, `all-apple` and
  `all-gpu`. `apple` and `gpu` list identical runtime packages (`:143-173` and `:175-205`); they
  differ only in which `abstractvision[...]` extra they pull.
- The README platform table (`README.md:53-57`) says macOS Apple silicon is **validated** ("the
  entire proof and certification record was produced on this profile"). Linux and Windows with
  NVIDIA or AMD are "Implemented (`abstract3d[gpu]` extra, `--device cuda`), **not validated** —
  no checked proof run exists on these hosts". No backlog record (planned, proposed or completed)
  contains an NVIDIA, CUDA or Windows run.

The problem: the `apple` setting could carry engines that are proven on Apple silicon but
doesn't, and the `gpu` setting can't honestly carry any engine yet, because none has ever run on
NVIDIA. On top of that, the packaging has concrete defects, found below, that must be fixed
before either setting can take abstract3d.

---

## Current code reality

### Per-engine, per-OS status (code read 2026-09-29; nothing was run on NVIDIA or Windows)

Every runtime clones its pinned upstream at first use into
`~/.cache/abstract3d/vendor/<engine>/<commit>` through `git` (for example
`triposr_runtime.py:29-30,706-744`, `step1x_runtime.py:44-45`, `hunyuan3d_runtime.py:94-95,386-388`,
`trellis2_runtime.py:34-35,250-251`). None of that upstream code is vendored under `src/`, and
every host needs `git` on `PATH`. Every runtime's `_select_device` prefers `mps`, then `cuda`,
then `cpu` (`triposr_runtime.py:688-705`, `step1x_runtime.py:182-197`,
`hunyuan3d_runtime.py:315-330`, `trellis2_runtime.py:110-128`). The code paths are
device-agnostic, but the MPS memory telemetry and caps are MPS-only (`step1x_runtime.py:932-968`,
`trellis2_runtime.py:1340`). A CUDA run would record no peak memory today.

| Engine | Apple silicon (`mps`) | NVIDIA / Linux (`cuda`) | Windows |
| --- | --- | --- | --- |
| TripoSR (`triposr`) | **Runs, validated default** (ADR 0002; `docs/models.md:9`; 20-60 s/object, `README.md:63`; 1.6 GB cache, `docs/models.md:13`). | **To measure.** No CUDA-only code: upstream `torchmcubes` (a CUDA/C++ build in upstream `requirements.txt`) is replaced by a scikit-image shim (`triposr_runtime.py:572-621`). The atlas bake falls back to a CPU rasterizer without GL (`:1279-1300`). Packaging: blocked on manylinux_2_28 (pymeshlab), resolves on 2_35. | **To measure; CPU only as packaged.** Resolves, but PyPI's `torch` for `win_amd64` is the CPU build (124 MB wheel), so `--device cuda` falls back to CPU. |
| Step1X geometry (`step1x`) | **Runs, experimental** (completed 0001; planned 0003: chair/owl below bar, espresso `i23d` over safe memory). `float32` on `mps` (ADR 0004). | **To measure.** Upstream is CUDA-native: abstract3d patches the hardcoded `cuda:{rank}` `get_device()` (`step1x_runtime.py:1469-1481`). `sageattention` (env `USE_SAGEATTN=1`), `diso` (`mc_algo` dmc only) and `torch_cluster` (`use_downsample` encoder path) are optional upstream imports, not in abstract3d's runtime import list (`step1x_runtime.py:63-79`). The default dtype on CUDA is `float16` (`:215-216`) and has never been run. Blocked on manylinux_2_28. | **To measure; CPU only as packaged** (same torch wheel). |
| Hunyuan3D-2.1 / 2mv (`hunyuan3d`) | **Runs, experimental, license-gated** (completed 0013/0014; 7-13 min/object at 30 steps, `README.md:64`). `float16` on `mps`. | **To measure.** The shape stage (`hy3dshape`) has only optional CUDA extras (`sageattention` env-gated; `diso`; `torch_cluster` fps). Texture uses abstract3d's projection bake, not upstream `hy3dpaint` (CUDA `custom_rasterizer`, `bpy`, `cupy-cuda12x` in upstream `requirements.txt`). The CUDA chunk default (`_DEFAULT_NUM_CHUNKS = 8000`, `hunyuan3d_runtime.py:275`) is untested. Blocked on manylinux_2_28. | **To measure; CPU only as packaged.** |
| TRELLIS.2 (`trellis2`) | **Blocked at runtime** on the gated DINOv3 companion (`docs/models.md:105`; `README.md:67`; ADR 0003). | **Blocked (same gate), then to measure.** abstract3d forces its pure-PyTorch shims on every device: `SPARSE_CONV_BACKEND=none`, `ATTN_BACKEND=sdpa`, a local `conv_none`, and an `o_voxel.convert` stand-in (`trellis2_runtime.py:531-556,631-650`). So on CUDA it runs the slow shim path, not upstream's CUDA stack (`setup.sh` flags `--flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast`). The `trellis2` extra resolves everywhere, including manylinux_2_28 (no pymeshlab). | **Blocked (same gate); CPU only as packaged.** |
| Composed `t23d` | **Runs.** The image comes from the host's `llm.vision` `t2i` or AbstractVision's plugin (`image_composition.py:240-280`). Validated with `AbstractFramework/flux.2-klein-4b-8bit` through MLX (`docs/models.md:133-137`). | **Blocked on abstractvision task 030.** The `gpu` text-to-image engine is Diffusers, whose FLUX.2 Klein / Qwen-Image path has no recorded NVIDIA run yet (`abstractvision/docs/backlog/planned/030_gpu_parity_with_mlx_gen.md`). A remote image provider works today. | Same as NVIDIA, CPU torch. |
| Mesh ops (`mesh`) | **Runs** (numpy/trimesh/scipy/matplotlib only, `pyproject.toml:130-141`). | **Runs at library level.** The CI test suite runs on `ubuntu-latest` with CPU torch under `xvfb` (`.github/workflows/release.yml:21,32,42-43`). Resolves on 2_28 and 2_35. | **Resolves; to measure** (no Windows CI). |
| Pixal3D (planned 0024) | **Not implemented.** Phase 1 of 0024 must first show that the CUDA-only `flex_gemm.grid_sample_3d` texture path has a non-CUDA replacement. | **Not implemented.** Upstream is CUDA-first (`flex_gemm`, `cuda` hardcoded about 155 times, per 0024), so NVIDIA is its natural host, but 0024's non-goals exclude an NVIDIA-only supported path. To decide in 0024. | Not implemented. |

### GL and preview rendering on headless hosts

`rendering.py:148-166` creates `moderngl.create_context(standalone=True)` with no `backend=`
argument, so on Linux glcontext needs an X display (CI wraps pytest in `xvfb-run` and installs
`libegl1 libgl1`, `release.yml:42-43`). Without GL, the atlas bake falls back to CPU
(`triposr_runtime.py:1279-1300`), depth occlusion falls back to facing-only visibility
(`:1636-1648`, "change the bake output"), and previews fall back to matplotlib
(`pyproject.toml:130-134`). A headless NVIDIA server therefore produces a *different* bake from a
Mac. To measure: whether `backend="egl"` works on NVIDIA/Linux, and how far the facing-only bake
drifts.

### Packaging reality (`uv pip compile`, Python 3.12, 2026-09-29)

Command shape: `uv pip compile <in> --python-version 3.12 --python-platform <target> --no-header
--no-annotate -q` (`MACOSX_DEPLOYMENT_TARGET=14.0` for macOS). The inputs and outputs are kept
under `untracked/camera-3d/3d/compile/` in the framework root (not committed). Sizes are the sum
of the chosen PyPI wheels (approximate, installed size is larger).

| Target | Input | Result |
| --- | --- | --- |
| aarch64-apple-darwin | `abstract3d[apple]==0.3.1` | resolves: 112 packages, about 569 MB of wheels (torch 127, pymeshlab 71, mlx-metal 68, opencv-python-headless 46, opencv-python 46) |
| aarch64-apple-darwin | each of `triposr` / `trellis2` / `step1x` / `hunyuan3d` / `t23d` / `mesh` | all resolve (77 / 48 / 94 / 78 / 2 / 15 packages) |
| x86_64-manylinux_2_28 | `abstract3d[gpu]`, `[triposr]`, `[step1x]`, `[hunyuan3d]` | **fail**: `pymeshlab>=2023.12,<2025.0` (`pyproject.toml:59,99,117,159,191`) has wheels only for `manylinux_2_31_x86_64`, macOS and `win_amd64`. `[trellis2]` and `[mesh]` resolve. |
| x86_64-manylinux_2_35 | `abstract3d[gpu]==0.3.1` | resolves: 130 packages, about 3.6 GB (torch 2.14.0 with the CUDA 13 `nvidia-*` wheels, triton, and `mlx-cuda-13` pulled by `abstractvision[gpu]==0.3.31`, which is fixed by the unreleased abstractvision 0.3.32) |
| x86_64-pc-windows-msvc | `abstract3d[gpu]==0.3.1` | resolves: 102 packages, about 409 MB. **torch is the 124 MB CPU wheel**: no CUDA from PyPI on Windows |
| aarch64-apple-darwin | `abstractcore[apple]==2.19.0` + `abstract3d[apple]==0.3.1` | resolves. It adds about 139 MB (30 package changes: pymeshlab 71, opencv-python-headless 46, scikit-image 12, rembg, timm, pytorch-lightning, xatlas, moderngl, PyMCubes...). But it **downgrades `psutil` 7.2.2 → 6.1.1 and therefore `unstructured` 0.27.10 → 0.18.32** (unstructured 0.27.10 requires `psutil>=7.2.2`; abstract3d pins `psutil>=5.9.0,<7.0.0` in every extra). |
| x86_64-manylinux_2_35 | `abstractcore[gpu]==2.19.0` + `abstract3d[gpu]==0.3.1` | resolves. It adds about 188 MB and has the same psutil/unstructured downgrade. It also downgrades `opencv-python-headless` 5.0.0.93 → 4.14.0.94 (abstract3d pins `<5.0.0`). rembg drops to 2.0.69 because the vLLM stack holds numpy at 2.2.6 and rembg 2.0.85 needs `numpy>=2.3`. torch stays at vLLM's 2.8.0 (CUDA 12.8). |
| x86_64-pc-windows-msvc | `abstractcore[gpu]==2.19.0` + `abstract3d[gpu]==0.3.1` | resolves. It adds about 116 MB, with the same psutil/unstructured and opencv downgrades. |
| x86_64-manylinux_2_28 | `abstractcore[gpu]==2.19.0` (± abstract3d) | **fails without abstract3d already** (`mlx[cuda13]` via `abstractvision[all-gpu]==0.3.31` needs manylinux_2_35; fixed in unreleased abstractvision 0.3.32). With abstract3d it also fails on pymeshlab. |
| every target, `--only-binary :all:` | the rows above | all fail, but not on abstract3d's own compiled packages. The only sdist in abstract3d's closure is `antlr4-python3-runtime==4.9.3` (via omegaconf; pure Python, and AbstractCore already carries it). PyMCubes 0.1.6 and easydict 1.13 have wheels. The other failures come from the rest of the stack: `stable-diffusion-cpp-python==0.4.5` (via `abstractvision[apple]`) on macOS, and `llama-cpp-python` for the AbstractCore settings. |

Two `cv2` owners are installed side by side: `opencv-python` (via `mlx-gen` / `mlx-vlm`, and
AbstractCore's stack) and `opencv-python-headless` (via abstract3d). Both write the same `cv2/`
package, so whichever installs last wins. This already happens in standalone
`abstract3d[apple]` / `[gpu]` on macOS and Linux.

`onnxruntime` (CPU) is the only ONNX runtime in every resolution. rembg runs on CPU, and there is
no `onnxruntime` / `onnxruntime-gpu` double install. Keep it that way: rembg's own `gpu` extra
would pull `onnxruntime-gpu`, which clashes with `onnxruntime`.

---

## Constraints

- Keep the `scene3d` contract, the `glb`-first artifact (ADR 0001) and the AbstractCore plugin
  contract stable. Heavy imports stay lazy.
- Official upstream assets only (ADR 0003, 0004, 0005). Keep the licence gates (Hunyuan3D
  territory acknowledgement, TRELLIS.2 DINOv3 access) as runtime gates. Installing a runtime must
  never imply licence acceptance.
- Never claim an engine on `gpu` or Windows without a real run on that hardware: the capability
  table marks only what was measured (same rule as ADR 0002's validation discipline and
  abstractvision 030).
- A setting installs only what runs on it. No CUDA-only kernel packages (flex_gemm, cumesh,
  o-voxel, nvdiffrast, flash-attn, spconv, sageattention, diso, torch-cluster, cupy) in any
  default setting, unless an engine is shown to need one and the item that admits it says so.
- Memory figures are engine measurements, labelled as such.
- No machine-specific heuristics. Device choice stays explicit (`--device`,
  `ABSTRACT3D_DEVICE`, `scene3d_device`).
- The AbstractCore-facing message names only `abstractcore`, `abstractcore[apple]` and
  `abstractcore[gpu]` (operator ruling 2026-09-29, `abstractcore_plugin.py:9-11`).

---

## Research, options, and references

- **Option A: `apple` first, `gpu` later (recommended).** Fix the packaging defects below. Then
  add `abstract3d[apple]` to `abstractcore[apple]` for the engines that already run on Apple
  silicon (TripoSR validated; Step1X and Hunyuan3D experimental; mesh ops; composed `t23d`
  through MLX-Gen). `gpu` gets abstract3d only after step 4's NVIDIA runs.
- **Option B: both settings at once, engines labelled "not validated" on `gpu`.** This is rejected
  for the default settings: it breaks the "setting installs only what runs" rule and the
  README's own honesty line, and would ship CUDA fp16 paths that have never executed.
- **Option C: a slimmer `abstract3d[apple]` / `[gpu]`.** Drop TRELLIS.2-only and research-only
  needs, and make pymeshlab optional (TripoSR's cleanup already catches its failure,
  `triposr_runtime.py:1076-1105`, "skipped" warning). Measure whether the pymeshlab-free path
  changes TripoSR output on the proof cases before choosing this.
- **pymeshlab floor.** `pymeshlab 2025.7.post1` ships `manylinux_2_35` / macOS / `win_amd64`
  wheels, and `<2025.0` ships `manylinux_2_31`. Either pin choice gives the `gpu` setting a glibc
  floor of 2.31 or higher. abstractvision 0.3.32 just removed the 2.35 floor from its `gpu`
  setting, so don't bring one back through abstract3d. To measure: why `<2025.0` was pinned
  (no rationale in CHANGELOG, docs or backlog) and whether 2025.7 changes TripoSR or Hunyuan
  cleanup output.
- **Windows CUDA.** PyPI torch on Windows is CPU-only. CUDA needs
  `https://download.pytorch.org/whl/cu*`, which an extra cannot select. Either Windows is
  documented as CPU-only for local 3D, or the installer grows an index step. This decision
  belongs to the root installer owner, not to this package.
- References: `https://pypi.org/project/pymeshlab/#files`, `https://pypi.org/project/unstructured/`
  (`requires_dist`: `psutil>=7.2.2`), `https://pypi.org/project/rembg/` (`numpy>=2.3`, extra
  `gpu` → `onnxruntime-gpu`), `https://github.com/moderngl/glcontext` (EGL backend),
  abstractvision planned 030 (`gpu` text-to-image parity).

---

## Plan

1. **Packaging hygiene (no engine changes).** Widen `psutil` to `<8.0.0` in every extra (it
   blocks unstructured 0.27). Decide the `opencv-python-headless <5` pin and the dual-`cv2` issue
   with the abstractvision owner (one `cv2` distribution per setting). Decide pymeshlab
   (floor 2.31 accepted, 2025.7 adopted, or made optional per option C). Add a hermetic
   packaging test that compiles `abstract3d[apple]` on aarch64-apple-darwin and `abstract3d[gpu]`
   on manylinux_2_28 and 2_35 and on Windows, and goes red on a resolution failure or on a
   psutil/unstructured downgrade when combined with the current `abstractcore[apple|gpu]`.
2. **CUDA telemetry.** Record `torch.cuda.max_memory_allocated` / reserved next to the existing
   MPS stats in every runtime, so an NVIDIA run leaves the same evidence an Apple run does.
3. **Headless GL.** On Linux without `DISPLAY`, try `moderngl.create_context(standalone=True,
   backend="egl")` before falling back. Record which path the bake took in generation metadata.
4. **NVIDIA measurement (one Linux host, CUDA 12.8 torch as AbstractCore's `gpu` resolves it).**
   TripoSR `i23d` + composed `t23d` on the four proof cases. Then Step1X and Hunyuan3D-2.1
   (`float16` defaults as coded, then `float32` if they fail). Record time, peak memory and
   contact sheets under `artifacts/validation/cuda/`. TRELLIS.2 only if the operator has DINOv3
   access.
5. **Windows statement.** One CPU smoke run of TripoSR on Windows (`git` present), or an explicit
   "CPU only / not supported" line per engine.
6. **Capability table.** Update `docs/models.md`, `model_catalog.py`, the README platform table
   (it still says "Current State (v0.2.0)" at `README.md:45` while the package is 0.3.1) and
   `docs/faq.md:60-64` ("Does it require CUDA?") with per-engine `apple` / `gpu` / Windows status.
7. **AbstractCore `apple` setting** (AbstractCore owner, not this repo): add
   `abstract3d[apple]>=<fixed version>; python_version >= '3.10'` to `abstractcore[apple]`.
   Replace the plugin install hint (`abstractcore_plugin.py:8-18`): on `apple` the local engines
   are carried; on light the hint names `abstractcore[apple]`. Update AbstractCore's
   `docs/installation.md` settings table.
8. **AbstractCore `gpu` setting** (after step 4, and after abstractvision 030 for local `t23d`):
   add `abstract3d[gpu]` to `abstractcore[gpu]`, limited to the engines with a recorded NVIDIA run
   (via a trimmed extra if needed). Update the hint and docs again.
9. **Root profile** (framework owner): bump the root pin `abstract3d==<version>` and state in
   `docs/install.md` which profile carries local 3D. Nothing else changes: the root profile
   reaches the engines through `abstractgateway[apple|gpu]` → AbstractCore.

---

## Scope

- abstract3d's extras and pins, CUDA memory telemetry, the headless GL path, NVIDIA measurement
  runs, a Windows status statement, and capability docs.
- The AbstractCore and root changes above, as handoff steps with acceptance criteria. They are
  executed by those packages' owners.

## Non-goals

- Upstream CUDA kernels (flex_gemm, cumesh, o-voxel, nvdiffrast, flash-attn) in any setting, or
  switching TRELLIS.2 off its pure-PyTorch shims on CUDA. That is a separate performance item, if
  ever wanted.
- Pixal3D. Planned 0024 decides whether it exists at all. This item only records its status.
- Promoting any engine over TripoSR as the validated default (ADR 0002).
- AMD ROCm validation. It is out of scope until an operator names a ROCm host.
- Editing abstractcore, abstractvision or the root package from this repository.

## Dependencies and related tasks

- abstractvision planned 030 (`gpu` parity with MLX-Gen): local composed `t23d` on `gpu` needs
  its Diffusers text-to-image path validated on NVIDIA. abstractvision 0.3.32 (unreleased on
  2026-09-29, PyPI latest 0.3.31) removes `mlx-cuda-13` and the glibc 2.35 floor from
  `abstractvision[gpu]`, which `abstract3d[gpu]` inherits.
- [0024 Pixal3D backend](0024_pixal3d_pixel_aligned_i23d_backend.md): its Phase 1 go/no-go
  decides the Pixal3D column.
- [0003 Step1X quality tuning on Apple Silicon](step1x/0003_step1x_quality_tuning_on_apple_silicon.md):
  Step1X stays experimental in either setting until it lands.
- Completed [0015 AbstractCore scene3d integration](../completed/integration/0015_abstractcore_scene3d_integration.md).
- Code: `pyproject.toml`, `src/abstract3d/backends/*_runtime.py` (`_select_device`, memory stats),
  `src/abstract3d/rendering.py`, `src/abstract3d/image_composition.py`,
  `src/abstract3d/integrations/abstractcore_plugin.py`, `.github/workflows/release.yml`.
- AbstractCore: `pyproject.toml:94-98,144-194`, `docs/installation.md`.

## Expected outcomes

- `abstractcore[apple]` installs TripoSR, Step1X, Hunyuan3D (gated) and mesh ops locally, with no
  downgrade of any AbstractCore dependency.
- `abstractcore[gpu]` carries exactly the engines with a recorded NVIDIA run. Until then it
  carries none, and the hint says so.
- The docs state per engine what runs on `apple`, `gpu` and Windows, with evidence paths.

## Acceptance criteria

- [ ] `abstract3d[apple]` and `abstract3d[gpu]`, combined with the current `abstractcore[apple]` /
      `abstractcore[gpu]`, resolve with no downgrade of `psutil`, `unstructured` or
      `opencv-python-headless` compared with AbstractCore alone, and install exactly one `cv2`
      distribution.
- [ ] `abstract3d[gpu]` resolves on x86_64 manylinux_2_28, or the glibc floor is chosen
      explicitly and written in the install docs.
- [ ] A packaging test goes red when any of the above regresses (delete the fix once and watch
      it fail).
- [ ] Every runtime records CUDA peak memory when it runs on `cuda`.
- [ ] Headless NVIDIA/Linux bakes use a real GL context (EGL) or record that they fell back.
- [ ] TripoSR (`i23d` + composed `t23d` with a remote or validated local image source),
      Step1X and Hunyuan3D-2.1 each have a recorded NVIDIA run (time, peak memory, contact
      sheet), or are marked "not on gpu yet" with the failure preserved.
- [ ] The capability table (`docs/models.md`, `model_catalog.py`, README platform table,
      `docs/faq.md`) lists `apple` / `gpu` / Windows status per engine; every "yes" has a run.
- [ ] AbstractCore (owner handoff): `abstractcore[apple]` depends on `abstract3d[apple]`.
      `abstractcore[gpu]` depends on `abstract3d[gpu]` (or a trimmed extra) once the NVIDIA runs
      exist. The plugin install hint names the setting that carries the engine.
- [ ] Root (owner handoff): `abstract3d` pin bumped and `docs/install.md` says which profile
      carries local 3D.

## Validation

- Hermetic: `uv pip compile` for `abstract3d[apple]` (aarch64-apple-darwin,
  `MACOSX_DEPLOYMENT_TARGET=14.0`), `abstract3d[gpu]` (x86_64 manylinux_2_28, 2_35,
  x86_64-pc-windows-msvc), each alone and with `abstractcore[apple|gpu]` at the current release.
  Diff the combined lock against AbstractCore alone. `pytest -q tests` on the existing CI matrix
  (`ubuntu-latest`, `macos-14`).
- Real runs: `scripts/validate_local.py --backend triposr --device cuda` (and step1x, hunyuan3d)
  on one NVIDIA Linux host, case-isolated and memory-guarded, with bundles under
  `artifacts/validation/cuda/`. Failures are preserved, not discarded. No proof asset is published
  to `docs/assets/validation/` without visual review.
- Apple regression: the existing Apple proof lane re-run after the packaging changes (pymeshlab /
  opencv / psutil) shows unchanged TripoSR output on the four proof cases.

## Progress checklist

- [ ] Step 1: psutil / opencv / pymeshlab decisions and packaging test
- [ ] Step 2: CUDA memory telemetry in every runtime
- [ ] Step 3: EGL headless context and bake-path metadata
- [ ] Step 4: NVIDIA runs (TripoSR, Step1X, Hunyuan3D; TRELLIS.2 if DINOv3 access)
- [ ] Step 5: Windows statement
- [ ] Step 6: capability table and stale README/FAQ lines
- [ ] Step 7: AbstractCore `apple` handoff
- [ ] Step 8: AbstractCore `gpu` handoff
- [ ] Step 9: root profile handoff

## Guidance for the implementing agent

Re-run the compiles first. Every figure above is from 2026-09-29 against PyPI as it was then
(abstractcore 2.19.0, abstractvision 0.3.31, abstract3d 0.3.1). Once abstractvision 0.3.32
publishes, the manylinux_2_28 and `mlx-cuda-13` rows change. Do not add `abstract3d[gpu]` to
AbstractCore on the strength of "the code is device-agnostic": the Step1X and Hunyuan3D CUDA
defaults (`float16`, chunk sizes) have never executed. Steps 7-9 belong to other owners; hand
them over as addressed asks with this item's evidence, don't edit their repos from here.
