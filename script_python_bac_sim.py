# =============================================================================
# THIOVULUM PIV & MORPHOMETRIC ANALYSIS PIPELINE — Windows Edition v1
# =============================================================================
# Authors  : Sébastien REGIS, Zohra MARIE, Elona OUSSELIN, Olivier GROS
# Purpose  : Reproducible PIV + morphometry + statistical comparison
#            between in vivo Thiovulum bacterial videos and MAS simulations.
# Platform : Windows 10/11, Python 3.11
#
# INSTALLATION (One-time setup, in a Windows terminal):
#    python -m venv venv_thiovulum
#    venv_thiovulum\Scripts\activate
#    pip install opencv-python numpy scipy scikit-image matplotlib openpiv
#
# USAGE:
#    1. Place the videos in VIDEO_DIR (see Section 1)
#       - Bacteria   : Bac_1.MP4, Bac_2.MP4, ..., Bac_30.MP4
#       - Simulations: Simu_1.MP4, Simu_2.MP4, ..., Simu_100.MP4
#    2. Run: python script_python_bac_sim.py
#    3. Results will be saved in OUTPUT_DIR:
#       - thiovulum_statistics.csv
#       - thiovulum_boxplots.png
#       - thiovulum_detection_check.png  (visual verification)
# =============================================================================

import sys, cv2, numpy as np, scipy, scipy.stats as stats
from scipy.ndimage import label, uniform_filter
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from pathlib import Path
import warnings, csv, time
warnings.filterwarnings("ignore")

try:
    from openpiv import pyprocess as piv_proc
    from openpiv import validation
    OPENPIV_AVAILABLE = True
except ImportError:
    OPENPIV_AVAILABLE = False

import skimage
print("=" * 60)
print("LIBRARY VERSIONS")
print(f"  Python       : {sys.version.split()[0]}")
print(f"  OpenCV       : {cv2.__version__}")
print(f"  NumPy        : {np.__version__}")
print(f"  SciPy        : {scipy.__version__}")
print(f"  scikit-image : {skimage.__version__}")
print(f"  OpenPIV      : {'OK' if OPENPIV_AVAILABLE else 'absent — fallback FFT'}")
print("=" * 60)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SECTION 1 — CONFIGURATION  ← MODIFY HERE                               │
# └─────────────────────────────────────────────────────────────────────────┘

# Directory containing ALL videos (Bac_x.MP4 and Simu_x.MP4)
VIDEO_DIR = Path(r"C:\Users\sebas\OneDrive\Bureau\Travail\Recherche\Collaboration"
                 r"\Collaboration_Olivier_Gros\Stage_MARIE_L2SVT"
                 r"\Elements_pour_soumission_article\Donnees_et_resultats"
                 r"\Elements_de_redaction\Article\Soumission_Biosystems"
                 r"\Calcul_PIV_Claude")

# Output directory (automatically created if it does not exist)
# Short name ("out" instead of "results") to stay under the Windows 
# MAX_PATH limit (260 characters) — the VIDEO_DIR path above is already very long.
OUTPUT_DIR = VIDEO_DIR / "out"

def long_path(p: Path) -> str:
    """
    Prefixes a Windows path with \\\\?\\ to bypass the MAX_PATH limit 
    (260 characters). Without this prefix, functions like PIL.Image.save() 
    or open() fail silently on long paths (common with OneDrive, which 
    extends path lengths).
    Has no effect on macOS/Linux (returns the path as-is).
    """
    s = str(p.resolve())
    if sys.platform.startswith("win") and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s

# Video number ranges
BAC_RANGE  = range(1, 31)    # Bac_1.MP4 to Bac_30.MP4
SIMU_RANGE = range(1, 101)   # Simu_1.MP4 to Simu_100.MP4

# Video file extension (case-sensitive on some systems)
VIDEO_EXT = ".MP4"           # Change to ".mp4" if necessary

# Optical calibration (leave as None if unknown → results in px/frame)
PIXEL_TO_UM = None          # e.g., 0.83 to convert to µm/s

# ── PIV Parameters ──
PIV_WINDOW    = 64    # Interrogation window size (pixels)
PIV_OVERLAP   = 32    # Overlap between windows (pixels)
PIV_DOWNSCALE = 2    # Downscaling factor before PIV (2 = half resolution)
                     # Set to 1 for full resolution (slower)
SAMPLE_HZ     = 1     # Analyzed frame pairs per second of video
PIV_MIN_STD   = 3.0   # Minimum standard deviation (grayscale values) within an
                     # interrogation window to be considered "textured". Below this,
                     # the window is deemed empty (uniform background) and excluded.
                     # Crucial for NetLogo simulation videos where a large part
                     # of the image is a blank background without agents.

# ── Bacteria Detection (white plume on dark background) ──
BAC_V_THRESH = 180   # HSV Value (brightness) threshold
BAC_S_THRESH = 60    # HSV Saturation threshold (low = white/grey)
BAC_MORPH_K  = 7     # Morphological kernel size (pixels)

# ── NetLogo Simulation Detection ──
# Simulation zone within the NetLogo window (pixel coordinates)
# Measured on simu_1.MP4 using a grid — ADJUST if resolution changes
SIM_CROP = (250, 167, 729, 675)   # (x1, y1, x2, y2)
# Coordinates measured on simu_1.MP4 (1920×1080) with a 5px margin
# to exclude the grey border of the NetLogo frame.
# Adjust if the NetLogo screen resolution changes.
SIM_S_THRESH = 40    # HSV saturation threshold to isolate colored agents
# Lowered from 60 to 40 to capture dark blue agents
# (S ∈ [40,60]) without introducing false positives on the white background
# (white background: S < 20, which is well below the threshold of 40).
SIM_MORPH_K  = 5     # Morphological kernel size (pixels)

FPS_DEFAULT = 30.0   # Used if video metadata is missing


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SECTION 2 — DETECTION                                                 │
# └─────────────────────────────────────────────────────────────────────────┘

def detect_bacterial_plume(frame: np.ndarray) -> np.ndarray:
    """
    Isolates the bacterial plume in an in-vivo frame.
    Strategy: HSV color space — the white/grey plume exhibits high
    brightness (V > BAC_V_THRESH) and low saturation (S < BAC_S_THRESH).
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:,:,2] > BAC_V_THRESH) &
            (hsv[:,:,1] < BAC_S_THRESH)).astype(np.uint8) * 255
    k_c  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BAC_MORPH_K, BAC_MORPH_K))
    k_o  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_c)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_o)
    return mask

def detect_agent_group(frame: np.ndarray) -> np.ndarray:
    """
    Isolates the group of agents in a NetLogo frame.
    Strategy:
      1. Crop to the simulation area (SIM_CROP) to ignore
         NetLogo control panels.
      2. Detect colored pixels (S > SIM_S_THRESH) = red/green/blue
         agents on a white background.
    Returns a mask of the cropped area size (not the full frame).
    """
    x1, y1, x2, y2 = SIM_CROP
    crop = frame[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[:,:,1] > SIM_S_THRESH).astype(np.uint8) * 255
    k_c  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SIM_MORPH_K, SIM_MORPH_K))
    k_o  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_c)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_o)
    return mask


def get_gray_for_piv(frame: np.ndarray, video_type: str) -> np.ndarray:
    """
    Prepares the frame in grayscale for PIV processing.

    For bacteria: Direct conversion to grayscale on the entire frame.

    For simulation: Two critical steps before PIV:
      1. Crop to the simulation area (excludes the NetLogo interface).
      2. Background-to-black masking: White background pixels (S < SIM_S_THRESH)
         are set to 0. Only colored agents remain visible.
         Without this masking, the uniform white background generates 
         artificially coherent PIV vectors (all ~zero = all 'aligned'), 
         which erroneously inflates the simulated spatial_correlation towards 1.
    """
    if video_type == "simulation":
        x1, y1, x2, y2 = SIM_CROP
        frame = frame[y1:y2, x1:x2]
        # White background masking: colored agents on black background
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        agent_mask = (hsv[:,:,1] > SIM_S_THRESH).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        agent_mask = cv2.morphologyEx(agent_mask, cv2.MORPH_DILATE, k)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.where(agent_mask > 0, gray, 0).astype(np.uint8)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if PIV_DOWNSCALE > 1:
        gray = cv2.resize(gray, None,
                          fx=1/PIV_DOWNSCALE, fy=1/PIV_DOWNSCALE,
                          interpolation=cv2.INTER_AREA)
    return gray


def generate_detection_check(video_path: str, video_type: str,
                             output_path: Path, n_frames: int = 4):
    """
    Generates a verification image showing detection results across n frames.
    To be inspected prior to launching the full analysis.
    """
    cap  = cv2.VideoCapture(long_path(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps  = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    indices = np.linspace(total * 0.1, total * 0.9, n_frames, dtype=int)

    fig, axes = plt.subplots(2, n_frames, figsize=(5 * n_frames, 8))
    for i, fi in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        if video_type == "bacteria":
            mask = detect_bacterial_plume(frame)
            disp = frame.copy()
        else:
            mask = detect_agent_group(frame)
            x1, y1, x2, y2 = SIM_CROP
            disp = frame[y1:y2, x1:x2].copy()

        overlay = disp.copy()
        overlay[mask > 0] = [0, 200, 0]
        blended = cv2.addWeighted(disp, 0.6, overlay, 0.4, 0)

        axes[0, i].imshow(cv2.cvtColor(disp,    cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"t={fi/fps:.1f}s", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        pct = mask.mean() / 255 * 100
        axes[1, i].set_title(f"Detection {pct:.1f}%", fontsize=9)
        axes[1, i].axis("off")

    cap.release()
    fig.suptitle(f"Detection check — {Path(video_path).name} [{video_type}]",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(long_path(output_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   → Detection check: {output_path.name}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SECTION 3 — PIV                                                        │
# └─────────────────────────────────────────────────────────────────────────┘

def _piv_fft(g1: np.ndarray, g2: np.ndarray) -> tuple:
    """PIV via FFT cross-correlation (fallback if OpenPIV is missing)."""
    h, w = g1.shape; half = PIV_WINDOW // 2
    step = PIV_WINDOW - PIV_OVERLAP
    xl, yl, ul, vl = [], [], [], []
    for yi in range(half, h - half, step):
        for xi in range(half, w - half, step):
            w1 = g1[yi-half:yi+half, xi-half:xi+half].astype(float)
            w2 = g2[yi-half:yi+half, xi-half:xi+half].astype(float)
            if w1.std() < 2.0 or w2.std() < 2.0:
                continue
            w1 = (w1 - w1.mean()) / (w1.std() + 1e-8)
            w2 = (w2 - w2.mean()) / (w2.std() + 1e-8)
            corr = np.fft.fftshift(
                np.fft.ifft2(np.fft.fft2(w1) * np.conj(np.fft.fft2(w2))).real)
            pk = np.unravel_index(corr.argmax(), corr.shape)
            dy, dx = pk[0] - half, pk[1] - half
            if abs(dx) < half and abs(dy) < half:
                xl.append(xi); yl.append(yi)
                ul.append(float(dx)); vl.append(float(dy))
    x = np.array(xl); y = np.array(yl)
    u = np.array(ul); v = np.array(vl)
    n = int(np.sqrt(len(u)))
    u2d = u[:n*n].reshape(n,n) if n > 1 else u.reshape(1,-1)
    v2d = v[:n*n].reshape(n,n) if n > 1 else v.reshape(1,-1)
    return x, y, u, v, u2d, v2d


def compute_piv(g1: np.ndarray, g2: np.ndarray) -> tuple:
    """
    Computes the PIV velocity field between two grayscale frames.
    Uses OpenPIV if available, otherwise falls back to a custom FFT cross-correlation.
    Returns (x, y, u, v, u2d, v2d) — 1D vectors without NaNs + 2D grids.

    Filtering non-textured windows (PIV_MIN_STD)
    --------------------------------------------------
    In NetLogo simulation videos, a large portion of the image consists of a
    uniform white background (without agents). An interrogation window containing
    only this background lacks exploitable texture: cross-correlation produces an
    arbitrary peak (often a zero displacement or edge artifact). This "false" vector
    subsequently distorts the spatial_correlation computation by artificially driving
    it toward 1 (since numerous identical/null vectors imply "perfect alignment").
    Therefore, a local variance mask is calculated here BEFORE calling OpenPIV,
    invalidating (NaN) any grid cell whose corresponding window exhibits a standard
    deviation below PIV_MIN_STD in either g1 OR g2.
    """
    step = PIV_WINDOW - PIV_OVERLAP
    half = PIV_WINDOW // 2

    if OPENPIV_AVAILABLE:
        u2d, v2d, sig2noise = piv_proc.extended_search_area_piv(
            g1.astype(np.int32), g2.astype(np.int32),
            window_size=PIV_WINDOW, overlap=PIV_OVERLAP,
            search_area_size=PIV_WINDOW, sig2noise_method="peak2peak")
        x2d, y2d = piv_proc.get_coordinates(
            image_size=g1.shape,
            search_area_size=PIV_WINDOW, overlap=PIV_OVERLAP)

        # ── Texture mask: compute local std of g1 and g2 on the same grid
        # as x2d/y2d, and invalidate excessively uniform windows
        rows, cols = u2d.shape
        no_texture = np.zeros((rows, cols), dtype=bool)
        g1f = g1.astype(np.float32); g2f = g2.astype(np.float32)
        for r in range(rows):
            yi = int(y2d[r, 0])
            y0, y1 = max(0, yi - half), min(g1.shape[0], yi + half)
            for c in range(cols):
                xi = int(x2d[r, c])
                x0, x1 = max(0, xi - half), min(g1.shape[1], xi + half)
                std1 = g1f[y0:y1, x0:x1].std()
                std2 = g2f[y0:y1, x0:x1].std()
                if std1 < PIV_MIN_STD or std2 < PIV_MIN_STD:
                    no_texture[r, c] = True

        u2d[no_texture] = np.nan
        v2d[no_texture] = np.nan

        # Adaptive thresholding on sig2noise (20th percentile) — applied
        # in addition to the texture filter, only on remaining vectors
        valid_s2n = sig2noise[~no_texture]
        valid_s2n = valid_s2n[np.isfinite(valid_s2n)]
        s2n_thr  = max(1.0, float(np.percentile(valid_s2n, 20))) \
                   if len(valid_s2n) > 0 else 1.0
        u2d[sig2noise < s2n_thr] = np.nan
        v2d[sig2noise < s2n_thr] = np.nan

        # Replace NaNs with global median (fast, avoids freezing)
        # If more than 70% of the grid lacks texture (virtually empty video),
        # no extrapolation is attempted — NaNs are preserved and handled upstream.
        frac_nan = np.isnan(u2d).mean()
        if frac_nan < 0.7:
            for arr in (u2d, v2d):
                nan_m = ~np.isfinite(arr)
                if nan_m.any():
                    med = float(np.nanmedian(arr))
                    arr[nan_m] = med if np.isfinite(med) else 0.0

        # Remove velocity outliers (|v| > median + 3×MAD)
        spd = np.sqrt(u2d**2 + v2d**2)
        med_s = float(np.nanmedian(spd))
        mad_s = float(np.nanmedian(np.abs(spd - med_s))) if np.isfinite(med_s) else 0.0
        if np.isfinite(med_s):
            out = spd > med_s + 3 * 1.4826 * mad_s
            u2d[out] = 0.0; v2d[out] = 0.0

        xf = x2d.ravel(); yf = y2d.ravel()
        uf = u2d.ravel(); vf = v2d.ravel()
        ok = np.isfinite(uf) & np.isfinite(vf)
        return xf[ok], yf[ok], uf[ok], vf[ok], u2d, v2d
    else:
        return _piv_fft(g1, g2)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SECTION 4 — METRICS                                                    │
# └─────────────────────────────────────────────────────────────────────────┘

def spatial_correlation(u: np.ndarray, v: np.ndarray) -> float:
    """
    Spatial correlation index (Eq. 1 of the paper).

    C(r) = <δu(X)·δu(X+r)> / <|δu(X)|²>

    At r→0, this function assesses whether neighboring velocity vectors
    are mutually aligned:
      C → 1 : all vectors point in the identical direction (coherent)
      C → 0 : random orientations (incoherent)

    Implementation: Mean square of the average unit vector,
    which is equivalent to <cos²θ> where θ is the angle relative
    to the mean direction. C = |<u_hat>|² = <ux_norm>²+<uy_norm>².

    Note: The legacy formula C = <|δu|²>/<|u|²> measured incoherence
    (relative variance), which is the INVERSE of the intended behavior.
    It yielded high values for random motion and low values for coherent
    motion — exactly opposite to the paper's definition.
    """
    u = u[np.isfinite(u)]; v = v[np.isfinite(v)]
    if len(u) < 4: return np.nan
    norms = np.sqrt(u**2 + v**2)
    ok = norms > 1e-6
    if ok.sum() < 4: return np.nan
    # Unit vectors
    un = u[ok] / norms[ok]
    vn = v[ok] / norms[ok]
    # C = |mean unit vector|² ∈ [0,1]
    return float(un.mean()**2 + vn.mean()**2)


def vorticity_divergence(u2d: np.ndarray, v2d: np.ndarray) -> tuple:
    """Vorticity ω and divergence Δ via finite differences on a 2D grid."""
    if u2d.shape[0] < 2 or u2d.shape[1] < 2: return np.nan, np.nan
    du_dy = np.nanmean(np.gradient(u2d, axis=0))
    du_dx = np.nanmean(np.gradient(u2d, axis=1))
    dv_dy = np.nanmean(np.gradient(v2d, axis=0))
    dv_dx = np.nanmean(np.gradient(v2d, axis=1))
    return float(abs(dv_dx - du_dy)), float(abs(du_dx + dv_dy))


def extract_morphometry(mask: np.ndarray) -> dict:
    """Morphometric descriptors of the primary blob."""
    labeled, n = label(mask > 0)
    if n == 0: return None
    props = regionprops(labeled)
    main  = max(props, key=lambda p: p.area)
    L, W  = main.axis_major_length, main.axis_minor_length
    return {"elongation"        : float(L / W) if W > 0 else np.nan,
            "area"              : int(main.area),
            "dispersion_angle" : float(abs(np.degrees(main.orientation)))}


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SECTION 5 — VIDEO-BY-VIDEO ANALYSIS                                     │
# └─────────────────────────────────────────────────────────────────────────┘

PARAMS = ["mean_velocity", "max_velocity", "direction",
          "spatial_correlation", "vorticity", "divergence",
          "elongation", "dispersion_angle", "expansion_rate"]

def analyse_video(video_path: Path, video_type: str,
                  verbose: bool = True) -> dict:
    """
    Complete single-video pipeline: PIV + morphometry → statistical summary.
    """
    cap = cv2.VideoCapture(long_path(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_f = max(1, int(fps / SAMPLE_HZ))

    if verbose:
        piv_lib = "OpenPIV" if OPENPIV_AVAILABLE else "FFT fallback"
        print(f"  {video_path.name} [{video_type}] "
              f"| {fps:.1f}fps {total}frames {total/fps:.1f}s "
              f"| PIV:{piv_lib} ×{PIV_DOWNSCALE} downscale")

    records   = []
    prev_area = None
    t0        = time.time()

    for fi in range(0, total - step_f, step_f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi);          ret1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi + step_f); ret2, f2 = cap.read()
        if not (ret1 and ret2): continue

        g1 = get_gray_for_piv(f1, video_type)
        g2 = get_gray_for_piv(f2, video_type)

        x, y, u, v, u2d, v2d = compute_piv(g1, g2)
        ok = np.isfinite(u) & np.isfinite(v)
        x = x[ok]; y = y[ok]; u = u[ok]; v = v[ok]

        if len(u) < 4: continue

        spd    = np.sqrt(u**2 + v**2) * PIV_DOWNSCALE
        scale  = (PIXEL_TO_UM * fps) if PIXEL_TO_UM else 1.0
        mean_v = float(spd.mean()  * scale)
        max_v  = float(spd.max()   * scale)
        direct = float(np.degrees(np.arctan2(v.mean(), u.mean())) % 360)
        sc     = spatial_correlation(u, v)
        vort, div = vorticity_divergence(u2d, v2d)

        mask  = detect_bacterial_plume(f1) if video_type == "bacteria" \
                else detect_agent_group(f1)
        morph = extract_morphometry(mask)

        area     = float(morph["area"])          if morph else np.nan
        elong    = morph["elongation"]           if morph else np.nan
        disp     = morph["dispersion_angle"]     if morph else np.nan
        exp_rate = float((area - prev_area) / prev_area) \
                   if (morph and prev_area and prev_area > 0) else np.nan
        prev_area = area if morph else prev_area

        records.append({"frame": fi, "time_s": fi/fps,
                        "mean_velocity": mean_v, "max_velocity": max_v,
                        "direction": direct, "spatial_correlation": sc,
                        "vorticity": vort, "divergence": div,
                        "elongation": elong, "dispersion_angle": disp,
                        "expansion_rate": exp_rate})

    cap.release()
    elapsed = time.time() - t0

    summary = {"video": video_path.name, "type": video_type,
               "n_pairs": len(records), "elapsed_s": elapsed}
    for p in PARAMS:
        vals = [r[p] for r in records if np.isfinite(r.get(p, np.nan))]
        summary[f"{p}_mean"] = float(np.mean(vals)) if vals else np.nan
        summary[f"{p}_std"]  = float(np.std(vals))  if vals else np.nan

    if verbose:
        nan_p = [p for p in PARAMS if np.isnan(summary.get(f"{p}_mean", np.nan))]
        ok_p  = len(PARAMS) - len(nan_p)
        warn  = ""
        if elapsed > 600:
            warn = f"   ⚠ SLOW ({elapsed/60:.0f}min) — video might be corrupted"
        print(f"     {len(records)} pairs  |  {ok_p}/{len(PARAMS)} params OK  "
              f"|  {elapsed:.0f}s  "
              f"{'⚠ NaN: '+','.join(nan_p) if nan_p else '✓'}{warn}")

    return summary


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SECTION 6 — STATISTICAL TESTS                                          │
# └─────────────────────────────────────────────────────────────────────────┘

def run_statistics(bac_summaries: list, sim_summaries: list) -> dict:
    """
    Two-sided Wilcoxon-Mann-Whitney and Kolmogorov-Smirnov tests (α=0.05).
    One observation per video (= video-wide mean).
    """
    nb = len(bac_summaries); ns = len(sim_summaries)
    print(f"\n{'='*75}")
    print(f"STATISTICAL TESTS   WMW + KS   n_bac={nb}  n_sim={ns}  α=0.05")
    print(f"{'='*75}")
    hdr = (f"{'Parameter':<22} {'MeanBac':>8} {'MeanSim':>8} "
           f"{'W':>7} {'p(WMW)':>8} {'D':>7} {'p(KS)':>8} "
           f"{'WMW':>13} {'KS':>12}")
    print(hdr); print("─" * len(hdr))

    results = {}
    for p in PARAMS:
        bv = np.array([s[f"{p}_mean"] for s in bac_summaries])
        sv = np.array([s[f"{p}_mean"] for s in sim_summaries])
        bv = bv[np.isfinite(bv)]; sv = sv[np.isfinite(sv)]
        if len(bv) < 2 or len(sv) < 2: continue

        W,  p_wmw = stats.mannwhitneyu(bv, sv, alternative="two-sided")
        D,  p_ks  = stats.ks_2samp(bv, sv, alternative="two-sided")

        results[p] = {"bac_mean": bv.mean(), "bac_std": bv.std(),
                      "sim_mean": sv.mean(), "sim_std": sv.std(),
                      "W": W, "p_wmw": p_wmw, "D": D, "p_ks": p_ks,
                      "wmw_significant": p_wmw < 0.05,
                      "ks_distinct":     p_ks  < 0.05,
                      "bac_vals": bv, "sim_vals": sv}
        wmw_lbl = "Significant" if p_wmw < 0.05 else "Non-sig."
        ks_lbl  = "Distinct"    if p_ks  < 0.05 else "Similar"
        print(f"{p:<22} {bv.mean():>8.3f} {sv.mean():>8.3f} "
              f"{W:>7.1f} {p_wmw:>8.4f} {D:>7.3f} {p_ks:>8.4f} "
              f"{wmw_lbl:>13} {ks_lbl:>12}")
    return results


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SECTION 7 — FIGURES & EXPORT                                           │
# └─────────────────────────────────────────────────────────────────────────┘

def plot_boxplots(stat_results: dict, output_path: Path):
    """In vivo vs MAS boxplots overlaid with p-value annotations."""
    params = list(stat_results.keys())
    ncols  = 3; nrows = (len(params) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()
    C = {"bac": "#4C72B0", "sim": "#DD8452"}

    for i, p in enumerate(params):
        ax = axes[i]; r = stat_results[p]
        bp = ax.boxplot([r["bac_vals"], r["sim_vals"]], patch_artist=True,
                        widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, col in zip(bp["boxes"], [C["bac"], C["sim"]]):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_xticklabels(["In vivo", "MAS"], fontsize=11)
        ax.set_title(p.replace("_"," ").title(), fontsize=11, fontweight="bold")
        unit = " (µm/s)" if (PIXEL_TO_UM and "velocity" in p) else " (px/f)" \
               if "velocity" in p else ""
        ax.set_ylabel("Value" + unit, fontsize=9)
        sw = "✱" if r["p_wmw"] < 0.05 else "ns"
        sk = "✱" if r["p_ks"]  < 0.05 else "ns"
        ax.text(0.97, 0.97,
                f"WMW p={r['p_wmw']:.3f} {sw}\nKS  p={r['p_ks']:.3f} {sk}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    for j in range(len(params), len(axes)): axes[j].set_visible(False)

    fig.suptitle("In Vivo Thiovulum vs MAS — PIV & Morphometry\n"
                 "Wilcoxon-Mann-Whitney & Kolmogorov-Smirnov Tests",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(long_path(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → Boxplots: {output_path.name}")


def export_csv(stat_results: dict, output_path: Path):
    """Exports the comprehensive statistical table to CSV."""
    with open(long_path(output_path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Bac_Mean", "Bac_SD",
                    "Sim_Mean", "Sim_SD",
                    "W_stat", "p_WMW", "WMW_significant",
                    "D_stat", "p_KS",  "KS_distinct"])
        for p, r in stat_results.items():
            w.writerow([p,
                        f"{r['bac_mean']:.4f}", f"{r['bac_std']:.4f}",
                        f"{r['sim_mean']:.4f}", f"{r['sim_std']:.4f}",
                        f"{r['W']:.2f}",        f"{r['p_wmw']:.4f}",
                        r["wmw_significant"],
                        f"{r['D']:.3f}",        f"{r['p_ks']:.4f}",
                        r["ks_distinct"]])
    print(f"   → Stats CSV: {output_path.name}")


def export_per_video_csv(bac_sums: list, sim_sums: list, output_path: Path):
    """Exports per-video averages (Supplementary Material)."""
    with open(long_path(output_path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["video", "type", "n_pairs"] + \
                 [f"{p}_mean" for p in PARAMS] + \
                 [f"{p}_std"  for p in PARAMS]
        w.writerow(header)
        for s in bac_sums + sim_sums:
            row = [s["video"], s["type"], s["n_pairs"]] + \
                  [f"{s.get(f'{p}_mean', np.nan):.4f}" for p in PARAMS] + \
                  [f"{s.get(f'{p}_std',  np.nan):.4f}" for p in PARAMS]
            w.writerow(row)
    print(f"   → Per-video CSV: {output_path.name}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SECTION 8 — MAIN                                                       │
# └─────────────────────────────────────────────────────────────────────────┘

if __name__ == "__main__":

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # On Windows, mkdir() may fail silently or throw an error on excessively
        # long paths. We retry using the \\?\ prefix.
        import os
        os.makedirs(long_path(OUTPUT_DIR), exist_ok=True)
    print(f"\nVideo directory: {VIDEO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # ── Video Discovery ────────────────────────────────────────────
    bac_paths  = []
    simu_paths = []

    for n in BAC_RANGE:
        p = VIDEO_DIR / f"Bac_{n}{VIDEO_EXT}"
        if p.exists(): bac_paths.append(p)

    for n in SIMU_RANGE:
        p = VIDEO_DIR / f"Simu_{n}{VIDEO_EXT}"
        if p.exists(): simu_paths.append(p)

    print(f"Videos found: {len(bac_paths)} bacteria, "
          f"{len(simu_paths)} simulations")

    if len(bac_paths) == 0 and len(simu_paths) == 0:
        print("\n⚠ No videos found. Please check VIDEO_DIR and VIDEO_EXT.")
        sys.exit(1)

    # ── Visual Verification of Detection (1 video each) ────────
    print("\n── Generating detection verification images ──")
    if bac_paths:
        generate_detection_check(
            bac_paths[0], "bacteria",
            OUTPUT_DIR / "check_detection_bac.png")
    if simu_paths:
        generate_detection_check(
            simu_paths[0], "simulation",
            OUTPUT_DIR / "check_detection_simu.png")

    print("\n⚠ IMPORTANT: Open the check_detection_*.png images in the")
    print("    output directory and verify that the highlighted detection")
    print("    (in green) accurately matches the plume / agents.")
    print("    If not, adjust the thresholds in Section 1 before proceeding.\n")
    input("Press Enter to continue with the full analysis...")

    # ── Bacteria Analysis ────────────────────────────────────────────────
    print(f"\n── Analyzing bacteria ({len(bac_paths)} videos) ──")
    bac_summaries = []
    for i, p in enumerate(bac_paths, 1):
        print(f"[{i:2d}/{len(bac_paths)}]", end=" ")
        try:
            bac_summaries.append(analyse_video(p, "bacteria"))
        except Exception as e:
            print(f"    ⚠ Error: {e}")

    # ── Simulations Analysis ──────────────────────────────────────────────
    print(f"\n── Analyzing simulations ({len(simu_paths)} videos) ──")
    sim_summaries = []
    for i, p in enumerate(simu_paths, 1):
        print(f"[{i:3d}/{len(simu_paths)}]", end=" ")
        try:
            sim_summaries.append(analyse_video(p, "simulation"))
        except Exception as e:
            print(f"    ⚠ Error: {e}")

    if len(bac_summaries) < 2 or len(sim_summaries) < 2:
        print("\n⚠ Insufficient data points successfully analyzed to execute statistical tests.")
        sys.exit(1)

    # ── Statistical Evaluation ───────────────────────────────────────────
    stat_results = run_statistics(bac_summaries, sim_summaries)

    # ── Results Export ───────────────────────────────────────────────────
    print("\n── Exporting results ──")
    plot_boxplots(stat_results,    OUTPUT_DIR / "thiovulum_boxplots.png")
    export_csv(stat_results,       OUTPUT_DIR / "thiovulum_statistics.csv")
    export_per_video_csv(bac_summaries, sim_summaries,
                         OUTPUT_DIR / "thiovulum_per_video.csv")

    total_time = sum(s["elapsed_s"] for s in bac_summaries + sim_summaries)
    print(f"\n{'='*60}")
    print(f"✓ Analysis completed successfully in {total_time/60:.1f} minutes.")
    print("=" * 60)