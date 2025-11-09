#!/usr/bin/env python3
"""
app.py
Unified pipeline:
 - download CAMS (if not present)
 - extract ENS_FORECAST.nc
 - ensure local copy (avoid OneDrive issues)
 - open dataset, compute allergy_index
 - run Flask server with /overlay and /point endpoints
"""

import os
import zipfile
import shutil
import traceback
import io
import numpy as np
from datetime import datetime
from matplotlib import cm, colors
from PIL import Image

# HTTP server
from flask import Flask, send_file, jsonify, abort, send_from_directory

# Data libs
import xarray as xr
import cdsapi

# ----------------- CONFIG -----------------
# cdsapi client uses ~/.cdsapirc by default; override if needed
CDS_DATASET = "cams-europe-air-quality-forecasts"
CDS_REQUEST = {
    "variable": ["alder_pollen","birch_pollen","grass_pollen","mugwort_pollen","olive_pollen","ragweed_pollen","pm2p5"],
    "model": ["ensemble"],
    "level": ["0"],
    "date": "2025-11-05/2025-11-08",
    "type": ["forecast"],
    "time": ["00:00"],
    "leadtime_hour": ["0"],
    "data_format": "netcdf_zip",
    "area": [72, -25, 34, 45]
}

ZIP_FILE = "cams_data.zip"
EXTRACT_DIR = "cams_data"
NC_BASENAME = "ENS_FORECAST.nc"

# Local safe copy (must be outside OneDrive) - change if needed
SAFE_DIR = r"C:\temp\cams_test"
os.makedirs(SAFE_DIR, exist_ok=True)
SAFE_NC_PATH = os.path.join(SAFE_DIR, NC_BASENAME)

# Europe bounding box (used for overlay bounds)
EU_N, EU_W, EU_S, EU_E = 72, -25, 34, 45

# Overlay image size for web
OUT_W, OUT_H = 900, 700

# variables naming in dataset (observed)
POLLEN_VARS = ['apg_conc','bpg_conc','gpg_conc','mpg_conc','opg_conc','rwpg_conc']  # these are actual variable names seen in file
PM_VAR = 'pm2p5'

# Flask static html filename (put interactive html in same dir as app.py as static file)
STATIC_HTML = "interactive_map_with_sidebar.html"  # optional: if present, will be served at /
# ------------------------------------------

app = Flask(__name__, static_folder='.')

def download_if_needed():
    """Download cams_data.zip using cdsapi if missing or empty."""
    if os.path.exists(ZIP_FILE) and os.path.getsize(ZIP_FILE) > 0:
        print("Zip already exists:", os.path.abspath(ZIP_FILE), "size:", os.path.getsize(ZIP_FILE))
        return
    print("Downloading CAMS data to", os.path.abspath(ZIP_FILE))
    client = cdsapi.Client()  # expects ~/.cdsapirc or env vars
    client.retrieve(CDS_DATASET, CDS_REQUEST, ZIP_FILE)
    print("Download finished.")

def inspect_zip():
    """Print information about zip and return list of .nc names inside."""
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError("ZIP not found: " + ZIP_FILE)
    print("\n-- ZIP info --")
    print("Path:", os.path.abspath(ZIP_FILE), "size:", os.path.getsize(ZIP_FILE))
    with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
        namelist = zf.namelist()
        print("Entries in zip:", len(namelist))
        for n in namelist[:50]:
            print(" -", n)
        nc_files = [n for n in namelist if n.endswith('.nc')]
        print("Found .nc files:", len(nc_files))
        for n in nc_files[:10]:
            print("  >", n)
    return nc_files

def extract_nc_to_dir(nc_member=None):
    """Extract the .nc (or first one) into EXTRACT_DIR and return path(s)."""
    with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
        members = [n for n in zf.namelist() if n.endswith('.nc')]
        if not members:
            raise FileNotFoundError("No .nc in zip")
        chosen = nc_member if nc_member is not None else members[0]
        print("Extracting", chosen, "to", EXTRACT_DIR)
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        zf.extract(chosen, EXTRACT_DIR)
        extracted_path = os.path.join(EXTRACT_DIR, os.path.basename(chosen))
        print("Extracted path:", extracted_path)
    return extracted_path

def ensure_local_copy(extracted_nc_path):
    """Ensure SAFE_NC_PATH contains the nc file (copy if needed). Return path to safe file."""
    if os.path.exists(SAFE_NC_PATH):
        # if exists and sizes match, keep it
        try:
            if os.path.getsize(SAFE_NC_PATH) == os.path.getsize(extracted_nc_path):
                print("Safe copy already exists:", SAFE_NC_PATH)
                return SAFE_NC_PATH
        except Exception:
            pass
    print("Copying to safe location:", SAFE_NC_PATH)
    shutil.copy2(extracted_nc_path, SAFE_NC_PATH)
    print("Copy done, size:", os.path.getsize(SAFE_NC_PATH))
    return SAFE_NC_PATH

# --- Globals filled at init ---
DS = None
ALLERGY_INDEX = None
LAT_NAME = None
LON_NAME = None
ALLERGY_EU = None
FULL_LATS = None
FULL_LONS = None
SEL = {}

def init_dataset(nc_path):
    """Open dataset and compute allergy_index and derived arrays."""
    global DS, ALLERGY_INDEX, LAT_NAME, LON_NAME, ALLERGY_EU, FULL_LATS, FULL_LONS, SEL
    print("Opening dataset:", nc_path)
    DS = xr.open_dataset(nc_path, engine='netcdf4')
    print("Variables:", list(DS.variables)[:40])

    # coords names
    LAT_NAME = next((n for n in ('latitude','lat','y') if n in DS.coords or n in DS.variables), None)
    LON_NAME = next((n for n in ('longitude','lon','x') if n in DS.coords or n in DS.variables), None)
    if LAT_NAME is None or LON_NAME is None:
        raise RuntimeError("Can't find lat/lon coords in dataset")

    # selector: time=0, level=0 if present
    SEL = {}
    if 'time' in DS.dims or 'time' in DS.coords:
        SEL['time'] = 0
    if 'level' in DS.dims or 'level' in DS.coords:
        SEL['level'] = 0
    print("Using selector:", SEL)

    # compute total pollen (use available pollen variables from DS, mapped to actual names if needed)
    total = None
    template = None
    for v in POLLEN_VARS:
        if v in DS.variables:
            arr = DS[v].isel(**SEL)
            template = arr if template is None else template
            total = arr if total is None else (total + arr)
    if total is None:
        # fallback: create zero array using any data var as template
        if template is None:
            anyvar = list(DS.data_vars)[0]
            template = DS[anyvar].isel(**SEL)
        total = xr.zeros_like(template)

    # pm2p5 normalized
    if PM_VAR in DS.variables:
        pm = DS[PM_VAR].isel(**SEL)
        pmmax = float(pm.max()) if float(pm.max()) > 0 else 0.0
        pm_norm = pm / pmmax if pmmax > 0 else xr.zeros_like(pm)
    else:
        pm_norm = xr.zeros_like(total)

    # allergy index
    ALLERGY_INDEX = total + pm_norm * 10.0
    print("Computed allergy_index.")

    # europe slice
    try:
        ALLERGY_EU = ALLERGY_INDEX.sel({LAT_NAME: slice(EU_S, EU_N), LON_NAME: slice(EU_W, EU_E)})
    except Exception:
        ALLERGY_EU = ALLERGY_INDEX

    # full coords arrays for nearest-index computations
    FULL_LATS = DS[LAT_NAME].values
    FULL_LONS = DS[LON_NAME].values
    if FULL_LONS.max() > 180:
        FULL_LONS = np.where(FULL_LONS > 180, FULL_LONS - 360, FULL_LONS)

    # make them available in globals
    globals().update({
        'LAT_NAME': LAT_NAME, 'LON_NAME': LON_NAME,
        'ALLERGY_INDEX': ALLERGY_INDEX, 'ALLERGY_EU': ALLERGY_EU,
        'FULL_LATS': FULL_LATS, 'FULL_LONS': FULL_LONS, 'SEL': SEL
    })

    print("Dataset init complete. EU subset shape:", getattr(ALLERGY_EU, 'shape', None))

def nearest_index(array, value):
    return int(np.abs(array - value).argmin())

def make_overlay_png_bytes(data_np, lats, lons, out_w=OUT_W, out_h=OUT_H):
    """Create RGBA PNG bytes from 2D numpy array data_np (lat, lon)."""
    # Ensure orientation (lat, lon)
    if data_np.shape[0] == lons.size and data_np.shape[1] == lats.size:
        data_np = data_np.T

    mask_nan = np.isnan(data_np)
    valid = data_np[~mask_nan]
    if valid.size == 0:
        rgba_img = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        pil = Image.fromarray(rgba_img, mode='RGBA')
        buf = io.BytesIO(); pil.save(buf, format='PNG'); buf.seek(0); return buf

    p_low = np.percentile(valid, 2)
    vmin = float(np.nanmin(valid)); vmax = float(np.nanmax(valid))
    threshold = max(p_low, 1e-6)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('OrRd')
    rgba_float = cmap(norm(np.nan_to_num(data_np, nan=vmin)))
    rgb_uint8 = (rgba_float[..., :3] * 255).astype(np.uint8)

    alpha = np.zeros_like(data_np, dtype=np.uint8)
    valid_mask = (~mask_nan) & (data_np >= threshold)
    alpha_vals = norm(data_np[valid_mask])
    if alpha_vals.size > 0:
        alpha[valid_mask] = np.clip((alpha_vals ** 0.8 * 255).astype(np.uint8), 30, 255)

    h, w = rgb_uint8.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb_uint8
    rgba[..., 3] = alpha

    pil = Image.fromarray(rgba, mode='RGBA')
    pil = pil.resize((out_w, out_h), resample=Image.BILINEAR)
    buf = io.BytesIO(); pil.save(buf, format='PNG'); buf.seek(0)
    return buf

# ----------------- Flask endpoints -----------------
@app.route('/')
def index():
    # serve static HTML if present in working dir
    if os.path.exists(STATIC_HTML):
        return send_from_directory('.', STATIC_HTML)
    return "<h3>Allergy server running. Put interactive_map_with_sidebar.html in same folder to view UI.</h3>"

@app.route('/overlay')
def overlay():
    """Return overlay PNG for Europe."""
    try:
        data_np = ALLERGY_EU.values
        lats_sub = ALLERGY_EU[LAT_NAME].values
        lons_sub = ALLERGY_EU[LON_NAME].values
        buf = make_overlay_png_bytes(data_np, lats_sub, lons_sub)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        print("Overlay error:", e)
        traceback.print_exc()
        abort(500, str(e))

@app.route('/point')
def point():
    """Return metrics near lat/lon: /point?lat=45.66&lon=12.24"""
    lat = None; lon = None
    try:
        from flask import request
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
    except Exception:
        pass
    if lat is None or lon is None:
        return jsonify({"error":"Provide lat and lon parameters"}), 400

    # normalize lon
    lon_norm = lon
    if FULL_LONS.max() > 180 and lon_norm > 180:
        lon_norm = lon_norm - 360

    ilat = nearest_index(FULL_LATS, lat)
    ilon = nearest_index(FULL_LONS, lon_norm)

    # compute pollen sum at indices
    pollen_sum = 0.0
    for v in POLLEN_VARS:
        if v in DS.variables:
            arr = DS[v].isel(**SEL)
            try:
                pollen_sum += float(arr.values[ilat, ilon])
            except Exception:
                try:
                    pollen_sum += float(arr.values[ilon, ilat])
                except Exception:
                    pass
    # pm2.5
    pmval = 0.0
    if PM_VAR in DS.variables:
        arrp = DS[PM_VAR].isel(**SEL)
        try:
            pmval = float(arrp.values[ilat, ilon])
        except Exception:
            try:
                pmval = float(arrp.values[ilon, ilat])
            except Exception:
                pmval = 0.0

    # allergy value at that grid
    try:
        val = float(ALLERGY_INDEX.isel({LAT_NAME: ilat, LON_NAME: ilon}).values)
    except Exception:
        # fallback
        val = pollen_sum + (pmval / (pmval+1e-6)) * 10.0 if pmval>0 else pollen_sum

    # compute eu bounds for normalization
    eu_vals = np.nan_to_num(ALLERGY_EU.values, nan=np.nanmin(ALLERGY_EU.values))
    eu_vmin = float(np.nanmin(eu_vals))
    eu_vmax = float(np.nanmax(eu_vals)) if float(np.nanmax(eu_vals))>eu_vmin else eu_vmin+1.0
    norm01 = (val - eu_vmin) / (eu_vmax - eu_vmin)
    norm01 = float(np.clip(norm01, 0.0, 1.0))
    overall_score = int(np.round(norm01 * 10))
    overall_score = int(np.clip(overall_score, 0, 10))

    # simple categories
    pollen_level = 0
    if pollen_sum > 50: pollen_level = 4
    elif pollen_sum > 20: pollen_level = 3
    elif pollen_sum > 5: pollen_level = 2
    elif pollen_sum > 1: pollen_level = 1
    else: pollen_level = 0

    aqi = int(np.clip(pmval, 0, 300))
    if aqi <= 50: aqi_label = "Good"
    elif aqi <= 100: aqi_label = "Moderate"
    elif aqi <= 150: aqi_label = "Unhealthy for SG"
    else: aqi_label = "Unhealthy"

    # dominant allergens top-3 by value
    dom = []
    for v in POLLEN_VARS:
        if v in DS.variables:
            try:
                vv = DS[v].isel(**SEL).values
                valv = float(vv[ilat, ilon]) if vv.shape[0] > ilat and vv.shape[1] > ilon else float(np.nan)
            except Exception:
                try:
                    valv = float(vv[ilon, ilat])
                except Exception:
                    valv = 0.0
            dom.append((v, valv))
    dom = sorted(dom, key=lambda x: x[1], reverse=True)[:3]
    dom_names = [d[0] for d in dom]

    resp = {
        "lat": float(lat), "lon": float(lon),
        "nearest_grid": {"ilat": ilat, "ilon": ilon},
        "pollen_sum": pollen_sum,
        "pm2p5": pmval,
        "allergy_value": val,
        "overall_score": overall_score,
        "pollen_level": pollen_level,
        "aqi": aqi,
        "aqi_label": aqi_label,
        "dominant_allergens": dom_names,
        "last_update": datetime.utcnow().isoformat()+"Z"
    }
    return jsonify(resp)

# ------------- Main init & run -------------
if __name__ == "__main__":
    # Step 1: download
    try:
        download_if_needed()
    except Exception as e:
        print("Download step failed (continuing if zip exists):", e)
        traceback.print_exc()

    # Step 2: inspect and extract
    try:
        ncs = inspect_zip()
        extracted = extract_nc_to_dir(nc_member=(ncs[0] if ncs else None))
    except Exception as e:
        print("Error during extract:", e)
        traceback.print_exc()
        # if extraction failed but an NC might already be present in extract_dir, try to find it
        extracted = None
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for f in files:
                if f.endswith('.nc'):
                    extracted = os.path.join(root, f); break
            if extracted: break
        if not extracted:
            print("No .nc found; aborting.")
            raise SystemExit(1)

    # Step 3: ensure safe copy (to SAFE_NC_PATH)
    try:
        safe_nc = ensure_local_copy(extracted)
    except Exception as e:
        print("Copy to safe location failed:", e); traceback.print_exc(); raise

    # Step 4: init dataset
    try:
        init_dataset(safe_nc)
    except Exception as e:
        print("Dataset init failed:", e); traceback.print_exc(); raise

    # Run Flask app (serve on 0.0.0.0:5000)
    print("Starting Flask server at http://0.0.0.0:5000 (interactive UI available at / if you placed html file)")
    app.run(host="0.0.0.0", port=5000, debug=True)
