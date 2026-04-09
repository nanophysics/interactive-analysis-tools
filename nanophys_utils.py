"""
Shared utilities for nanophysics measurement notebooks.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = "drive_index_cache_files"
DATA_ROOT = "Z:/POBox/Jonas Gerber/05 - Measurements (Data)"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# ---------------------------------------------------------------------------
# File discovery (with JSON disk cache)
# ---------------------------------------------------------------------------

def get_all_files(root_path, force_update=False):
    """
    Return a list of all file paths under *root_path*.
    Results are cached in a per-path JSON file so repeated calls are instant.
    Pass force_update=True to rescan the drive.
    """
    path_hash = hashlib.md5(str(root_path).encode("utf-8")).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"cache_{path_hash}.json")

    if os.path.exists(cache_file) and not force_update:
        print(f"Loading file list from local cache: {cache_file}...")
        t0 = time.time()
        try:
            with open(cache_file, "r") as f:
                file_list = json.load(f)
            print(f"Loaded {len(file_list)} files in {time.time()-t0:.2f}s")
            return file_list
        except Exception as e:
            print(f"Cache corrupted ({e}), rescanning...")

    print(f"Scanning drive (this may take a while)... {root_path}")
    t0 = time.time()
    root = Path(root_path)
    if not root.exists():
        print(f"Warning: Path not found: {root_path}")
        return []

    files = [p.as_posix() for p in root.rglob("*") if p.is_file()]
    print(f"Scan complete. Found {len(files)} files in {time.time()-t0:.2f}s")

    print(f"Saving cache to {cache_file}...")
    with open(cache_file, "w") as f:
        json.dump(files, f)

    return files


def find_file_cached(root_path, substring, force_refresh=False):
    """Return all cached file paths whose filename contains *substring*."""
    all_files = get_all_files(root_path, force_update=force_refresh)
    return [f for f in all_files if substring in f.split("/")[-1]]


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------

# Channel name patterns per measurement mode.
# Channels listed in _OPTIONAL return -1 if absent instead of raising.
_PATTERNS = {
    "gfactor": {
        "V_PG":  r"PG.?Virt(?:ual)?|V.?[PC]G",
        "B":     r"Magnet|\bB\b",
        "I_ACx": r"I.?(?:SD)?.*AC.*x",
        "I_SD":  r"I.?SD(?!.*AC)",
    },
    "diamond": {
        "V_SD":  r"V.?SD",
        "V_PG":  r"PG.?Virtual|V.?PG|V.?CG",
        "I_ACx": r"I.?SD.*AC.*x",
        "I_SD":  r"I.?SD.*?(C18|C17)",
        "B":     r".*Magnet.*|\bB\b",
    },
}

# Channels that are allowed to be absent (return index -1 instead of raising).
_OPTIONAL = {"I_ACx", "B"}


def _resolve_filepath(short, folder, name, path, exclude):
    """Return the HDF5 filepath and a display name."""
    if short is not None and folder is not None:
        search_root = f"{DATA_ROOT}/{folder}/"
        filepaths = find_file_cached(search_root, short)
        if not filepaths:
            raise FileNotFoundError(f"No files found for short='{short}' in folder='{folder}'")
        print(f"Found files: {[os.path.basename(f) for f in filepaths]}")
        if exclude is not None:
            filepaths = [f for f in filepaths if exclude not in os.path.basename(f)]
            if not filepaths:
                raise FileNotFoundError(
                    f"No files found for short='{short}' in folder='{folder}' "
                    f"after excluding '{exclude}'"
                )
            print(f"After excluding '{exclude}': {[os.path.basename(f) for f in filepaths]}")
        filepath = next((f for f in filepaths if f.endswith(".hdf5")), None)
        if filepath is None:
            raise FileNotFoundError(f"No .hdf5 file found for short='{short}' in folder='{folder}'")
        print(f"Chose file: {filepath}")
        display_name = short
    elif name is not None and path is not None:
        if not path.startswith("Z:/"):
            path = DATA_ROOT + path
        filepath = os.path.join(path, name + ".hdf5")
        display_name = name
    else:
        raise ValueError("Provide either (short, folder) or (name, path).")
    return filepath, display_name


def _find_channel(pattern, channel_names, optional=False):
    """Return the index of the first channel matching *pattern*, or -1 if optional and absent."""
    for i, ch in enumerate(channel_names):
        if re.search(pattern, ch, re.IGNORECASE):
            return i
    if optional:
        return -1
    raise ValueError(f"Could not find channel matching pattern: '{pattern}'")


def load_data(
    short=None,
    folder=None,
    name=None,
    path=None,
    mode="gfactor",
    manual_indices=None,
    sort=False,
    exclude=None,
):
    """
    Load an HDF5 measurement file and return aligned (x, y, z) arrays.

    Parameters
    ----------
    short, folder : str
        Short filename substring and experiment folder (alternative to name/path).
    name, path : str
        Full filename (without .hdf5) and directory path (alternative to short/folder).
    mode : {'gfactor', 'diamond'}
        'gfactor' — X = V_PG, Y = B (magnetic field).
        'diamond' — X = V_PG, Y = V_SD (bias voltage).
    manual_indices : list of int, optional
        Override auto-detected channel indices (same order as the pattern dict for *mode*).
    sort : bool
        Ensure X and Y axes are strictly increasing. Default False.
    exclude : str, optional
        Skip files whose name contains this substring (used with short/folder lookup).

    Returns
    -------
    display_name : str
    x_centers : 1-D ndarray   (V_PG values)
    y_centers : 1-D ndarray   (B or V_SD values)
    z_final   : 2-D ndarray   shape (len(y_centers), len(x_centers))
    """
    if mode not in _PATTERNS:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'gfactor' or 'diamond'.")

    filepath, display_name = _resolve_filepath(short, folder, name, path, exclude)

    with h5py.File(filepath, "r") as data:
        raw_names = data["Data"]["Channel names"][:]
        channel_names = [x[0].decode("utf-8").strip() for x in raw_names]
        print("Found channels:", channel_names)

        patterns = _PATTERNS[mode]
        indices = {
            key: _find_channel(pat, channel_names, optional=(key in _OPTIONAL))
            for key, pat in patterns.items()
        }
        print("Mapped indices:", indices)

        if manual_indices is not None:
            if len(manual_indices) == len(patterns):
                indices = dict(zip(patterns.keys(), manual_indices))
                print("Overwrote indices:", indices)
            else:
                print(
                    f"Could not overwrite: manual_indices has length {len(manual_indices)}, "
                    f"needs {len(patterns)}"
                )

        dataset = data["Data"]["Data"]

        V_PG = dataset[:, indices["V_PG"], :]

        # Compute Z (differential conductance)
        if indices["I_ACx"] == -1:
            I_SD = dataset[:, indices["I_SD"], :]
            dx = V_PG[1, 0] - V_PG[0, 0]
            if dx == 0:
                dx = V_PG[0, 1] - V_PG[0, 0]
            print(f"dx = {dx}")
            I_ACx = np.gradient(I_SD, dx, axis=1) * 1e-8
            print("Manually computed I_ACx via np.gradient(I_SD).")
        else:
            I_ACx = dataset[:, indices["I_ACx"], :] * 1e-8

        z_all = 1e12 * I_ACx  # pA

        # Determine Y axis
        if mode == "gfactor":
            if indices["B"] == -1:
                raise ValueError("B-field channel not found — required for mode='gfactor'.")
            Y_raw = dataset[:, indices["B"], :]
        else:  # diamond
            Y_raw = dataset[:, indices["V_SD"], :] * 2  # factor-of-2 hardware correction

        # Orientation detection: is V_PG the fast (column) axis?
        is_pg_horizontal = abs(V_PG[0, 1] - V_PG[0, 0]) > 1e-9

        if is_pg_horizontal:
            print("Detected orientation: V_PG varies along columns.")
            x_centers = V_PG[0, :]
            y_centers = Y_raw[:, 0]
            z_final = z_all
        else:
            print("Detected orientation: V_PG varies along rows — transposing.")
            x_centers = V_PG[:, 0]
            y_centers = Y_raw[0, :]
            z_final = z_all.T

    # Sorting
    if sort:
        if np.any(np.diff(x_centers) <= 0):
            order = np.argsort(x_centers)
            x_centers = x_centers[order]
            z_final = z_final[:, order]
        if np.any(np.diff(y_centers) <= 0):
            order = np.argsort(y_centers)
            y_centers = y_centers[order]
            z_final = z_final[order, :]

    # NaN / Inf guard
    if not np.isfinite(z_final).all():
        n_nan = np.isnan(z_final).sum()
        n_inf = np.isinf(z_final).sum()
        print(f"Warning: {n_nan} NaNs and {n_inf} Infs in Z — replacing with zeros.")
        z_final = np.nan_to_num(
            z_final,
            nan=0.0,
            posinf=np.nanmax(z_final[np.isfinite(z_final)]),
            neginf=0.0,
        )

    return display_name, x_centers, y_centers, z_final


def lorentzian(x, A, x0, gamma, C):
    """
    Standard Lorentzian lineshape.
    A: Amplitude (Height above background)
    x0: Center position
    gamma: Half-width at half-maximum (HWHM)
    C: Constant background offset
    """
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + C


from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import Affine2D

class HandlerVerticalTuple(HandlerTuple):
    def __init__(self, vertical_pad=0.5, y_offset=-2.0, **kwargs):
        self.vertical_pad = vertical_pad
        self.y_offset = y_offset # Manual adjustment for vertical centering
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        n_elements = len(orig_handle)
        artists = []
        
        for i, handle in enumerate(orig_handle):
            # Get the handler for the individual line/item
            handler = legend.get_legend_handler(legend.get_legend_handler_map(), handle)
            a_list = handler.create_artists(legend, handle, xdescent, ydescent, 
                                            width, height, fontsize, trans)
            
            # 1. Calculate the spread (the gap between the two lines)
            spread_offset = ((n_elements - 1) / 2.0 - i) * (fontsize * self.vertical_pad)
            
            # 2. Combine spread with the global y_offset to center against text
            # We add ydescent to stay relative to the handlebox's actual vertical center
            total_v_shift = spread_offset + self.y_offset
            
            for a in a_list:
                # Apply the combined translation
                a.set_transform(Affine2D().translate(0, total_v_shift) + trans)
                artists.append(a)
                
        return artists