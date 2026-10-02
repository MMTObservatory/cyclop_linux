"""
Configuration, loaded from a TOML file over these defaults. Defaults reproduce the settings of the
Windows installation at the MMTO (from HKCU\\Software\\MiniCyclop).
"""

import copy
import tomllib

DEFAULTS = {
    'site': {
        'name': 'MMTO',
        'latitude': 31.686944,        # deg (31 41 13), as configured in the Windows software
        'longitude': -110.884167,     # deg, east-positive (110 53 03 W)
        'timezone': 'America/Phoenix',
        'max_sun_alt': -5.0,          # deg; measure only when the Sun is below this
    },
    'camera': {
        'simulate': False,
        'address': '192.168.2.59',    # None/"" = first camera found
        'exposure_us': 3906.25,       # Windows "Exposure=-8" is 2**-8 s
        'gain': 12.5,                 # dB; chosen on sky 2026-10-02 (15 dB saturates ~40% of frames)
        'frame_rate': 132.0,          # fps; the 640x480 tracking region's maximum (clipped per region)
        'n_buffers': 128,             # frame buffers queued with Aravis (~0.3 MB each at 640x480)
        'socket_buffer_mb': 8.0,      # UDP receive buffer; raise net.core.rmem_max to allow it
        'roi_width': 640,
        'roi_height': 480,
    },
    'star': {
        'min_snr': 10.0,              # peak / background noise to accept a frame
        'centroid': 'xcorr',          # 'xcorr' (Gaussian cross-correlation) or 'moments' (thresholded)
        'xcorr_sigma': 1.0,           # px, Gaussian correlated with the star; ~FWHM / 2.355
        'xcorr_reach': 10,            # px searched around the last position
        'xcorr_min_snr': 5.0,         # correlation peak / its noise to accept a frame
        'box': 10,                    # moments centroid half-width, px
        'search': 40,                 # search radius around the previous position in the ROI, px
        'max_fwhm': 8.0,
        'recenter_margin': 120,       # move the ROI when the star is closer than this to an edge, px
        'confirm_frames': 3,          # consecutive full-frame detections needed before tracking
    },
    'measurement': {
        'n_samples': 3000,
        'max_zenith_seeing': 7.0,     # arcsec; larger values are discarded
        'detrend': 1,                 # polynomial degree removed from x(t), y(t)
        'clip': 5.0,                  # drop samples this many robust sigmas out; 0 keeps all
        'lost_timeout': 30.0,         # s without a valid centroid before searching full frame again
        'search_interval': 10.0,      # s between full-frame search attempts
        'chunk_frames': 256,          # frames centroided together while the next chunk is acquired
        'chunk_seconds': 2.0,         # hand a chunk over after this long even if not full
    },
    'output': {
        'data_dir': '~/cyclop_data',
        'save_motion': False,
    },
    'web': {
        'enabled': True,
        'host': '0.0.0.0',            # read-only status pages; use 127.0.0.1 to keep them local
        'port': 8080,
        'history_days': 31,           # seeing history loaded from Seeing_Data.txt at startup
        'history_files': [],          # more Seeing_Data.txt files to plot, e.g. the Windows one
    },
    'redis': {
        'enabled': True,
        'host': None,                 # None = $REDISHOST or redis.mmto.arizona.edu
        'port': None,
    },
}


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, 'rb') as f:
            _merge(cfg, tomllib.load(f))
    if cfg['star']['centroid'] not in ('xcorr', 'moments'):
        raise ValueError(f"star.centroid must be 'xcorr' or 'moments', not {cfg['star']['centroid']!r}")
    return cfg
