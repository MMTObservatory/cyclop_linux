import copy

import numpy as np
import pytest

from cyclop import config, star
from cyclop.camera import SimCamera
from cyclop.monitor import MEASURING, Monitor


def test_centroid_accuracy():
    cam = SimCamera(jitter_px=0.0, drift_px_per_s=0.0, seed=2)
    cam.set_region(1796 - 320, 679 - 240, 640, 480)
    errs = []
    for _ in range(50):
        _, img = cam.grab()
        s = star.measure(img)
        x0, y0, _, _ = cam.region
        tx, ty = cam.true_position()
        errs.append((s.x + x0 - tx, s.y + y0 - ty))
        assert s.fwhm == pytest.approx(2.2, abs=0.4)
    errs = np.array(errs)
    assert np.abs(errs.mean(0)).max() < 0.05
    assert errs.std(0).max() < 0.05


def test_no_star_rejected():
    cam = SimCamera(seed=3)
    cam.visible = False
    _, img = cam.grab()
    assert star.measure(img) is None


def test_monitor_recovers_jitter(tmp_path):
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg['measurement']['n_samples'] = 1500
    cams = []

    def factory():
        cams.append(SimCamera(jitter_px=0.15, drift_px_per_s=0.5, seed=4))
        return cams[-1]

    mon = Monitor(cfg, factory, sleep=lambda s: None)
    mon.run(ignore_sun=True, max_results=2)
    assert len(mon.results) == 2
    for r in mon.results:
        # centroid noise adds a little to the injected jitter
        assert r['sigma_x'] == pytest.approx(0.15, rel=0.1)
        assert r['sigma_y'] == pytest.approx(0.15, rel=0.1)
        assert r['rate'] == pytest.approx(60.0, rel=0.01)


def test_monitor_handles_lost_star():
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg['measurement']['n_samples'] = 100
    t = {'now': 0.0}
    cam = SimCamera(seed=5)

    mon = Monitor(cfg, lambda: cam, sleep=lambda s: t.__setitem__('now', t['now'] + s),
                  clock=lambda: t['now'])
    mon.step(ignore_sun=True)
    assert mon.state == MEASURING
    cam.visible = False
    for _ in range(10):
        t['now'] += 5
        mon.step(ignore_sun=True)
    assert mon.state != MEASURING
    assert mon.samples['t'] == []
    cam.visible = True
    # full-frame searches are retried every search_interval (10 s); each idle step sleeps 0.5 s
    for _ in range(40):
        mon.step(ignore_sun=True)
    assert mon.state == MEASURING


def test_monitor_counts_dropped_frames():
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg['measurement']['n_samples'] = 1000
    cam = SimCamera(drop_fraction=0.5, seed=6)
    mon = Monitor(cfg, lambda: cam, sleep=lambda s: None)
    mon.run(ignore_sun=True, max_results=1)
    r = mon.results[0]
    assert r['drop_fraction'] == pytest.approx(0.5, abs=0.04)
    assert r['camera_fps'] == pytest.approx(60.0, rel=0.02)    # the camera's rate, not ours
    assert r['rate'] == pytest.approx(30.0, rel=0.1)
    assert r['proc_ms'] > 0
    assert mon.cam_stats[-1][1]['dropped'] > 0


def test_frame_id_gaps():
    from cyclop.camera import AravisCamera
    cam = object.__new__(AravisCamera)          # no hardware: exercise only the counting
    cam.delivered = cam.dropped = 0
    cam._last_id = None
    for fid in (65530, 65531, 65534, 65535, 1, 3):   # 16-bit ids wrap 65535 -> 1, skipping 0
        cam._count(fid)
    assert (cam.delivered, cam.dropped) == (6, 3)
    cam._count(2)                                   # out of order / restarted: not counted
    assert cam.dropped == 3


def test_measure_cube_matches_measure():
    cam = SimCamera(seed=8)
    cam.set_region(1480, 440, 640, 480)
    r = 40
    cutouts = []
    for i in range(64):
        _, img = cam.grab()
        x, y = (cam.true_position() - (1480, 440)).astype(int)
        cutouts.append(img[y - r:y + r + 1, x - r + i % 9 - 4:x + r + 1 + i % 9 - 4])
    cube = np.stack(cutouts)
    # blank frames (clouds) must come back as not ok, like measure returning None
    cube[::10] = np.random.default_rng(0).normal(12, 2, cube[::10].shape).clip(0, 255)
    res = star.measure_cube(cube)
    assert res['background'][10] == pytest.approx(12, abs=1)
    for i, c in enumerate(cube):
        s = star.measure(c)
        assert res['ok'][i] == (s is not None)
        if s is not None:
            for k in ('x', 'y', 'flux', 'peak', 'fwhm', 'snr', 'n_saturated'):
                assert res[k][i] == pytest.approx(getattr(s, k), rel=1e-5, abs=1e-6), k


def test_acquirer_chunks():
    from cyclop.camera import Acquirer
    cam = SimCamera(seed=9)
    cam.set_region(1480, 440, 640, 480)
    acq = Acquirer(cam, center=(1796, 679), radius=40, chunk_frames=50, n_chunks=2).start()
    try:
        first = acq.get(timeout=10)
        assert first.n == 50 and first.cube.shape == (50, 81, 81)
        assert (np.diff(first.t) > 0).all()
        assert (first.x0 == 1796 - 40).all() and (first.y0 == 679 - 40).all()
        acq.center = (2500, 0)                     # outside the region: cutouts stay inside it
        acq.release(first)
        for _ in range(3):                         # chunks filled before the move may still come
            chunk = acq.get(timeout=10)
            moved = chunk.x0[-1] == 1480 + 640 - 81
            acq.release(chunk)
            if moved:
                break
        assert moved and chunk.y0[-1] == 440
    finally:
        acq.stop()
    assert not acq._thread.is_alive()


def _sim_cutouts(seed, n=64, r=40, offset=(0.0, 0.0)):
    sx, sy = 1796.0 + offset[0], 679.0 + offset[1]
    cam = SimCamera(star_xy=(sx, sy), jitter_px=0.0, drift_px_per_s=0.0, seed=seed)
    cam.set_region(1480, 440, 640, 480)
    x, y = 1796 - 1480, 679 - 440
    cube = np.stack([cam.grab()[1][y - r:y + r + 1, x - r:x + r + 1] for _ in range(n)])
    return cube, np.tile((r + offset[0], r + offset[1]), (n, 1))


def test_correlate_cube_accuracy():
    errs = []
    for i, off in enumerate([(fx, fy) for fx in (-0.4, -0.2, 0.0, 0.2, 0.4) for fy in (-0.3, 0.1, 0.45)]):
        cube, truth = _sim_cutouts(seed=10 + i, n=20, offset=off)
        c = star.correlate_cube(cube, 40, 40)
        assert (c['snr'] > 20).all()
        errs.append(np.stack([c['x'], c['y']], axis=1) - truth)
    errs = np.concatenate(errs)
    assert np.abs(errs.mean(0)).max() < 0.03        # no bias, nor pixel-phase dependent error
    assert np.abs(errs).max() < 0.2
    assert errs.std(0).max() < 0.05


def test_correlate_cube_stays_near_guess():
    cube, truth = _sim_cutouts(seed=11, n=8)
    cube = cube.copy()
    cube[:, 2:5, 2:5] = 255                          # a brighter blob far from the star
    c = star.correlate_cube(cube, 40, 40, reach=10)
    assert np.abs(c['x'] - truth[:, 0]).max() < 0.1
    assert star.measure_cube(cube)['x'].max() < 10   # whereas the full-cutout search takes the blob


def test_correlate_cube_rejects_blank_frames():
    blank = np.random.default_rng(0).normal(12, 3, (16, 81, 81)).clip(0, 255).astype(np.uint8)
    assert (star.correlate_cube(blank, 40, 40)['snr'] < 5).all()


def test_inliers_clip_outliers():
    from cyclop import seeing
    rng = np.random.default_rng(1)
    t = np.arange(3000) / 132.0
    x, y = rng.normal(0, 0.3, 3000) + 0.1 * t, rng.normal(0, 0.3, 3000)
    x[100], y[2000] = 25.0, -20.0
    keep = seeing.inliers(t, x, y)
    assert not keep[100] and not keep[2000] and keep.sum() >= 2990
    assert seeing.inliers(t, x, y, clip=None).all()


def test_column_flux_factor():
    x = np.linspace(1000, 1012, 49)
    assert np.all(star.column_flux_factor(x, []) == 1.0)
    terms = config.DEFAULTS['star']['column_flux_terms']
    f = star.column_flux_factor(x, terms)
    assert np.allclose(f, star.column_flux_factor(x + 4, terms))       # repeats every 4 columns
    assert abs(f.mean() - 1) < 1e-3 and 0.05 < f.max() - 1 < 0.15
    # one harmonic: 1 + a cos(pi x / 2) + b sin(pi x / 2)
    assert star.column_flux_factor(1.0, [0.1, 0.2]) == pytest.approx(1.2)
