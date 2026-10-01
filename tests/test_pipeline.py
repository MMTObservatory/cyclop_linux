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
