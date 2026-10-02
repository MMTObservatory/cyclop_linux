"""
Command line entry point.

    cyclop run [-c config.toml] [--simulate] [--ignore-sun] [--no-redis] [--no-web] [-n N]
    cyclop replay MOTION_FILE [...]      reduce Windows *_Motion.txt files with this code
    cyclop camera-info [--address IP]    list camera features through Aravis
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

from cyclop import config, seeing


def _setup_logging(level, logfile=None):
    from cyclop.web import LogBuffer
    handlers = [logging.StreamHandler(), LogBuffer()]
    handlers[1].setLevel(logging.INFO)
    if logfile:
        Path(logfile).expanduser().parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(logfile).expanduser()))
    logging.basicConfig(level=level, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cmd_run(args):
    from cyclop.monitor import Monitor
    from cyclop.output import FileWriter, RedisPublisher

    cfg = config.load(args.config)
    cam_cfg = cfg['camera']
    if args.simulate or cam_cfg['simulate']:
        from cyclop.camera import SimCamera

        def factory():
            return SimCamera(frame_rate=cam_cfg['frame_rate'], realtime=True)
    else:
        from cyclop.camera import AravisCamera

        def factory():
            return AravisCamera(address=cam_cfg['address'] or None, exposure_us=cam_cfg['exposure_us'],
                                gain=cam_cfg['gain'], frame_rate=cam_cfg['frame_rate'])

    out = cfg['output']
    writer = FileWriter(args.data_dir or out['data_dir'], tz=cfg['site']['timezone'],
                        save_motion=out['save_motion'],
                        detrend=cfg['measurement']['detrend'])
    publisher = None
    if cfg['redis']['enabled'] and not args.no_redis:
        publisher = RedisPublisher(host=cfg['redis']['host'], port=cfg['redis']['port'])

    stop = {'flag': False}

    def handler(signum, frame):
        logging.getLogger(__name__).info(f"Signal {signum} received, stopping")
        stop['flag'] = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    mon = Monitor(cfg, factory, writer=writer, publisher=publisher)

    web = None
    wcfg = cfg['web']
    if wcfg['enabled'] and not args.no_web:
        from cyclop.web import LogBuffer, WebServer, WebState
        logbuf = next(h for h in logging.getLogger().handlers if isinstance(h, LogBuffer))
        state = WebState(mon, logbuf, history_days=wcfg['history_days'],
                         history_files=[writer.data_dir / "Seeing_Data.txt", *wcfg['history_files']])
        port = args.web_port or wcfg['port']
        try:
            web = WebServer(state, host=wcfg['host'], port=port).start()
        except OSError as e:
            logging.getLogger(__name__).error(f"Web interface not started on port {port}: {e}")

    try:
        mon.run(ignore_sun=args.ignore_sun, max_results=args.n, should_stop=lambda: stop['flag'])
    finally:
        if web:
            web.stop()


def read_motion(path):
    """Read a Windows *_motion.txt file: seconds, x, y, fwhm (separator-agnostic)."""
    import numpy as np
    rows = []
    for line in Path(path).read_text(errors='replace').splitlines():
        parts = line.replace('|', ' ').replace(';', ' ').replace(',', '.').split()
        try:
            rows.append([float(p) for p in parts[:4]])
        except ValueError:
            continue
    return np.array([r for r in rows if len(r) == 4])


def cmd_replay(args):
    cfg = config.load(args.config)
    lat = cfg['site']['latitude']
    for path in args.files:
        a = read_motion(path)
        if len(a) == 0:
            print(f"{path}: no data rows")
            continue
        line = [f"{Path(path).name}: n={len(a)}"]
        for d in (None, 1, 2):
            r = seeing.compute(a[:, 0], a[:, 1], a[:, 2], lat, detrend=d)
            line.append(f"detrend={d}: zen={r['zenith']:.3f} sx={r['sigma_x']:.4f} sy={r['sigma_y']:.4f}")
        print("  ".join(line))


def cmd_camera_info(args):
    import gi
    gi.require_version('Aravis', '0.8')
    from gi.repository import Aravis
    cam = Aravis.Camera.new(args.address)
    print(cam.get_vendor_name(), cam.get_model_name(), cam.get_device_serial_number())
    print("sensor", cam.get_sensor_size(), "region", cam.get_region())
    print("exposure", cam.get_exposure_time(), cam.get_exposure_time_bounds())
    print("gain", cam.get_gain(), cam.get_gain_bounds())
    print("frame rate", cam.get_frame_rate(), cam.get_frame_rate_bounds())
    print("pixel formats", cam.dup_available_pixel_formats_as_display_names())


def main(argv=None):
    p = argparse.ArgumentParser(prog="cyclop", description="Linux replacement for SeeingMonitor_Cyclop")
    p.add_argument('-c', '--config', help="TOML configuration file")
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--log-file')
    sub = p.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help="run the seeing monitor")
    r.add_argument('--simulate', action='store_true', help="use a synthetic camera")
    r.add_argument('--ignore-sun', action='store_true', help="measure regardless of Sun altitude")
    r.add_argument('--no-redis', action='store_true', help="do not publish to redis")
    r.add_argument('--data-dir', help="override output.data_dir")
    r.add_argument('--no-web', action='store_true', help="do not start the web interface")
    r.add_argument('--web-port', type=int, help="override web.port")
    r.add_argument('-n', type=int, help="stop after N seeing measurements")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser('replay', help="reduce Windows *_motion.txt files")
    rp.add_argument('files', nargs='+')
    rp.set_defaults(func=cmd_replay)

    ci = sub.add_parser('camera-info', help="show camera information")
    ci.add_argument('--address', default='192.168.2.59')
    ci.set_defaults(func=cmd_camera_info)

    args = p.parse_args(argv)
    _setup_logging(logging.DEBUG if args.verbose else logging.INFO, args.log_file)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
