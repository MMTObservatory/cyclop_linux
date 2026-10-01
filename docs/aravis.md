# Building Aravis into the conda environment

conda-forge has no Aravis package, and mixing the system's Aravis/GLib with conda's PyGObject
does not work reliably, so Aravis is built from source into the `cyclop` environment.

```
conda install -n cyclop -c conda-forge pygobject gobject-introspection glib libxml2 zlib \
    meson ninja pkg-config c-compiler cxx-compiler
E=~/miniconda3/envs/cyclop
curl -LO https://github.com/AravisProject/aravis/releases/download/0.8.35/aravis-0.8.35.tar.xz
tar xf aravis-0.8.35.tar.xz && cd aravis-0.8.35
PATH=$E/bin:$PATH PKG_CONFIG_PATH=$E/lib/pkgconfig:$E/share/pkgconfig \
  meson setup build --prefix=$E --libdir=lib -Dviewer=disabled -Dgst-plugin=disabled \
    -Dusb=disabled -Dpacket-socket=enabled -Dintrospection=enabled -Ddocumentation=disabled -Dtests=false
PATH=$E/bin:$PATH ninja -C build install
```

This also installs `arv-tool-0.8` and `arv-camera-test-0.8` into the environment.

## WSL2 notes

With `networkingMode=mirrored`, broadcast GigE Vision discovery replies do not reach WSL, so
`arv-tool-0.8` with no arguments finds nothing. Unicast works: always address the camera by IP
(`camera.address = "192.168.2.59"`, `arv-tool-0.8 -a 192.168.2.59 ...`).
