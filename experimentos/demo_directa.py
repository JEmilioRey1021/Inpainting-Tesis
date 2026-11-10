"""
Demostración de la ruta directa sobre una imagen real.

    python -m experimentos.demo_directa

Guarda resultados/demo_directa.png con el dato dañado y las cuatro
reconstrucciones, y escribe el error y el tiempo de cada una.
"""

from __future__ import annotations

import os
import numpy as np
from skimage import data, draw
from skimage.morphology import disk, dilation

from inpainting import armonico, calor, perona_malik, variacion_total_estacionaria

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")
os.makedirs(SALIDA, exist_ok=True)


def mascara_trazos(forma, grosor=4, semilla=1):
    """Trazos a mano alzada, como rayones sobre una foto."""
    rng = np.random.default_rng(semilla)
    M = np.zeros(forma, bool)
    for _ in range(7):
        pts = np.cumsum(rng.normal(0, 25, (6, 2)), axis=0) + rng.uniform(80, 430, 2)
        pts = np.clip(pts, 10, np.array(forma) - 11).astype(int)
        for a, b in zip(pts[:-1], pts[1:]):
            rr, cc = draw.line(a[0], a[1], b[0], b[1])
            M[rr, cc] = True
    return dilation(M, disk(grosor))


def psnr(u, ref, D):
    mse = np.mean((u[D] - ref[D]) ** 2)
    return 10 * np.log10(1.0 / mse)


if __name__ == "__main__":
    ref = data.camera().astype(float) / 255
    D = mascara_trazos(ref.shape)
    dato = np.where(D, 0.0, ref)

    modelos = {
        "Armónico": lambda: armonico(dato, D),
        "Calor (T = 20)": lambda: calor(dato, D, T=20, dt=1.0),
        "Perona–Malik": lambda: perona_malik(dato, D, K=0.1, T=200, dt=2.0),
        "Variación total": lambda: variacion_total_estacionaria(dato, D, eps=1e-2,
                                                                tol=1e-5),
    }
    res = {}
    print(f"Píxeles dañados: {D.sum()} ({100 * D.mean():.1f} % de la imagen)\n")
    print(f"{'modelo':18s} {'PSNR en D (dB)':>15} {'tiempo (s)':>11} {'iter':>6}")
    for nombre, f in modelos.items():
        r = f()
        res[nombre] = r
        print(f"{nombre:18s} {psnr(r.u, ref, D):15.2f} {r.tiempo:11.2f} {r.iteraciones:6d}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(12, 8.4))
    ax = ax.ravel()
    vista = np.s_[60:330, 120:420]  # recorte para ver el detalle
    paneles = [("Original", ref), ("Dato dañado", np.where(D, 1.0, ref))] + \
              [(n, r.u) for n, r in res.items()]
    for a, (t, im) in zip(ax, paneles):
        a.imshow(im[vista], cmap="gray", vmin=0, vmax=1)
        a.set_title(t); a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "demo_directa.png"), dpi=130)
