"""
Las dos rutas con daños realistas, manchas pequeñas y rayones cortos.

    python -m experimentos.demo_danos_pequenos

Es el régimen en que el inpainting por EDP tiene sentido. Guarda
resultados/demo_danos_pequenos.png.
"""

from __future__ import annotations

import os
import numpy as np
from skimage import data, draw
from skimage.morphology import disk, dilation

from inpainting import (RutaConforme, SolverArmonico, perona_malik,
                        variacion_total_estacionaria)

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")


def mascara(forma):
    M = np.zeros(forma, bool)
    # manchas pequeñas, de radio 4 a 8 píxeles
    for (f, c, r) in [(120, 250, 7), (200, 330, 5), (290, 220, 8), (150, 380, 4),
                      (330, 300, 6)]:
        rr, cc = draw.ellipse(f, c, r, r * 1.3, shape=forma, rotation=0.6)
        M[rr, cc] = True
    # rayones cortos de unos 4 píxeles de ancho
    T = np.zeros(forma, bool)
    for (f0, c0, f1, c1) in [(90, 160, 120, 190), (250, 380, 265, 410),
                             (360, 150, 345, 185)]:
        rr, cc = draw.line(f0, c0, f1, c1)
        T[rr, cc] = True
    return M | dilation(T, disk(2))


def psnr(u, ref, D):
    return 10 * np.log10(1 / np.mean((u[D] - ref[D]) ** 2))


if __name__ == "__main__":
    ref = data.camera().astype(float) / 255
    D = mascara(ref.shape)
    dato = np.where(D, 0.0, ref)

    rc = RutaConforme(D, verbose=True)
    res = {
        "Armónico, ruta directa": SolverArmonico(D).resolver(dato).u,
        "Armónico, ruta conforme": rc.resolver(dato).u,
        "TV, ruta directa": variacion_total_estacionaria(dato, D, eps=1e-2,
                                                         tol=1e-5).u,
        "TV estándar en el disco": rc.resolver_disco(dato, "tv", N_disco=120,
                                                     eps=1e-2, tol=1e-5).u,
        "Perona–Malik, ruta directa": perona_malik(dato, D, K=0.05, T=100, dt=2).u,
    }
    print(f"\nPíxeles dañados: {D.sum()}, componentes con mapeo: "
          f"{len(rc.componentes)}, sin mapeo: {len(rc.respaldo)}")
    for k, u in res.items():
        print(f"   {k:28s} PSNR en la máscara = {psnr(u, ref, D):6.2f} dB")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vista = np.s_[70:380, 140:430]
    fig, ax = plt.subplots(2, 4, figsize=(15, 8.3))
    paneles = [("Original", ref), ("Dato dañado", np.where(D, 1.0, ref))] + \
              list(res.items())
    for a, (t, im) in zip(ax.ravel(), paneles):
        a.imshow(im[vista], cmap="gray", vmin=0, vmax=1)
        titulo = t if t in ("Original", "Dato dañado") else \
            f"{t}\n{psnr(im, ref, D):.1f} dB"
        a.set_title(titulo, fontsize=10); a.axis("off")
    # último panel: diferencia entre las dos rutas en el armónico
    dif = np.abs(res["Armónico, ruta directa"] - res["Armónico, ruta conforme"])
    dif = np.where(D, dif, np.nan)
    h = ax[1, 3].imshow(dif[vista], cmap="magma", vmin=0, vmax=0.05)
    ax[1, 3].set_title("|directa − conforme|, armónico", fontsize=10)
    ax[1, 3].axis("off"); fig.colorbar(h, ax=ax[1, 3], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "demo_danos_pequenos.png"), dpi=130)
