"""
Primera comparación entre la ruta directa y la ruta conforme.

    python -m experimentos.comparar_rutas

Sobre la foto del camarógrafo (256 x 256) y cuatro máscaras:
    mancha        una región irregular redondeada
    tres manchas  tres regiones separadas (tres mapeos)
    raya L=8      un rectángulo de 6 x 48 píxeles
    raya L=16     un rectángulo de 6 x 96 píxeles (aglomeración extrema)

Mide, para el modelo armónico, el PSNR de cada ruta, la diferencia entre
ellas, el costo de la etapa por máscara (factorización o mapeo) y el de la
etapa por dato, y el número de datos k* a partir del cual la ruta conforme
sería más barata. Para PM y TV compara la ruta directa con el modelo estándar
en el disco y mide si la diferencia se concentra donde |psi''/psi'| es grande
(predicción P2 del capítulo V).
"""

from __future__ import annotations

import os
import numpy as np
from scipy.stats import spearmanr
from skimage import data, transform

from inpainting import (RutaConforme, SolverArmonico, perona_malik,
                        variacion_total_estacionaria)

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")
os.makedirs(SALIDA, exist_ok=True)
N = 256


def mascaras():
    y, x = np.mgrid[0:N, 0:N].astype(float)

    def mancha(cx, cy, r0, a, k, fase=0):
        r, th = np.hypot(x - cx, y - cy), np.arctan2(y - cy, x - cx)
        return r < r0 * (1 + a * np.cos(k * th + fase))

    return {
        "mancha": mancha(150, 110, 32, 0.3, 3),
        "tres manchas": mancha(70, 80, 14, 0.25, 3) | mancha(180, 70, 12, 0.3, 2, 1)
                        | mancha(140, 190, 16, 0.2, 4),
        "raya L=8": (np.abs(x - 128) < 24) & (np.abs(y - 140) < 3),
        "raya L=16": (np.abs(x - 128) < 48) & (np.abs(y - 140) < 3),
    }


def psnr(u, ref, D):
    return 10 * np.log10(1 / np.mean((u[D] - ref[D]) ** 2))


def log_derivada(c):
    """|psi''/psi'| = |sum_k beta_k/(w - w_k)| en phi(z) (fórmula de SC)."""
    m = c.mapa
    return np.abs(np.sum(m.beta / (c.w[:, None] - m.w[None, :]), axis=1))


if __name__ == "__main__":
    ref = transform.resize(data.camera().astype(float) / 255, (N, N),
                           anti_aliasing=True)
    print(f"{'máscara':13s} {'píx':>5} | {'PSNR dir':>8} {'PSNR conf':>9} "
          f"{'dif max':>8} | {'factoriz.':>9} {'mapeo':>8} | {'dato dir':>8} "
          f"{'dato conf':>9} | notas")
    figs = {}
    for nombre, D in mascaras().items():
        dato = np.where(D, 0.0, ref)
        sd = SolverArmonico(D)
        rc = RutaConforme(D)
        ud = sd.resolver(dato).u
        uc = rc.resolver(dato).u
        td = np.median([sd.resolver(dato).tiempo for _ in range(5)])
        tc = np.median([rc.resolver(dato).tiempo for _ in range(5)])
        notas = ""
        if rc.fallas:
            notas = f"{len(rc.fallas)} componente(s) sin mapeo, ruta directa"
        print(f"{nombre:13s} {D.sum():5d} | {psnr(ud, ref, D):8.2f} "
              f"{psnr(uc, ref, D):9.2f} {np.abs(ud - uc)[D].max():8.1e} | "
              f"{1e3 * sd.tiempo_factorizacion:7.1f}ms {rc.tiempo_mapeo:7.2f}s | "
              f"{1e3 * td:6.2f}ms {1e3 * tc:7.2f}ms | {notas}")
        figs[nombre] = (D, dato, ud, uc, rc)

    # ------------------------------------------------------------------
    print("\nModelos no invariantes en la máscara 'mancha'")
    D, dato, ud, uc, rc = figs["mancha"]
    c = rc.componentes[0]
    ld = log_derivada(c)
    no_inv = {
        "Perona–Malik": (perona_malik(dato, D, K=0.05, T=300, dt=5).u,
                         rc.resolver_disco(dato, "perona_malik", N_disco=300,
                                           K=0.05, T=300, dt=5).u),
        "TV": (variacion_total_estacionaria(dato, D, eps=1e-2, tol=1e-5).u,
               rc.resolver_disco(dato, "tv", N_disco=300, eps=1e-2, tol=1e-5).u),
    }
    arm_disco = rc.resolver_disco(dato, "armonico", N_disco=300).u
    d_arm = np.abs(arm_disco - ud)[c.filas, c.cols]
    titulo_sp = "Spearman(|dif|, |psi''/psi'|)"
    print(f"   {'modelo':13s} {'dif media':>9} {'PSNR dir':>9} {'PSNR disco':>10}"
          f" {titulo_sp:>30}")
    print(f"   {'armónico':13s} {d_arm.mean():9.4f} {psnr(ud, ref, D):9.2f} "
          f"{psnr(arm_disco, ref, D):10.2f} {spearmanr(d_arm, ld)[0]:30.2f}")
    for k, (a, b) in no_inv.items():
        d = np.abs(a - b)[c.filas, c.cols]
        print(f"   {k:13s} {d.mean():9.4f} {psnr(a, ref, D):9.2f} "
              f"{psnr(b, ref, D):10.2f} {spearmanr(d, ld)[0]:30.2f}")

    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    vista = np.s_[60:165, 95:205]
    fig, ax = plt.subplots(2, 4, figsize=(14, 7.2))
    fac = np.full(D.shape, np.nan); fac[c.filas, c.cols] = c.factor
    dif = np.full(D.shape, np.nan); dif[D] = np.abs(ud - uc)[D] + 1e-16
    a, b = no_inv["TV"]
    dtv = np.full(D.shape, np.nan); dtv[D] = np.abs(a - b)[D]
    paneles = [
        ("Dato", np.where(D, 1.0, ref), "gray", None),
        ("Armónico, ruta directa", ud, "gray", None),
        ("Armónico, ruta conforme", uc, "gray", None),
        ("|directa − conforme| (armónico)", dif, "magma", LogNorm(1e-6, 1e-1)),
        ("Factor conforme |ψ'| en la máscara", fac, "viridis", LogNorm()),
        ("TV, ruta directa", a, "gray", None),
        ("TV estándar en el disco", b, "gray", None),
        ("|directa − disco| (TV)", dtv, "magma", None),
    ]
    for axi, (t, im, cm, nrm) in zip(ax.ravel(), paneles):
        kw = dict(vmin=0, vmax=1) if cm == "gray" else {}
        h = axi.imshow(im[vista], cmap=cm, norm=nrm, **kw)
        axi.set_title(t, fontsize=10); axi.axis("off")
        if cm != "gray":
            fig.colorbar(h, ax=axi, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "comparacion_mancha.png"), dpi=130)
    print("\nFigura guardada en resultados/comparacion_mancha.png")
