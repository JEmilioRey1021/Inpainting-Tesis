"""
Predicción P2 del capítulo V, medida sin error de discretización del disco.

    python -m experimentos.medir_p2

Resolver el modelo ESTÁNDAR en el disco y regresar con phi equivale a
resolver en D un modelo con el campo rho = lambda |phi'| (ver
RutaConforme.campo_rho). Así se puede calcular en la misma malla de píxeles
que la ruta directa, y la diferencia entre las dos es pura no invariancia.

Para el armónico la diferencia es cero por construcción (el laplaciano no ve
rho). Para TV y Perona–Malik se mide la diferencia y si se concentra donde
|grad log rho| es grande.

También se comprueba la equivalencia. En la mancha grande, donde no hay
aglomeración, TV resuelta en una malla cartesiana del disco cada vez más fina
debe acercarse a TV con rho en la malla de píxeles.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.stats import spearmanr
from skimage import data, transform

from inpainting import (RutaConforme, perona_malik, variacion_total_estacionaria)
from experimentos.demo_danos_pequenos import mascara as mascara_pequena
from experimentos.comparar_rutas import mascaras as mascaras_256

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")


def psnr(u, ref, D):
    return 10 * np.log10(1 / np.mean((u[D] - ref[D]) ** 2))


def analizar(nombre, ref, D, tabla, figura=False):
    dato = np.where(D, 0.0, ref)
    rc = RutaConforme(D)
    rho = rc.campo_rho()
    glr = rc.grad_log_rho()
    ok = ~np.isnan(glr)                      # píxeles con mapeo
    modelos = {
        "TV": (lambda r: variacion_total_estacionaria(dato, D, eps=1e-2, tol=1e-5,
                                                      rho=r).u),
        "PM (T=100)": (lambda r: perona_malik(dato, D, K=0.05, T=100, dt=2,
                                              rho=r).u),
        "PM (T=2000)": (lambda r: perona_malik(dato, D, K=0.05, T=2000, dt=20,
                                               rho=r).u),
    }
    res = {}
    for k, f in modelos.items():
        a, b = f(None), f(rho)
        d = np.abs(a - b)
        g, dd = glr[ok], d[ok]
        alto = g >= np.quantile(g, 0.9)
        bajo = g <= np.quantile(g, 0.5)
        tabla.append((nombre, k, psnr(a, ref, D), psnr(b, ref, D), dd.mean(),
                      dd[alto].mean(), dd[bajo].mean(), spearmanr(dd, g)[0]))
        res[k] = (a, b, d)
    if figura:
        _figura(nombre, ref, D, rho, glr, res)
    return rc


def _figura(nombre, ref, D, rho, glr, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    a, b, d = res["TV"]
    # acercamiento al rayón de arriba a la izquierda
    vista = np.s_[78:132, 150:204]
    fig, ax = plt.subplots(1, 5, figsize=(19, 4.4))
    r = np.where(D, rho, np.nan)
    g = np.where(D, glr, np.nan)
    dd = np.where(D, d, np.nan)
    paneles = [("TV, ruta directa", a, "gray", None),
               ("TV estándar del disco, en D", b, "gray", None),
               ("|diferencia|", dd, "magma", None),
               ("rho = lambda |phi'|", r, "viridis", LogNorm()),
               ("|grad log rho|", g, "viridis", LogNorm(1e-3, 1))]
    for axi, (t, im, cm, nrm) in zip(ax, paneles):
        kw = dict(vmin=0, vmax=1) if cm == "gray" else {}
        h = axi.imshow(im[vista], cmap=cm, norm=nrm, interpolation="nearest", **kw)
        axi.set_title(t, fontsize=10); axi.axis("off")
        if cm != "gray":
            fig.colorbar(h, ax=axi, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "p2_" + nombre.replace(" ", "_") + ".png"),
                dpi=130)


def comprobar_equivalencia():
    print("\nEquivalencia, TV en una malla del disco contra TV con rho en D"
          " (mancha grande, 256 x 256)")
    ref = transform.resize(data.camera().astype(float) / 255, (256, 256),
                           anti_aliasing=True)
    D = mascaras_256()["mancha"]
    dato = np.where(D, 0.0, ref)
    rc = RutaConforme(D)
    u_rho = variacion_total_estacionaria(dato, D, eps=1e-2, tol=1e-6,
                                         rho=rc.campo_rho()).u
    u_dir = variacion_total_estacionaria(dato, D, eps=1e-2, tol=1e-6).u
    print(f"   {'malla del disco':>16} {'|disco - rho| medio':>20}"
          f" {'|disco - directa| medio':>24}")
    for N in [100, 200, 400]:
        u_d = rc.resolver_disco(dato, "tv", N_disco=N, eps=1e-2, tol=1e-6).u
        print(f"   {N:>13} px {np.abs(u_d - u_rho)[D].mean():20.4f}"
              f" {np.abs(u_d - u_dir)[D].mean():24.4f}")


if __name__ == "__main__":
    tabla = []
    ref512 = data.camera().astype(float) / 255
    analizar("daños pequeños", ref512, mascara_pequena(ref512.shape), tabla,
             figura=True)
    ref256 = transform.resize(ref512, (256, 256), anti_aliasing=True)
    for k in ["mancha", "raya L=8"]:
        analizar(k, ref256, mascaras_256()[k], tabla)

    print(f"{'máscara':15s} {'modelo':12s} {'PSNR dir':>8} {'PSNR disco':>10}"
          f" {'dif media':>9} {'dif 10% alto':>12} {'dif 50% bajo':>12}"
          f" {'Spearman':>9}")
    for fila in tabla:
        print(f"{fila[0]:15s} {fila[1]:12s} {fila[2]:8.2f} {fila[3]:10.2f}"
              f" {fila[4]:9.4f} {fila[5]:12.4f} {fila[6]:12.4f} {fila[7]:9.2f}")
    print("(dif 10% alto / 50% bajo: diferencia media en los píxeles con el 10% más"
          " alto / el 50% más bajo de |grad log rho|)")
    comprobar_equivalencia()
