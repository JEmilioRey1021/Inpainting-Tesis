"""
Figuras del capítulo VII a partir de resultados/banco.csv.

    python -m experimentos.figuras_tesis
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experimentos.banco import leer_csv

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")
Ls = [2, 4, 6, 8, 10, 12, 14]


def figura_armonico(filas):
    imgs = sorted({r["imagen"] for r in filas}, key=lambda k: [r["imagen"] for r in filas].index(k))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for nom in imgs:
        g = {r["parametro"]: r for r in filas if r["imagen"] == nom and r["familia"] == "rayas"}
        ax[0].semilogy(Ls, [g[L]["sep_min"] for L in Ls], "o-", label=nom)
        ax[1].plot(Ls, [g[L]["psnr_arm_dir"] - g[L]["psnr_arm_conf"]
                        if g[L]["comp_mapeo"] else np.nan for L in Ls], "o-", label=nom)
    Lef = (4 * np.array(Ls) + 2) / 6
    ax[0].semilogy(Ls, 8 * np.exp(-np.pi * Lef / 2), "k--",
                   label=r"$8e^{-\pi L_{ef}/2}$, $L_{ef}=(4L+2)/6$")
    ax[0].set_title("(a) Separación mínima entre prevértices")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("(b) PSNR directa − PSNR conforme, armónico (dB)")
    for a in ax:
        a.set_xlabel("Razón de aspecto L de la raya"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "banco_armonico.png"), dpi=150)


def figura_tv(filas):
    imgs = sorted({r["imagen"] for r in filas}, key=lambda k: [r["imagen"] for r in filas].index(k))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for nom in imgs:
        g = {r["parametro"]: r for r in filas if r["imagen"] == nom and r["familia"] == "rayas"}
        ok = lambda L: g[L]["comp_mapeo"] > 0
        ax[0].plot(Ls, [g[L]["psnr_tv_dir"] - g[L]["psnr_tv_disco"] if ok(L) else np.nan
                        for L in Ls], "o-", label=nom)
        ax[1].plot(Ls, [g[L]["dif_tv"] if ok(L) else np.nan for L in Ls], "o-", label=nom)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_title("(a) PSNR TV directa − PSNR TV del disco (dB)")
    ax[1].set_title("(b) Diferencia media entre las dos versiones de TV")
    for a in ax:
        a.set_xlabel("Razón de aspecto L de la raya"); a.grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "banco_tv.png"), dpi=150)




def figura_costo(ruta_csv=None):
    """Tiempos de la corrida final (computadora de Emilio) en escala log-log."""
    import csv
    ruta_csv = ruta_csv or os.path.join(os.path.dirname(__file__), "..",
                                        "resultados_finales", "costo_escala_emilio.csv")
    filas = list(csv.DictReader(open(ruta_csv, encoding="utf-8")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (masc, titulo) in zip(ax, [("mancha", "(a) Mancha"), ("raya", "(b) Raya")]):
        g = [r for r in filas if r["mascara"] == masc]
        n = np.array([float(r["pixeles"]) for r in g])
        serie = [("Factorización", [float(r["factorizacion_ms"]) / 1e3 for r in g], "C0", "-"),
                 ("Mapeo", [float(r["mapeo_s"]) for r in g], "C3", "-"),
                 ("Por dato, directa", [float(r["dato_dir_ms"]) / 1e3 for r in g], "C0", "--"),
                 ("Por dato, conforme", [float(r["dato_conf_ms"]) / 1e3 for r in g], "C3", "--")]
        for nombre, t, col, est in serie:
            a.loglog(n, t, est, color=col, marker="o", label=nombre)
        a.set_xlabel("Píxeles en la máscara"); a.set_ylabel("Tiempo (s)")
        a.set_title(titulo); a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "costo_escala.png"), dpi=150)


if __name__ == "__main__":
    filas = leer_csv()
    figura_armonico(filas)
    figura_tv(filas)
    figura_costo()
