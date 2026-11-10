"""
Cómo crece el costo de cada ruta con la resolución.

    python -m experimentos.costo_escala

La misma escena (foto del camarógrafo) y la misma máscara geométrica a
128, 256, 512 y 1024 píxeles de lado. El número de términos de la serie crece
con la resolución, M = 4 N, para que el dato de frontera se resuelva igual
de bien. Se mide la etapa por máscara (factorización o mapeo) y la etapa por
dato de cada ruta, y se ajusta el exponente p de tiempo ~ (píxeles en D)^p.
"""

from __future__ import annotations

import os
import numpy as np
from skimage import data, transform

from inpainting import RutaConforme, SolverArmonico

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")


def escena(N, tipo):
    ref = transform.resize(data.camera().astype(float) / 255, (N, N),
                           anti_aliasing=True, order=3)
    y, x = np.mgrid[0:N, 0:N].astype(float) / N
    if tipo == "mancha":
        r, th = np.hypot(x - 0.586, y - 0.43), np.arctan2(y - 0.43, x - 0.586)
        D = r < 0.125 * (1 + 0.3 * np.cos(3 * th))
    else:  # raya de razón de aspecto 6
        D = (np.abs(x - 0.5) < 0.12) & (np.abs(y - 0.55) < 0.02)
    return ref, D


def psnr(u, ref, D):
    return 10 * np.log10(1 / np.mean((u[D] - ref[D]) ** 2))


if __name__ == "__main__":
    for tipo in ["mancha", "raya"]:
        print(f"\nMáscara: {tipo}")
        print(f"{'N':>5} {'píx D':>7} | {'factoriz.':>9} {'dato dir':>9} |"
              f" {'mapeo':>8} {'dato conf':>9} | {'vértices':>8} {'PSNR dir':>8}"
              f" {'PSNR conf':>9}")
        filas = []
        for N in [128, 256, 512, 1024]:
            ref, D = escena(N, tipo)
            dato = np.where(D, 0.0, ref)
            sd = SolverArmonico(D)
            rc = RutaConforme(D, M=4 * N)
            ud, uc = sd.resolver(dato).u, rc.resolver(dato).u
            td = np.median([sd.resolver(dato).tiempo for _ in range(5)])
            tc = np.median([rc.resolver(dato).tiempo for _ in range(3)])
            nv = sum(c.mapa.n for c in rc.componentes)
            filas.append((D.sum(), sd.tiempo_factorizacion, td, rc.tiempo_mapeo, tc))
            print(f"{N:5d} {D.sum():7d} | {1e3 * sd.tiempo_factorizacion:7.1f}ms"
                  f" {1e3 * td:7.2f}ms | {rc.tiempo_mapeo:7.2f}s {1e3 * tc:7.1f}ms |"
                  f" {nv:8d} {psnr(ud, ref, D):8.2f} {psnr(uc, ref, D):9.2f}"
                  + ("" if not rc.fallas else "  (sin mapeo)"))
        F = np.array(filas)
        p = [np.polyfit(np.log(F[:, 0]), np.log(F[:, j]), 1)[0] for j in (1, 2, 3, 4)]
        print(f"exponentes p (tiempo ~ píxeles^p): factorización {p[0]:.2f}, "
              f"dato directa {p[1]:.2f}, mapeo {p[2]:.2f}, dato conforme {p[3]:.2f}")
