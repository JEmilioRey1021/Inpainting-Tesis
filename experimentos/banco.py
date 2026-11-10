"""
Banco de pruebas, 5 imágenes x 3 familias de máscaras.

    python -m experimentos.banco

Familias
    manchas   seis elipses de radio 4 a 8 px en posiciones al azar
    rayas     una raya de 4 px de ancho y razón de aspecto L = 2, 4, ..., 14
    texto     una palabra superpuesta (letras delgadas, algunas con huecos)

Para cada caso se mide
    - armónico, PSNR de la ruta directa y de la conforme y su diferencia media,
    - TV, PSNR de la ruta directa y del modelo estándar del disco (con rho,
      en la misma malla) y su diferencia media (no invariancia, P2),
    - costo de las etapas por máscara y por dato,
    - cuántas componentes tuvieron mapeo y cuántas se resolvieron por la ruta
      directa (huecos o aglomeración).
Guarda resultados/banco.csv, resultados/banco_resumen.txt y
resultados/banco_rayas.png.
"""

from __future__ import annotations

import csv
import os
import time
import numpy as np
from skimage import color, data, draw, transform

from inpainting import RutaConforme, SolverArmonico, variacion_total_estacionaria

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")
N = 256


def imagenes():
    gris = lambda im: color.rgb2gray(im) if im.ndim == 3 else im / 255.0
    crudas = {"camarógrafo": data.camera(), "astronauta": data.astronaut(),
              "café": data.coffee(), "luna": data.moon(), "gato": data.chelsea()}
    out = {}
    for k, im in crudas.items():
        g = gris(im).astype(float)
        out[k] = transform.resize(g, (N, N), anti_aliasing=True)
    return out


def manchas(semilla):
    rng = np.random.default_rng(semilla)
    D = np.zeros((N, N), bool)
    for _ in range(6):
        f, c = rng.uniform(30, N - 30, 2)
        r = rng.uniform(4, 8)
        rr, cc = draw.ellipse(f, c, r, 1.3 * r, shape=D.shape,
                              rotation=rng.uniform(0, np.pi))
        D[rr, cc] = True
    return D


def raya(L, semilla, ancho=4):
    rng = np.random.default_rng(semilla)
    largo = L * ancho
    ang = rng.uniform(0, np.pi)
    c0 = np.array([N / 2, N / 2]) + rng.uniform(-40, 40, 2)
    y, x = np.mgrid[0:N, 0:N].astype(float)
    u = (x - c0[1]) * np.cos(ang) + (y - c0[0]) * np.sin(ang)
    v = -(x - c0[1]) * np.sin(ang) + (y - c0[0]) * np.cos(ang)
    return (np.abs(u) < largo / 2) & (np.abs(v) < ancho / 2)


def texto(palabra="Tesis UVG"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(N / 100, N / 100), dpi=100)
    fig.text(0.08, 0.45, palabra, fontsize=30, color="k", family="DejaVu Sans")
    fig.canvas.draw()
    im = np.asarray(fig.canvas.buffer_rgba())[..., 0]
    plt.close(fig)
    return im[:N, :N] < 128


def psnr(u, ref, D):
    return 10 * np.log10(1 / np.mean((u[D] - ref[D]) ** 2))


def caso(img_nombre, ref, familia, param, D):
    dato = np.where(D, 0.0, ref)
    t0 = time.perf_counter(); sd = SolverArmonico(D); t_fact = time.perf_counter() - t0
    rc = RutaConforme(D)
    ud = sd.resolver(dato); uc = rc.resolver(dato)
    tv_d = variacion_total_estacionaria(dato, D, eps=1e-2, tol=1e-5).u
    tv_r = variacion_total_estacionaria(dato, D, eps=1e-2, tol=1e-5,
                                        rho=rc.campo_rho()).u
    sep = min([c.mapa.separacion_min for c in rc.componentes], default=np.nan)
    return dict(
        imagen=img_nombre, familia=familia, parametro=param, pixeles=int(D.sum()),
        comp_mapeo=len(rc.componentes), comp_directa=len(rc.respaldo),
        psnr_arm_dir=psnr(ud.u, ref, D), psnr_arm_conf=psnr(uc.u, ref, D),
        dif_arm=float(np.abs(ud.u - uc.u)[D].mean()),
        psnr_tv_dir=psnr(tv_d, ref, D), psnr_tv_disco=psnr(tv_r, ref, D),
        dif_tv=float(np.abs(tv_d - tv_r)[D].mean()),
        t_factorizacion=t_fact, t_mapeo=rc.tiempo_mapeo,
        t_dato_dir=ud.tiempo, t_dato_conf=uc.tiempo, sep_min=sep)


def leer_csv():
    ruta = os.path.join(SALIDA, "banco.csv")
    try:
        with open(ruta, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
    except UnicodeDecodeError:
        # archivos escritos por versiones anteriores en Windows (cp1252)
        with open(ruta, encoding="cp1252") as f:
            filas = list(csv.DictReader(f))
    for r in filas:
        for k, v in r.items():
            if k in ("imagen", "familia"):
                continue
            try:
                r[k] = int(v) if v.isdigit() else float(v)
            except ValueError:
                pass
    return filas


def correr():
    filas = []
    for i, (nom, ref) in enumerate(imagenes().items()):
        print(f"{nom}...", flush=True)
        filas.append(caso(nom, ref, "manchas", "", manchas(i)))
        for L in [2, 4, 6, 8, 10, 12, 14]:
            filas.append(caso(nom, ref, "rayas", L, raya(L, 100 + i)))
        filas.append(caso(nom, ref, "texto", "", texto()))
    with open(os.path.join(SALIDA, "banco.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    return filas


def resumen_y_figura(filas):
    lineas = []
    def p(s=""):
        print(s); lineas.append(s)

    p("Promedios sobre las 5 imágenes")
    p(f"{'familia':10s} {'param':>5} | {'arm dir':>7} {'arm conf':>8} {'dif arm':>8} |"
      f" {'TV dir':>6} {'TV disco':>8} {'dif TV':>7} | {'mapeo':>6} {'fact':>7} |"
      f" {'dato conf/dir':>13} | {'comp. sin mapeo':>15}")
    claves = [("manchas", "")] + [("rayas", L) for L in [2, 4, 6, 8, 10, 12, 14]] + \
             [("texto", "")]
    for fam, par in claves:
        g = [r for r in filas if r["familia"] == fam and r["parametro"] == par]
        m = lambda k: np.mean([r[k] for r in g])
        sin = sum(r["comp_directa"] for r in g); tot = sin + sum(r["comp_mapeo"] for r in g)
        p(f"{fam:10s} {str(par):>5} | {m('psnr_arm_dir'):7.2f} {m('psnr_arm_conf'):8.2f}"
          f" {m('dif_arm'):8.4f} | {m('psnr_tv_dir'):6.2f} {m('psnr_tv_disco'):8.2f}"
          f" {m('dif_tv'):7.4f} | {m('t_mapeo'):5.1f}s {1e3 * m('t_factorizacion'):5.1f}ms |"
          f" {m('t_dato_conf') / m('t_dato_dir'):13.1f} | {sin:>6d} de {tot:<6d}")
    with open(os.path.join(SALIDA, "banco_resumen.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")

    # figura de las rayas en función de L
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = [2, 4, 6, 8, 10, 12, 14]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for nom in imagenes():
        g = [r for r in filas if r["imagen"] == nom and r["familia"] == "rayas"]
        ax[0].semilogy(Ls, [r["sep_min"] for r in g], "o-", label=nom)
        ax[1].plot(Ls, [r["psnr_arm_dir"] - r["psnr_arm_conf"] if r["comp_mapeo"]
                        else np.nan for r in g], "o-")
        ax[2].plot(Ls, [r["dif_tv"] for r in g], "o-")
    # el polígono rodea la raya dilatada, de ancho 4 + 2 y largo 4L + 2, así
    # que su razón de aspecto efectiva es (4L + 2)/6
    Lef = (4 * np.array(Ls) + 2) / 6
    ax[0].semilogy(Ls, 8 * np.exp(-np.pi * Lef / 2), "k--",
                   label="8 exp(-pi L_ef/2), L_ef=(4L+2)/6")
    ax[0].set_title("Separación mínima de prevértices"); ax[0].legend(fontsize=7)
    ax[1].set_title("PSNR directa − PSNR conforme (armónico, dB)")
    ax[2].set_title("Diferencia media TV directa − TV del disco")
    for a in ax:
        a.set_xlabel("razón de aspecto L de la raya"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "banco_rayas.png"), dpi=130)


if __name__ == "__main__":
    filas = leer_csv() if os.environ.get("SOLO_FIGURA") else correr()
    resumen_y_figura(filas)