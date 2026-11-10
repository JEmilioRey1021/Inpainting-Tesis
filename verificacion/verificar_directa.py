"""
Verificación de la ruta directa.

Cada prueba compara el código con algo que se sabe de antemano por la teoría
de los capítulos IV y V. Ejecutar desde la carpeta del proyecto con

    python -m verificacion.verificar_directa

Imprime una tabla por prueba y guarda las figuras en resultados/.
"""

from __future__ import annotations

import os
import numpy as np
import scipy.sparse.linalg as spla

from inpainting import (Malla, SolverArmonico, armonico, calor, perona_malik,
                        variacion_total, variacion_total_estacionaria)

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")
os.makedirs(SALIDA, exist_ok=True)


def malla_cuadrado(N, L=1.0):
    """Centros de píxel en [-L, L]^2 con N x N píxeles. Devuelve X, Y, h."""
    h = 2 * L / N
    t = -L + h * (np.arange(N) + 0.5)
    X, Y = np.meshgrid(t, t)
    return X, Y, h


def encabezado(texto):
    print("\n" + "=" * 72 + "\n" + texto + "\n" + "=" * 72)


# ----------------------------------------------------------------------
def prueba_simetria():
    encabezado("1. La matriz A es simétrica y definida positiva")
    X, Y, h = malla_cuadrado(40)
    D = X**2 + 0.5 * Y**2 < 0.6
    A, _ = Malla(D, h).ensamblar(np.zeros_like(X))
    asim = abs(A - A.T).max()
    lam_min = spla.eigsh(A, k=1, sigma=0, which="LM")[0][0]
    print(f"   max |A - A^T| = {asim:.1e}    menor valor propio = {lam_min:.4f}")
    ok = asim < 1e-12 and lam_min > 0
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def prueba_convergencia():
    encabezado("2. Orden de convergencia del armónico (solución exacta e^x cos y)")
    exacta = lambda x, y: np.exp(x) * np.cos(y)
    print(f"   {'N':>5} {'h':>9} {'error max':>12} {'orden':>7}")
    errores, hs = [], []
    for N in [32, 64, 128, 256, 512]:
        X, Y, h = malla_cuadrado(N)
        # máscara no convexa, en forma de riñón, para que no sea un caso trivial
        r, th = np.hypot(X, Y), np.arctan2(Y, X)
        D = r < 0.7 + 0.2 * np.cos(3 * th)
        u_ex = exacta(X, Y)
        dato = np.where(D, 0.0, u_ex)
        u = armonico(dato, D, h).u
        err = np.abs(u - u_ex)[D].max()
        orden = "" if not errores else f"{np.log2(errores[-1] / err):7.2f}"
        print(f"   {N:5d} {h:9.5f} {err:12.3e} {orden}")
        errores.append(err); hs.append(h)
    p = np.polyfit(np.log(hs), np.log(errores), 1)[0]
    print(f"   orden ajustado = {p:.2f} (se espera 2)")
    ok = 1.8 < p < 2.2
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def prueba_maximo():
    encabezado("3. Principio del máximo discreto con dato aleatorio")
    rng = np.random.default_rng(0)
    X, Y, h = malla_cuadrado(100)
    D = (np.abs(X) < 0.5) & (np.abs(Y) < 0.3) | (X**2 + (Y - 0.4)**2 < 0.1)
    dato = rng.uniform(-1, 1, X.shape)
    u = armonico(dato, D, h).u
    # vecinos conocidos de la máscara (el "borde" discreto)
    borde = np.zeros_like(D)
    borde[1:, :] |= D[:-1, :]; borde[:-1, :] |= D[1:, :]
    borde[:, 1:] |= D[:, :-1]; borde[:, :-1] |= D[:, 1:]
    borde &= ~D
    lo, hi = dato[borde].min(), dato[borde].max()
    print(f"   dato en el borde en [{lo:.4f}, {hi:.4f}]")
    print(f"   solución en D   en [{u[D].min():.4f}, {u[D].max():.4f}]")
    ok = u[D].min() >= lo - 1e-12 and u[D].max() <= hi + 1e-12
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def prueba_calor():
    encabezado("4. El calor converge al armónico cuando T crece")
    X, Y, h = malla_cuadrado(80)
    D = X**2 + Y**2 < 0.5
    dato = np.where(D, 0.0, np.sin(3 * X) + Y**2)
    u_arm = armonico(dato, D, h).u
    print(f"   {'T':>7} {'max |u_T - u_arm|':>20}")
    difs = []
    for T in [0.01, 0.05, 0.1, 0.2, 0.4]:
        u = calor(dato, D, T=T, dt=1e-3, h=h).u
        d = np.abs(u - u_arm)[D].max()
        difs.append(d)
        print(f"   {T:7.2f} {d:20.3e}")
    ok = all(a > b for a, b in zip(difs, difs[1:])) and difs[-1] < 1e-2
    # La diferencia decae como e^{-lambda T}. En un disco, el relleno inicial
    # (el promedio del dato en el borde) no tiene componente en la primera
    # función propia phi_1, porque por la fórmula de Green
    #   int u_arm phi_1 = (1/lambda_1) int_borde u (-d_n phi_1)
    # y en el disco d_n phi_1 es constante en el borde. Por eso la tasa
    # observada es el SEGUNDO valor propio, no el primero.
    A, _ = Malla(D, h).ensamblar(np.zeros_like(X))
    lams = np.sort(spla.eigsh(A, k=3, sigma=0, which="LM")[0])
    tasa = -np.polyfit([0.1, 0.2, 0.4], np.log(difs[2:]), 1)[0]
    print(f"   tasa observada {tasa:.1f}   valores propios discretos"
          f" lambda_1 = {lams[0]:.1f}, lambda_2 = {lams[1]:.1f}")
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def prueba_rampa():
    encabezado("5. Una rampa lineal es estacionaria para Perona–Malik y TV")
    X, Y, h = malla_cuadrado(60)
    D = (X - 0.1)**2 + Y**2 < 0.4
    rampa = 0.7 * X - 0.3 * Y
    ok = True
    for nombre, fn in [
        ("Perona–Malik", lambda: perona_malik(rampa, D, K=0.5, T=1.0, dt=0.1,
                                              h=h, x0=rampa[D])),
        ("TV (flujo)", lambda: variacion_total(rampa, D, eps=1e-2, T=1.0,
                                               dt=0.1, h=h, x0=rampa[D])),
    ]:
        d = np.abs(fn().u - rampa).max()
        print(f"   {nombre:14s} partiendo de la rampa, max cambio = {d:.1e}")
        ok &= d < 1e-10
    # además, partiendo del relleno constante, deben llegar a la rampa
    for nombre, fn in [
        ("Perona–Malik", lambda: perona_malik(rampa, D, K=5.0, T=5.0, dt=0.05, h=h)),
        ("TV estac.", lambda: variacion_total_estacionaria(rampa, D, eps=1e-2, h=h)),
    ]:
        d = np.abs(fn().u - rampa).max()
        print(f"   {nombre:14s} partiendo de una constante,  error final = {d:.1e}")
        ok &= d < 1e-3
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def prueba_escalon():
    encabezado("6. Escalón vertical. TV lo conserva y el armónico lo difumina")
    N = 64
    X, Y, h = malla_cuadrado(N)
    escalon = (X > 0.0).astype(float)
    D = (np.abs(X) < 0.4) & (np.abs(Y) < 0.4)
    dato = np.where(D, 0.5, escalon)
    u_arm = armonico(dato, D, h).u
    u_tv = variacion_total_estacionaria(dato, D, eps=1e-3, h=h, tol=1e-8).u
    u_pm = perona_malik(dato, D, K=3.0, T=2.0, dt=0.01, h=h).u
    for nombre, u in [("armónico", u_arm), ("Perona–Malik", u_pm), ("TV", u_tv)]:
        e = np.abs(u - escalon)[D]
        print(f"   {nombre:13s} error medio = {e.mean():.3f}   error max = {e.max():.3f}")
    ok = np.abs(u_tv - escalon)[D].mean() < 0.25 * np.abs(u_arm - escalon)[D].mean()
    _figura_escalon(escalon, D, u_arm, u_pm, u_tv)
    print("   OK" if ok else "   FALLA")
    return ok


def _figura_escalon(escalon, D, u_arm, u_pm, u_tv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.5))
    for a, (t, im) in zip(ax, [("Dato (máscara en blanco)", np.where(D, 1.0, escalon)),
                               ("Armónico", u_arm), ("Perona–Malik", u_pm),
                               ("Variación total", u_tv)]):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.contour(D, levels=[0.5], colors="tab:red", linewidths=0.8)
        a.set_title(t); a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "verificacion_escalon.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
def prueba_amortizacion():
    encabezado("7. Amortización de la ruta directa (imagen de 512 x 512)")
    from skimage import data
    img = data.camera().astype(float) / 255
    X, Y, _ = malla_cuadrado(512)
    D = ((X + 0.2)**2 / 0.15 + (Y - 0.1)**2 / 0.04 < 1) | \
        ((np.abs(X - 0.4) < 0.05) & (np.abs(Y) < 0.6))
    s = SolverArmonico(D)
    tiempos = [s.resolver(img).tiempo for _ in range(10)]
    print(f"   píxeles en la máscara      {D.sum():d}")
    print(f"   factorización (una vez)    {1e3 * s.tiempo_factorizacion:8.2f} ms")
    print(f"   resolución por dato        {1e3 * np.median(tiempos):8.2f} ms (mediana de 10)")
    return True


# ----------------------------------------------------------------------
if __name__ == "__main__":
    pruebas = [prueba_simetria, prueba_convergencia, prueba_maximo,
               prueba_calor, prueba_rampa, prueba_escalon, prueba_amortizacion]
    resultados = [p() for p in pruebas]
    encabezado(f"Resumen: {sum(resultados)} de {len(resultados)} pruebas OK")
