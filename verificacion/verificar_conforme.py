"""
Verificación de la ruta conforme.

    python -m verificacion.verificar_conforme

Cada prueba contrasta el código con un resultado conocido de los capítulos IV
y V. Imprime tablas y guarda figuras en resultados/.
"""

from __future__ import annotations

import os
import time
import numpy as np

from inpainting import MapeoSC, RutaConforme, armonico, perona_malik, \
    variacion_total_estacionaria

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")
os.makedirs(SALIDA, exist_ok=True)


def encabezado(texto):
    print("\n" + "=" * 72 + "\n" + texto + "\n" + "=" * 72)


L_FORMA = np.array([0, 2, 2 + 1j, 1 + 1j, 1 + 2j, 2j])
ESTRELLA = np.array([2 * np.exp(2j * np.pi * k / 10) * (1 if k % 2 == 0 else 0.55)
                     for k in range(10)])


# ----------------------------------------------------------------------
def prueba_mapeo():
    encabezado("1. El mapeo envía prevértices a vértices y phi invierte a psi")
    ok = True
    for nombre, z, c in [("forma de L", L_FORMA, 0.5 + 0.5j),
                         ("estrella de 10 vértices", ESTRELLA, 0)]:
        t0 = time.perf_counter()
        m = MapeoSC(z, c)
        tm = time.perf_counter() - t0
        rng = np.random.default_rng(0)
        # puntos interiores al azar
        from matplotlib.path import Path
        caja = rng.uniform(-2, 2, (4000, 2))
        dentro = Path(np.c_[z.real, z.imag]).contains_points(caja)
        pts = caja[dentro, 0] + 1j * caja[dentro, 1]
        w = m.phi(pts)
        res = np.abs(m.psi(w) - pts).max()
        print(f"   {nombre:24s} |psi(w_k) - z_k| = {m.error_vertices:.1e}   "
              f"|psi(phi(z)) - z| = {res:.1e}   ({len(pts)} puntos, {tm:.2f} s)")
        ok &= m.error_vertices < 1e-10 and res < 1e-9 and np.all(np.abs(w) < 1)
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def prueba_dirichlet_poligono():
    encabezado("3. Problema de Dirichlet en polígonos con solución exacta e^x cos y")
    exacta = lambda z: np.real(np.exp(z))
    ok = True
    for nombre, z, c in [("cuadrado (esquinas de 90°)", np.array([1 + 1j, -1 + 1j,
                                                                 -1 - 1j, 1 - 1j]), 0),
                         ("forma de L (una de 270°)", L_FORMA, 0.5 + 0.5j)]:
        m = MapeoSC(z, c)
        # puntos de prueba: interiores, a distancia >= 0.05 de la frontera
        from matplotlib.path import Path
        g = np.linspace(-2, 2, 161)
        X, Y = np.meshgrid(g, g)
        P = X.ravel() + 1j * Y.ravel()
        camino = Path(np.c_[z.real, z.imag])
        dentro = camino.contains_points(np.c_[P.real, P.imag], radius=-0.1)
        P = P[dentro]
        w = m.phi(P)
        lejos = np.abs(w) < 0.9
        print(f"   {nombre}")
        print(f"   {'términos':>9} | {'uniforme: max':>14} {'|w|<0.9':>9} |"
              f" {'graduada: max':>14} {'|w|<0.9':>9}")
        for M in [64, 128, 256, 512, 1024, 2048]:
            t = 2 * np.pi * np.arange(M) / M
            b, _ = m.psi_circulo(t)
            cu = RutaConforme.coeficientes(exacta(b), t, np.full(M, 2 * np.pi / M),
                                           M // 2 - 1)
            tq, wq = m.cuadratura_circulo(N=M // 2)
            bq, _ = m.psi_circulo(tq)
            cg = RutaConforme.coeficientes(exacta(bq), tq, wq, M // 2 - 1)
            eu = np.abs(RutaConforme.serie(cu, w) - exacta(P))
            eg = np.abs(RutaConforme.serie(cg, w) - exacta(P))
            print(f"   {M // 2 - 1:9d} | {eu.max():14.2e} {eu[lejos].max():9.2e} |"
                  f" {eg.max():14.2e} {eg[lejos].max():9.2e}")
        ok &= eg[lejos].max() < 1e-8 and eg.max() < eu.max()
    print("   Con muestras equiespaciadas, las singularidades |t-t_k|^alpha_k de G"
          " en los\n   prevértices producen aliasing y el error es algebraico"
          " incluso lejos de la\n   frontera. Con la cuadratura graduada, el"
          " interior converge geométricamente y\n   solo queda el truncamiento"
          " cerca de la frontera.")
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def prueba_aglomeracion():
    encabezado("4. Aglomeración en rectángulos de razón de aspecto L")
    print(f"   {'L':>4} {'separación mínima':>18} {'8 e^(-pi L/2)':>15} "
          f"{'|psi(w_k)-z_k|':>15} {'tiempo':>8}")
    filas = []
    for L in [2, 4, 6, 8, 10, 12, 13, 14, 16]:
        t0 = time.perf_counter()
        try:
            m = MapeoSC(np.array([0, L, L + 1j, 1j]), L / 2 + 0.5j)
            filas.append((L, m.separacion_min))
            print(f"   {L:4d} {m.separacion_min:18.2e} {8 * np.exp(-np.pi * L / 2):15.2e}"
                  f" {m.error_vertices:15.1e} {time.perf_counter() - t0:7.2f}s")
        except RuntimeError:
            print(f"   {L:4d} {'no converge':>18} {8 * np.exp(-np.pi * L / 2):15.2e}")
    Ls, s = np.array(filas).T
    pend = np.polyfit(Ls, np.log(s), 1)[0]
    print(f"   pendiente de log(separación) contra L: {pend:.4f}"
          f"  (teoría -pi/2 = {-np.pi / 2:.4f})")
    ok = abs(pend + np.pi / 2) < 0.02
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def malla_imagen(N=256):
    y, x = np.mgrid[0:N, 0:N].astype(float)
    r, th = np.hypot(x - N / 2, y - N / 2), np.arctan2(y - N / 2, x - N / 2)
    D = r < 60 + 20 * np.cos(3 * th)
    return x, y, D


def prueba_mascara_exacta():
    encabezado("5. Máscara de píxeles con dato armónico exacto. Las dos rutas")
    x, y, D = malla_imagen()
    u_ex = np.exp((x - 128) / 80) * np.cos((y - 128) / 80)
    dato = np.where(D, 0, u_ex)
    rc = RutaConforme(D, M=1024)
    c = rc.componentes[0]
    e_c = np.abs(rc.resolver(dato).u - u_ex)
    e_d = np.abs(armonico(dato, D).u - u_ex)
    ec = e_c[c.filas, c.cols]
    lejos = np.abs(c.w) < 0.9
    print(f"   píxeles {D.sum()}, polígono de {c.mapa.n} vértices")
    print(f"   ruta directa   error max = {e_d[D].max():.1e}")
    print(f"   ruta conforme  error max = {ec.max():.1e}  (en |phi(z)|<0.9:"
          f" {ec[lejos].max():.1e})")
    i = np.argmax(ec)
    print(f"   el error máximo de la conforme está en |phi(z)| = {abs(c.w[i]):.3f},"
          f" donde |psi'| = {c.factor[i]:.0f} (mediana {np.median(c.factor):.0f})")
    ok = ec.max() < 1e-2
    print("   OK" if ok else "   FALLA")
    return ok


# ----------------------------------------------------------------------
def prueba_invariancia():
    encabezado("6. Modelos no lineales. Formulación con rho contra malla del disco")
    from skimage import data, transform
    img = transform.resize(data.camera().astype(float) / 255, (256, 256),
                           anti_aliasing=True)
    y, x = np.mgrid[0:256, 0:256].astype(float)
    r, th = np.hypot(x - 150, y - 110), np.arctan2(y - 110, x - 150)
    D = r < 32 * (1 + 0.3 * np.cos(3 * th))          # la mancha grande
    dato = np.where(D, 0, img)
    rc = RutaConforme(D)
    rho = rc.campo_rho()
    # TV estándar en mallas del disco cada vez más finas contra TV con rho.
    # (En el armónico rho no aparece en la ecuación, así que no hace falta
    # comprobarlo.)
    u_rho = variacion_total_estacionaria(dato, D, eps=1e-2, tol=1e-6, rho=rho).u
    u_dir = variacion_total_estacionaria(dato, D, eps=1e-2, tol=1e-6).u
    print(f"   {'malla del disco':>16} {'|disco - rho|':>14} {'|disco - directa|':>18}")
    d_rho = []
    for N in [100, 200, 400]:
        u_d = rc.resolver_disco(dato, "tv", N_disco=N, eps=1e-2, tol=1e-6).u
        d1, d2 = np.abs(u_d - u_rho)[D].mean(), np.abs(u_d - u_dir)[D].mean()
        d_rho.append((d1, d2))
        print(f"   {N:>13} px {d1:14.4f} {d2:18.4f}")
    ok = all(d1 < d2 for d1, d2 in d_rho) and d_rho[-1][0] <= d_rho[0][0]
    print("   La malla del disco queda más cerca de la versión con rho que de la"
          " directa,\n   pero la distancia se estanca por el error de borde de la"
          " malla cartesiana.")
    print("   OK" if ok else "   FALLA")
    return ok


def prueba_poligonos_regulares():
    encabezado("2. Polígonos regulares, cuyo mapeo tiene forma cerrada")
    # Para el n-ágono regular, por simetría las prevértices son las raíces
    # n-ésimas de la unidad, y la fórmula de SC se integra en términos de la
    # función hipergeométrica,
    #     psi(w) = w 2F1(2/n, 1/n; 1 + 1/n; w^n),
    # con vértices psi(e^{2 pi i k/n}) = e^{2 pi i k/n} 2F1(2/n, 1/n; 1+1/n; 1).
    from scipy.special import hyp2f1, gamma
    rng = np.random.default_rng(0)
    print(f"   {'n':>3} {'error prevértices':>18} {'|psi - exacta|':>15}"
          f" {'|phi - exacta^-1|':>18}")
    ok = True
    for n in [3, 4, 6, 8]:
        a, b, c = 2 / n, 1 / n, 1 + 1 / n
        exacta = lambda w: w * hyp2f1(a, b, c, w**n)
        v = gamma(c) * gamma(c - a - b) / (gamma(c - a) * gamma(c - b))
        raices = np.exp(2j * np.pi * np.arange(1, n + 1) / n)
        m = MapeoSC(raices * v, 0)
        e_th = np.max(np.abs(np.exp(1j * m.theta) - raices))
        r = np.sqrt(rng.uniform(0, 0.98**2, 500))
        w = r * np.exp(1j * rng.uniform(0, 2 * np.pi, 500))
        e_psi = np.max(np.abs(m.psi(w) - exacta(w)))
        e_phi = np.max(np.abs(m.phi(exacta(w)) - w))
        print(f"   {n:3d} {e_th:18.1e} {e_psi:15.1e} {e_phi:18.1e}")
        ok &= e_th < 1e-12 and e_psi < 1e-12 and e_phi < 1e-9
    print("   OK" if ok else "   FALLA")
    return ok


if __name__ == "__main__":
    pruebas = [prueba_mapeo, prueba_poligonos_regulares, prueba_dirichlet_poligono,
               prueba_aglomeracion, prueba_mascara_exacta, prueba_invariancia]
    res = [p() for p in pruebas]
    encabezado(f"Resumen: {sum(res)} de {len(res)} pruebas OK")
