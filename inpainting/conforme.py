"""
Ruta conforme.

Los cuatro pasos de la sección V.B de la tesis:

    1. Calcular el mapeo. Se extrae un polígono que encierra cada componente
       de la máscara, se resuelve el problema de parámetros de
       Schwarz–Christoffel, se calcula psi en M puntos del círculo y
       phi = psi^{-1} en los píxeles de la máscara.
    2. Trasladar el dato. G(e^{it_j}) = g(psi(e^{it_j})), interpolando la
       imagen en los puntos psi(e^{it_j}) del polígono.
    3. Resolver en el disco. Coeficientes de Fourier de G. Por defecto se
       calculan con una cuadratura graduada hacia los prevértices, porque G
       tiene singularidades |t - t_k|^alpha_k ahí y las muestras
       equiespaciadas (la FFT) producen aliasing. La opción
       muestreo='uniforme' usa M muestras equiespaciadas, es decir, la DFT.
    4. Regresar. u(z) = Re( c_0 + 2 sum_{n>=1} c_n phi(z)^n ) en cada píxel.

El paso 1 depende solo de la máscara (etapa por máscara). Los pasos 2 a 4 se
repiten para cada dato (etapa por dato), y es lo que se compara con las dos
sustituciones triangulares de la ruta directa.

Para los modelos no lineales, el paso 3 se cambia por la resolución del
modelo ESTÁNDAR en el disco, discretizado en una malla cartesiana del disco
con el mismo código de la ruta directa. Por el capítulo V, esto NO resuelve el
mismo problema que la ruta directa, y la diferencia es lo que se quiere medir.

Decisiones de modelado (van en la metodología)
----------------------------------------------
- El polígono es la curva de nivel `nivel` (0.9) de la máscara dilatada un
  píxel, simplificada con Douglas–Peucker. Pasa a una décima de píxel de los
  centros del primer anillo de píxeles conocidos, que son justamente los que
  la ruta directa usa como dato de Dirichlet. Así las dos rutas resuelven,
  hasta O(h), el mismo problema, y el dato g se interpola solo con datos
  verdaderos.
- Componentes que no son simplemente conexas (máscaras con huecos) no admiten
  un mapeo de Riemann (Observación sobre el anillo, capítulo IV). Se informan y
  se resuelven con la ruta directa, marcadas en `self.respaldo`.
"""

from __future__ import annotations

import time
import numpy as np
from matplotlib.path import Path
from scipy import ndimage as ndi
from scipy.special import roots_legendre
from skimage.measure import approximate_polygon, find_contours

from .directa import Resultado, SolverArmonico
from . import directa
from .sc import MapeoSC


class Componente:
    """Datos de la etapa por máscara para una componente conexa."""
    pass


class RutaConforme:
    def __init__(self, mascara, M=1024, dilatacion=1, nivel=0.9, tol_poligono=0.8,
                 muestreo="graduado", orden_interp=1, r_umbral=None,
                 verbose=False):
        t0 = time.perf_counter()
        D = np.asarray(mascara, bool)
        self.D = D
        self.nivel = nivel
        # |w| a partir del cual se usa la integral de Poisson. Por defecto, el
        # radio en que r^(número de términos) = 1e-12, es decir, donde la serie
        # truncada deja de tener 12 cifras.
        if r_umbral is None:
            r_umbral = max(0.95, 1e-12 ** (1.0 / (M // 2 - 1)))
        self.r_umbral = r_umbral
        self.M = M                      # número de términos de la serie
        self.muestreo = muestreo        # 'graduado' o 'uniforme'
        self.orden_interp = orden_interp  # 1 bilineal, 3 bicúbica
        # índices del píxel conocido más cercano, para rellenar la máscara al
        # interpolar el dato (en la práctica casi no se usan)
        _, self._ind = ndi.distance_transform_edt(D, return_indices=True)

        R = ndi.binary_dilation(D, structure=np.ones((3, 3)), iterations=dilatacion)
        etiquetas, ncomp = ndi.label(R, structure=np.ones((3, 3)))
        self.componentes, self.respaldo, self.fallas = [], [], []
        for e in range(1, ncomp + 1):
            Rc = etiquetas == e
            Dc = D & Rc
            if not Dc.any():
                continue
            if (ndi.binary_fill_holes(Rc) != Rc).any():
                self.respaldo.append(Dc)
                if verbose:
                    print(f"   componente {e}: tiene huecos, se usa la ruta directa")
                continue
            try:
                self.componentes.append(self._preparar(Rc, Dc, tol_poligono, verbose))
            except (RuntimeError, ValueError) as ex:
                # típicamente, aglomeración extrema: el problema de parámetros
                # no converge en doble precisión
                self.respaldo.append(Dc)
                self.fallas.append(str(ex))
                if verbose:
                    print(f"   componente {e}: falla el mapeo ({ex}), se usa la ruta directa")
        self._respaldo_solver = (SolverArmonico(np.logical_or.reduce(self.respaldo))
                                 if self.respaldo else None)
        self.tiempo_mapeo = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _preparar(self, Rc, Dc, tol, verbose):
        c = Componente()
        cont = max(find_contours(np.pad(Rc, 1).astype(float), self.nivel),
                   key=len) - 1
        filas, cols = np.nonzero(Dc)
        zpix = cols + 1j * filas
        # simplificar el contorno sin dejar píxeles de la máscara afuera
        while True:
            pol = approximate_polygon(cont, tolerance=tol)[:-1]
            camino = Path(np.c_[pol[:, 1], pol[:, 0]])
            if camino.contains_points(np.c_[cols, filas]).all() or tol < 0.1:
                break
            tol *= 0.5
        z = pol[:, 1] + 1j * pol[:, 0]
        # centro conforme: entre los píxeles casi tan alejados de la frontera
        # como el más alejado, el más cercano al centroide. En una raya, el
        # máximo de la distancia se alcanza en toda la línea media, y escoger
        # un extremo aglomera exponencialmente el otro.
        dist = np.where(Dc, ndi.distance_transform_edt(Rc), -1)
        cand = np.argwhere(dist >= 0.9 * dist.max())
        cen = np.array([filas.mean(), cols.mean()])
        i, j = cand[np.argmin(np.sum((cand - cen) ** 2, axis=1))]
        c.mapa = MapeoSC(z, j + 1j * i)
        if self.muestreo == "uniforme":
            c.tq = 2 * np.pi * np.arange(self.M) / self.M
            c.wq = np.full(self.M, 2 * np.pi / self.M)
        else:
            c.tq, c.wq = c.mapa.cuadratura_circulo(N=self.M // 2)
        c.frontera, _ = c.mapa.psi_circulo(c.tq)        # psi(e^{it_q})
        c.w = c.mapa.phi(zpix)                           # phi en los píxeles
        c.filas, c.cols = filas, cols
        c.factor = np.abs(c.mapa.dpsi(c.w))              # |psi'| en phi(z)
        if self.muestreo != "uniforme":
            self._preparar_cercanos(c)
        if verbose:
            print(f"   componente: {Dc.sum()} píxeles, polígono de {c.mapa.n} "
                  f"vértices, separación mínima de prevértices "
                  f"{c.mapa.separacion_min:.1e}, residuo de phi "
                  f"{c.mapa.residuo_inversa:.1e}")
        return c

    # ------------------------------------------------------------------
    def _preparar_cercanos(self, c, nodos=8):
        """Pesos de la integral de Poisson para los píxeles con |phi(z)| > r_umbral.

        La serie necesita del orden de 1/(1-|w|) términos, y en las máscaras
        alargadas hay píxeles con |w| = 0.99999. Para ellos se evalúa
        directamente la integral de Poisson

            u(w) = (1/2pi) int P(w, t) G(t) dt,  P = (1-|w|^2)/|e^{it}-w|^2,

        con la cuadratura global, salvo en los paneles cercanos a arg(w), que
        se sustituyen por paneles graduados hacia arg(w) a escala 1-|w|, que
        es el ancho del pico del núcleo. Todo esto depende solo de la máscara.
        """
        x, wl = roots_legendre(nodos)
        cerca = np.abs(c.w) > self.r_umbral
        c.cerca = np.nonzero(cerca)[0]
        m = c.mapa
        pan = m.paneles
        nq = c.tq.size
        # en máscaras grandes esta matriz es grande, y se guarda en precisión
        # simple (7 cifras, de sobra para intensidades de imagen)
        tipo = np.float32 if c.cerca.size * nq > 2e7 else np.float64
        Wg = np.zeros((c.cerca.size, nq), dtype=tipo)
        loc_t, loc_w, loc_i = [], [], []
        for fila, k in enumerate(c.cerca):
            w = c.w[k]
            r, tau = abs(w), np.angle(w)
            delta = 1 - r
            # coordenadas relativas a tau, en [-pi, pi)
            rel = lambda t: np.mod(t - tau + np.pi, 2 * np.pi) - np.pi
            a_rel, b_rel = rel(pan[:, 0]), rel(pan[:, 1])
            largo = pan[:, 1] - pan[:, 0]
            cruza = b_rel < a_rel                       # panel cortado en +-pi
            dist = np.where((a_rel <= 0) & (b_rel >= 0) & ~cruza, 0.0,
                            np.minimum(np.abs(a_rel), np.abs(b_rel)))
            reempl = (dist < 2 * largo) & (largo > delta) & ~cruza
            P = (1 - r**2) / np.abs(np.exp(1j * c.tq) - w) ** 2
            peso = c.wq * P / (2 * np.pi)
            if reempl.any():
                a, b = a_rel[reempl].min(), b_rel[reempl].max()
                # se quitan TODOS los nodos globales de la ventana [a, b],
                # incluidos los de paneles pequeños que no se marcaron
                rq = rel(c.tq)
                peso[(rq >= a) & (rq <= b)] = 0.0
                esc = delta * 2.0 ** np.arange(0, 60)
                cortes = [a, b, 0.0, *(-esc[esc < -a]), *esc[esc < b]]
                # graduación hacia los prevértices que caen en la ventana
                for tk in rel(m.theta):
                    if a < tk < b:
                        sep = 0.5 * min(abs(tk), tk - a, b - tk)
                        off = sep * 0.25 ** np.arange(0, 14)
                        cortes += [tk, *(tk + off), *(tk - off)]
                cortes = np.unique(np.array(cortes))
                cortes = cortes[(cortes >= a) & (cortes <= b)]
                cortes = cortes[np.r_[True, np.diff(cortes) > 1e-13]]
                mid = 0.5 * (cortes[1:] + cortes[:-1])
                hlf = 0.5 * (cortes[1:] - cortes[:-1])
                tl = (mid[:, None] + hlf[:, None] * x[None, :]).ravel() + tau
                wtl = (hlf[:, None] * wl[None, :]).ravel()
                Pl = (1 - r**2) / np.abs(np.exp(1j * tl) - w) ** 2
                loc_t.append(np.mod(tl, 2 * np.pi))
                loc_w.append(wtl * Pl / (2 * np.pi))
                loc_i.append(np.full(tl.size, fila))
            Wg[fila] = peso
        c.Wg = Wg
        if loc_t:
            c.loc_t = np.concatenate(loc_t)
            c.loc_w = np.concatenate(loc_w)
            c.loc_i = np.concatenate(loc_i)
            c.loc_frontera, _ = m.psi_circulo(c.loc_t)
        else:
            c.loc_t = np.zeros(0)

    # ------------------------------------------------------------------
    def _dato_frontera(self, imagen, c):
        """g(psi(e^{it_q})), interpolando la imagen en los puntos del polígono."""
        relleno = imagen[tuple(self._ind)]
        return ndi.map_coordinates(relleno, [c.frontera.imag, c.frontera.real],
                                   order=self.orden_interp, mode="nearest")

    @staticmethod
    def coeficientes(G, tq, wq, N):
        """c_n = (1/2pi) integral G(t) e^{-int} dt, n = 0..N, por cuadratura.

        Con nodos equiespaciados y pesos 2pi/M es exactamente la DFT.
        """
        P = wq * G / (2 * np.pi) + 0j
        z = np.exp(-1j * tq)
        cf = np.empty(N + 1, complex)
        for n in range(N + 1):
            cf[n] = P.sum()
            P *= z
        return cf

    @staticmethod
    def serie(cf, w):
        """Re( c_0 + 2 sum_{n>=1} c_n w^n ), por Horner."""
        s = np.zeros(w.shape, complex)
        for n in range(len(cf) - 1, 0, -1):
            s = s * w + cf[n]
        return cf[0].real + 2 * np.real(s * w)

    def _armonico_componente(self, u, c):
        G = self._dato_frontera(u, c)
        cf = self.coeficientes(G, c.tq, c.wq, self.M // 2 - 1)
        if not hasattr(c, "cerca"):
            return self.serie(cf, c.w)
        lejos = np.ones(c.w.size, bool)
        lejos[c.cerca] = False
        out = np.empty(c.w.size)
        out[lejos] = self.serie(cf, c.w[lejos])
        vals = (c.Wg @ G.astype(c.Wg.dtype)).astype(float)
        if c.loc_t.size:
            relleno = u[tuple(self._ind)]
            Gl = ndi.map_coordinates(relleno, [c.loc_frontera.imag,
                                               c.loc_frontera.real],
                                     order=self.orden_interp, mode="nearest")
            vals += np.bincount(c.loc_i, c.loc_w * Gl, minlength=c.cerca.size)
        out[c.cerca] = vals
        return out

    def resolver(self, imagen) -> Resultado:
        """Inpainting armónico por la ruta conforme (etapa por dato)."""
        t0 = time.perf_counter()
        u = np.array(imagen, dtype=float, copy=True)
        for c in self.componentes:
            u[c.filas, c.cols] = self._armonico_componente(u, c)
        if self._respaldo_solver is not None:
            u = np.where(np.logical_or.reduce(self.respaldo),
                         self._respaldo_solver.resolver(u).u, u)
        return Resultado(u=u, iteraciones=1, tiempo=time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def campo_rho(self):
        """rho(z) = lambda |phi'(z)| = lambda / |psi'(phi(z))| en la máscara.

        lambda es la mediana de |psi'| en cada componente, de modo que rho
        vale 1 "en promedio". Fuera de la máscara se copia el valor del píxel
        de la máscara más cercano (solo se usa en las aristas del borde).

        Con este campo, el modelo ESTÁNDAR del disco, regresado a D con phi,
        es exactamente (Proposición de divergencia del capítulo V)
            TV:  div( rho ∇u / sqrt((rho eps)^2 + |∇u|^2) ) = 0,
            PM:  rho^2 u_t = div( c(|∇u|^2/(K rho)^2) ∇u ),
        y se puede resolver en la MISMA malla de píxeles que la ruta directa.
        La diferencia con la ruta directa es entonces pura no invariancia,
        sin error de discretización del disco ni de interpolación.
        """
        rho = np.ones(self.D.shape)
        for c in self.componentes:
            rho[c.filas, c.cols] = np.median(c.factor) / c.factor
        _, ind = ndi.distance_transform_edt(~self.D, return_indices=True)
        return rho[tuple(ind)]

    def grad_log_rho(self):
        """|grad_z log rho| = |psi''/psi'| / |psi'| en phi(z), por la fórmula de SC."""
        g = np.full(self.D.shape, np.nan)
        for c in self.componentes:
            m = c.mapa
            ld = np.abs(np.sum(m.beta / (c.w[:, None] - m.w[None, :]), axis=1))
            g[c.filas, c.cols] = ld / c.factor
        return g

    # ------------------------------------------------------------------
    def resolver_disco(self, imagen, modelo, N_disco=None, **params) -> Resultado:
        """Modelo ESTÁNDAR en el disco y regreso con phi.

        modelo: 'armonico', 'calor', 'perona_malik' o 'tv'.
        Los parámetros con unidades (K, eps, T, dt) se dan en las unidades de
        la imagen y se convierten al disco con el factor conforme típico
        lambda = mediana de |psi'| en la máscara, de modo que |grad| se
        multiplica por lambda y el tiempo por 1/lambda^2. Esa conversión es
        exacta solo si |psi'| es constante, y la diferencia con la ruta
        directa mide justamente la no invariancia.
        """
        t0 = time.perf_counter()
        u = np.array(imagen, dtype=float, copy=True)
        for c in self.componentes:
            lam = np.median(c.factor)
            N = N_disco or int(2 * np.sqrt(c.filas.size / np.pi) * 2) + 8
            h = 2.0 / N
            s = -1 + h * (np.arange(N) + 0.5)
            S, T = np.meshgrid(s, s)
            W = S + 1j * T
            Dd = np.abs(W) < 1
            # dato: G evaluada en el ángulo de cada píxel exterior
            G = self._dato_frontera(u, c)
            ang = np.mod(np.angle(W), 2 * np.pi)
            Gp = np.interp(ang.ravel(), np.r_[c.tq[-1] - 2 * np.pi, c.tq,
                                              c.tq[0] + 2 * np.pi],
                           np.r_[G[-1], G, G[0]]).reshape(W.shape)
            img_d = np.where(Dd, 0.0, Gp)
            p = dict(params)
            if "K" in p: p["K"] = p["K"] * lam
            if "eps" in p: p["eps"] = p["eps"] * lam
            if "T" in p: p["T"] = p["T"] / lam**2
            if "dt" in p: p["dt"] = p["dt"] / lam**2
            if modelo == "armonico":
                v = directa.armonico(img_d, Dd, h).u
            elif modelo == "calor":
                v = directa.calor(img_d, Dd, h=h, **p).u
            elif modelo == "perona_malik":
                v = directa.perona_malik(img_d, Dd, h=h, **p).u
            elif modelo == "tv":
                v = directa.variacion_total_estacionaria(img_d, Dd, h=h, **p).u
            else:
                raise ValueError(modelo)
            fil = (c.w.imag + 1) / h - 0.5
            col = (c.w.real + 1) / h - 0.5
            u[c.filas, c.cols] = ndi.map_coordinates(v, [fil, col], order=1,
                                                     mode="nearest")
        return Resultado(u=u, tiempo=time.perf_counter() - t0)
