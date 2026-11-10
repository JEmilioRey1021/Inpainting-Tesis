"""
Mapeo de Schwarz–Christoffel del disco unitario sobre un polígono.

Fórmula (sección IV.C de la tesis)
----------------------------------
Si el polígono tiene vértices z_0, ..., z_{n-1} en sentido antihorario y
ángulos interiores alpha_k * pi, el mapeo conforme del disco sobre su
interior es

    psi(w) = A + C * integral_0^w  prod_k (1 - zeta/w_k)^(beta_k) d zeta,

con beta_k = alpha_k - 1 y prevértices w_k = e^{i theta_k} en el círculo.
A, C y los theta_k son las incógnitas (el "problema de parámetros").

Normalización
-------------
Se fija psi(0) = z_c, un punto interior elegido por el usuario (el "centro
conforme"), y el último prevértice w_{n-1} = 1. Eso deja n-1 incógnitas
reales (los n-1 espaciamientos libres entre prevértices) y n-1 ecuaciones,
que son
    - n-3 cocientes de longitudes de lados,
    - 2 ecuaciones reales psi(0) = z_c.
Es equivalente a fijar tres prevértices (Observación sobre el problema de
parámetros del capítulo IV).

Cuadratura
----------
El integrando tiene singularidades integrables |zeta - w_k|^{beta_k} en los
prevértices. Se usa la técnica de Driscoll y Trefethen (2002):
    - si el camino empieza en un prevértice, el primer tramo se integra con
      Gauss–Jacobi, que absorbe exactamente la singularidad,
    - el resto del camino se recorre por tramos, cada uno de longitud menor
      que la mitad de la distancia de su punto inicial a la singularidad más
      cercana, y en cada tramo se usa Gauss–Legendre.
Así cada tramo ve un integrando analítico en un disco que lo contiene, y la
cuadratura converge geométricamente.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_jacobi, roots_legendre
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
#  Geometría del polígono
# ----------------------------------------------------------------------
def area_con_signo(z):
    return 0.5 * np.imag(np.sum(np.conj(z) * np.roll(z, -1)))


def angulos_interiores(z):
    """alpha_k tal que el ángulo interior en z_k es alpha_k * pi (antihorario)."""
    d_in = z - np.roll(z, 1)
    d_out = np.roll(z, -1) - z
    giro = np.angle(d_out / d_in)          # ángulo de giro en (-pi, pi]
    return 1.0 - giro / np.pi


def limpiar_poligono(z, tol_ang=1e-3):
    """Quita vértices repetidos y colineales, y orienta en sentido antihorario."""
    z = np.asarray(z, complex)
    z = z[np.abs(z - np.roll(z, -1)) > 1e-12]
    if area_con_signo(z) < 0:
        z = z[::-1]
    while True:
        a = angulos_interiores(z)
        colin = np.abs(a - 1) < tol_ang
        if not colin.any() or len(z) <= 3:
            break
        z = z[~colin]
    return z


# ----------------------------------------------------------------------
class MapeoSC:
    """psi: disco -> polígono, y su inversa phi, para un polígono dado."""

    def __init__(self, vertices, centro, n_nodos=16, tol=1e-10, verbose=False):
        z = limpiar_poligono(vertices)
        self.z = z
        self.n = n = len(z)
        if n < 3:
            raise ValueError("El polígono necesita al menos 3 vértices.")
        self.alpha = angulos_interiores(z)
        self.beta = self.alpha - 1.0
        if not np.isclose(self.alpha.sum(), n - 2, atol=1e-8):
            raise ValueError("Los ángulos no suman (n-2)pi. ¿Polígono simple?")
        if np.any(self.alpha <= 0) or np.any(self.alpha >= 2):
            raise ValueError("Hay ángulos interiores degenerados (0 o 2pi).")
        self.zc = complex(centro)
        self.escala = np.max(np.abs(z - z.mean()))
        self.nn = n_nodos
        self.xl, self.wl = roots_legendre(n_nodos)
        # nodos de Gauss–Jacobi para cada exponente distinto
        self._jac = {}
        for b in self.beta:
            self._nodos_jacobi(b)
        self._resolver_parametros(tol, verbose)

    # ------------------------------------------------------------------
    def _nodos_jacobi(self, b):
        key = round(float(b), 14)
        if key not in self._jac:
            # peso (1-x)^0 (1+x)^b sobre [-1, 1]
            self._jac[key] = roots_jacobi(self.nn, 0.0, b)
        return self._jac[key]

    # ------------------------------------------------------------------
    #  Integrando
    # ------------------------------------------------------------------
    def _f(self, zeta, excluir=None, w=None):
        """prod_k (1 - zeta/w_k)^beta_k, omitiendo el factor k = excluir.

        zeta: arreglo de forma (P, Q). excluir: arreglo de forma (P,) de
        índices, o None.
        """
        w = self.w if w is None else w
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log1p(-zeta[..., None] / w)        # (P, Q, n)
        B = np.broadcast_to(self.beta, L.shape).copy()
        if excluir is not None:
            ex = np.asarray(excluir)
            B[np.arange(len(ex)), :, ex] = 0.0
        with np.errstate(invalid="ignore", over="ignore"):
            return np.exp(np.sum(B * L, axis=-1))

    def _dist_sing(self, p, w):
        """Distancia de cada punto p a la singularidad más cercana."""
        return np.min(np.abs(p[:, None] - w[None, :]), axis=1)

    # ------------------------------------------------------------------
    #  Cuadraturas vectorizadas
    # ------------------------------------------------------------------
    def _marcha(self, a, b, w=None, tramos_max=200):
        """integral de f de a a b (puntos regulares), por tramos Legendre."""
        w = self.w if w is None else w
        a = np.asarray(a, complex).copy()
        b = np.asarray(b, complex)
        total = np.zeros(a.shape, complex)
        activo = np.abs(b - a) > 0
        for _ in range(tramos_max):
            if not activo.any():
                break
            idx = np.nonzero(activo)[0]
            c, d = a[idx], b[idx]
            largo = np.abs(d - c)
            paso = np.minimum(largo, 0.5 * self._dist_sing(c, w))
            fin = c + (d - c) * np.divide(paso, largo, out=np.zeros_like(paso),
                                          where=largo > 0)
            mid, half = 0.5 * (c + fin), 0.5 * (fin - c)
            zeta = mid[:, None] + half[:, None] * self.xl[None, :]
            total[idx] += half * (self._f_trozos(zeta, None, w) @ self.wl)
            a[idx] = fin
            activo[idx] = np.abs(b[idx] - fin) > 1e-15 * (1 + np.abs(b[idx]))
        return total

    def _desde_prevertice(self, k, b, w=None):
        """integral de f desde el prevértice w_k hasta los puntos b."""
        w = self.w if w is None else w
        k = np.broadcast_to(np.asarray(k), np.shape(b)).astype(int)
        b = np.asarray(b, complex)
        wk = w[k]
        # distancia de w_k a los otros prevértices
        D = np.abs(w[:, None] - w[None, :]) + np.diag(np.full(len(w), np.inf))
        dk = D.min(axis=1)[k]
        largo = np.abs(b - wk)
        s = np.minimum(largo, 0.5 * dk)
        frac = np.divide(s, largo, out=np.zeros_like(s), where=largo > 0)
        fin = wk + (b - wk) * frac
        total = np.zeros(b.shape, complex)
        for kk in np.unique(k):
            sel = k == kk
            beta = self.beta[kk]
            x, wj = self._nodos_jacobi(beta)
            half = 0.5 * (fin[sel] - wk[sel])
            zeta = wk[sel][:, None] + half[:, None] * (1 + x[None, :])
            g = self._f_trozos(zeta, np.full(sel.sum(), kk), w)
            p = 1 - fin[sel] / w[kk]                     # (1 - zeta/w_k) = p (1+x)/2
            with np.errstate(all="ignore"):
                val = half * p**beta * 2.0**(-beta) * (g @ wj)
            total[sel] = np.where(half != 0, val, 0)
        total += self._marcha(fin, b, w)
        return total

    def _f_trozos(self, zeta, excluir, w, trozo=4000):
        out = np.empty(zeta.shape, complex)
        for i in range(0, zeta.shape[0], trozo):
            ex = None if excluir is None else excluir[i:i + trozo]
            out[i:i + trozo] = self._f(zeta[i:i + trozo], ex, w)
        return out

    def _lados(self, w):
        """integrales de f sobre las cuerdas w_k -> w_{k+1}."""
        n = self.n
        k = np.arange(n)
        k1 = (k + 1) % n
        mid = 0.5 * (w[k] + w[k1])
        # el punto medio de la cuerda puede estar muy cerca del círculo si los
        # prevértices están juntos, pero sigue siendo interior
        return (self._desde_prevertice(k, mid, w)
                - self._desde_prevertice(k1, mid, w))

    # ------------------------------------------------------------------
    #  Problema de parámetros
    # ------------------------------------------------------------------
    @staticmethod
    def _theta(y):
        e = np.exp(np.concatenate([y, [0.0]]) - np.max(np.concatenate([y, [0.0]])))
        d = 2 * np.pi * e / e.sum()
        return np.cumsum(d)          # theta_{n-1} = 2 pi, es decir, w_{n-1} = 1

    def _residuos(self, y):
        n = self.n
        w = np.exp(1j * self._theta(y))
        I = self._lados(w)
        lz = np.abs(np.roll(self.z, -1) - self.z)
        r = [np.log(np.abs(I[k]) / np.abs(I[0])) - np.log(lz[k] / lz[0])
             for k in range(1, n - 2)]
        C = (self.z[1] - self.z[0]) / I[0]
        A = self.z[0] + C * self._desde_prevertice(0, np.array([0j]), w)[0]
        e = (A - self.zc) / self.escala
        return np.array(r + [e.real, e.imag])

    def _residuos_seguros(self, y):
        """Residuos, con un valor grande si la cuadratura falla (NaN o inf)."""
        with np.errstate(all="ignore"):
            r = self._residuos(y)
        return np.where(np.isfinite(r), r, 1e3)

    def _resolver_parametros(self, tol, verbose):
        n = self.n
        # Dos valores iniciales: los ángulos con que se ven los vértices desde
        # el centro (bueno si la región es estrellada respecto de él) y
        # espaciamientos proporcionales a las longitudes de los lados. Con cada
        # uno se prueba Levenberg–Marquardt y, si no converge, región de
        # confianza (trf). Se probó que la convergencia de un solo intento
        # depende de la versión de SciPy, y estos reintentos la hacen robusta.
        iniciales = []
        ang = np.angle(self.z - self.zc)
        th = np.mod(ang - ang[-1], 2 * np.pi)
        th[-1] = 2 * np.pi
        d = np.diff(np.concatenate([[0.0], th]))
        if np.all(d > 0):
            iniciales.append(d)
        lz = np.abs(np.roll(self.z, -1) - self.z)
        iniciales.append(2 * np.pi * lz / lz.sum())
        mejor = None
        for d in iniciales:
            d = np.maximum(d, 1e-6)
            y0 = np.log(d[:-1] / d[-1])
            for metodo in ("lm", "trf"):
                sol = least_squares(self._residuos_seguros, y0, method=metodo,
                                    xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                    max_nfev=200 * n)
                res = np.max(np.abs(sol.fun))
                if mejor is None or res < mejor[0]:
                    mejor = (res, sol)
                if res <= 1e-10:
                    break
            if mejor[0] <= 1e-10:
                break
        self.residuo_parametros, sol = mejor
        if verbose:
            print(f"   SC: n = {n}, evaluaciones = {sol.nfev}, "
                  f"residuo = {self.residuo_parametros:.1e}")
        if self.residuo_parametros > 1e-6:
            raise RuntimeError("El problema de parámetros no convergió "
                               f"(residuo {self.residuo_parametros:.1e}).")
        self.theta = self._theta(sol.x)
        self.w = np.exp(1j * self.theta)
        I = self._lados(self.w)
        self.C = (self.z[1] - self.z[0]) / I[0]
        self.A = self.z[0] + self.C * self._desde_prevertice(0, np.array([0j]))[0]
        self.separacion_min = np.min(np.diff(np.concatenate(
            [self.theta, [self.theta[0] + 2 * np.pi]])))
        # error de cierre: dónde caen los prevértices
        self.error_vertices = np.max(np.abs(self.psi_frontera_vertices() - self.z))

    # ------------------------------------------------------------------
    #  Evaluación
    # ------------------------------------------------------------------
    def psi_frontera_vertices(self):
        """psi(w_k) para cada prevértice. Debe coincidir con z_k."""
        k = np.arange(self.n)
        return self.A - self.C * self._desde_prevertice(k, np.zeros(self.n, complex))

    def dpsi(self, w):
        w = np.asarray(w, complex)
        return self.C * self._f_trozos(w.reshape(-1, 1), None, self.w).reshape(w.shape)

    def psi(self, w):
        """psi en puntos interiores, integrando desde el origen."""
        w = np.asarray(w, complex)
        return self.A + self.C * self._marcha(np.zeros(w.size, complex),
                                              w.ravel()).reshape(w.shape)

    def psi_circulo(self, t):
        """psi(e^{it}) para t en [0, 2pi). Devuelve los puntos y el lado.

        Cada punto se integra desde el prevértice más cercano y, como debe
        caer sobre un lado, se proyecta sobre él para quitar el error de
        redondeo.
        """
        t = np.mod(np.asarray(t, float), 2 * np.pi)
        th = np.mod(self.theta, 2 * np.pi)
        orden = np.argsort(th)
        ths = th[orden]
        # lado: entre el prevértice k y el k+1 (en el orden del polígono)
        pos = np.searchsorted(ths, t, side="right") - 1
        k = orden[pos % self.n]
        k1 = (k + 1) % self.n
        dk = np.abs(np.angle(np.exp(1j * (t - th[k]))))
        dk1 = np.abs(np.angle(np.exp(1j * (t - th[k1]))))
        desde = np.where(dk <= dk1, k, k1)
        b = np.exp(1j * t)
        zk = self.z[desde]
        p = zk + self.C * self._desde_prevertice(desde, b)
        # a distancia de redondeo de un prevértice, el punto es el vértice
        p = np.where(np.abs(b - self.w[desde]) < 1e-13, zk, p)
        # proyección sobre el lado k -> k1
        a0, a1 = self.z[k], self.z[k1]
        s = np.real((p - a0) * np.conj(a1 - a0)) / np.abs(a1 - a0)**2
        s = np.clip(s, 0, 1)
        return a0 + s * (a1 - a0), k

    # ------------------------------------------------------------------
    def cuadratura_circulo(self, N=1024, paneles=12, nodos=8, razon=0.25):
        """Nodos y pesos en t para integrar funciones G(t) = g(psi(e^{it})).

        Cada arco entre prevértices consecutivos se parte por la mitad, y cada
        mitad en paneles que se achican geométricamente hacia el prevértice
        (razón `razon`). Los paneles se subdividen para que cada uno tenga
        longitud menor que nodos/N, y en cada uno se usa Gauss–Legendre. Así se integran
        con precisión las singularidades |t - t_k|^alpha_k que G tiene en los
        prevértices, que con nodos equiespaciados producen aliasing.
        """
        th = np.sort(np.mod(self.theta, 2 * np.pi))
        ext = np.r_[th, th[0] + 2 * np.pi]
        x, wl = roots_legendre(nodos)
        # puntos de corte relativos en [0, 1/2], graduados hacia 0
        cortes = np.r_[0.0, 0.5 * razon ** np.arange(paneles - 1, -1, -1)]
        T, W, PAN = [], [], []
        for a, b in zip(ext[:-1], ext[1:]):
            L = b - a
            for lado in (0, 1):
                bordes = a + L * cortes if lado == 0 else b - L * cortes[::-1]
                for p0, p1 in zip(bordes[:-1], bordes[1:]):
                    # además, cada panel se subdivide para resolver la
                    # oscilación de e^{-int} hasta n = N
                    m_sub = max(1, int(np.ceil((p1 - p0) * N / nodos)))
                    sub = np.linspace(p0, p1, m_sub + 1)
                    for q0, q1 in zip(sub[:-1], sub[1:]):
                        m, hlf = 0.5 * (q0 + q1), 0.5 * (q1 - q0)
                        T.append(m + hlf * x); W.append(hlf * wl)
                        PAN.append((q0, q1))
        T, W = np.concatenate(T), np.concatenate(W)
        ipan = np.repeat(np.arange(len(PAN)), nodos)
        orden = np.argsort(np.mod(T, 2 * np.pi))
        # paneles (en t sin reducir módulo 2pi) y el panel de cada nodo, por si
        # se quiere refinar localmente (evaluación cerca del círculo)
        self.paneles = np.array(PAN)
        self.panel_de_nodo = ipan[orden]
        return np.mod(T, 2 * np.pi)[orden], W[orden]

    # ------------------------------------------------------------------
    def phi(self, zs, iter_max=30, tol=1e-11):
        """Inversa por Newton, con valor inicial de una tabla de psi."""
        zs = np.asarray(zs, complex).ravel()
        if not hasattr(self, "_arbol"):
            self._construir_tabla()
        _, j = self._arbol.query(np.c_[zs.real, zs.imag])
        w = self._tabla_w[j].copy()
        pw = self._tabla_z[j].copy()
        for it in range(iter_max):
            e = pw - zs
            listo = np.abs(e) < tol * self.escala
            if listo.all():
                break
            idx = np.nonzero(~listo)[0]
            paso = -e[idx] / self.dpsi(w[idx])
            # amortiguar para no salir del disco
            for _ in range(60):
                fuera = np.abs(w[idx] + paso) >= 1
                if not fuera.any():
                    break
                paso[fuera] *= 0.5
            nuevo = w[idx] + paso
            pw[idx] = pw[idx] + self.C * self._marcha(w[idx], nuevo)
            w[idx] = nuevo
        self.residuo_inversa = np.max(np.abs(pw - zs)) / self.escala
        return w

    def _construir_tabla(self, n_r=40, n_t=256, n_off=25):
        """psi en una malla polar refinada cerca del círculo y de los prevértices.

        Se integra a lo largo de cada rayo, de un radio al siguiente, y se
        acumula, de modo que cada integral es corta.
        """
        r = np.concatenate([np.linspace(0, 0.9, 15, endpoint=False),
                            1 - np.geomspace(0.1, 1e-7, n_r)])
        t = list(np.linspace(0, 2 * np.pi, n_t, endpoint=False))
        gaps = np.diff(np.concatenate([self.theta, [self.theta[0] + 2 * np.pi]]))
        for k, th in enumerate(self.theta):
            g = min(gaps[k], gaps[k - 1])
            off = np.geomspace(1e-9, 0.5, n_off) * g
            t += list(th + off) + list(th - off)
        t = np.mod(np.array(t), 2 * np.pi)
        E = np.exp(1j * t)
        a = (r[:-1, None] * E[None, :]).ravel()
        b = (r[1:, None] * E[None, :]).ravel()
        I = self._marcha(a, b).reshape(len(r) - 1, len(t))
        Z = self.A + self.C * np.vstack([np.zeros((1, len(t))),
                                         np.cumsum(I, axis=0)])
        W = r[:, None] * E[None, :]
        W, Z = W[1:].ravel(), Z[1:].ravel()
        W = np.r_[0j, W]; Z = np.r_[self.A, Z]
        self._tabla_w, self._tabla_z = W, Z
        self._arbol = cKDTree(np.c_[Z.real, Z.imag])
