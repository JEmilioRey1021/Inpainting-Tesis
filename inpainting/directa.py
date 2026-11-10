"""
Ruta directa. Los cuatro modelos se resuelven sobre la máscara de píxeles.

    armonico        Δu = 0 en D, u = dato fuera de D.          (sistema lineal)
    calor           u_t = Δu                                    (Euler implícito)
    perona_malik    u_t = div( c(|∇u|^2) ∇u ),  c(s)=1/(1+s/K^2) (semi-implícito)
    variacion_total u_t = div( ∇u / |∇u|_eps )                  (semi-implícito)
                    y su estado estacionario por difusividad retardada.

"Semi-implícito" significa que en cada paso la conductancia se calcula con la
solución del paso anterior y luego se resuelve un sistema lineal,

    (I + dt A(u^n)) x^{n+1} = x^n + dt rhs(u^n).

Este esquema es estable para todo dt porque la matriz es definida positiva,
y por eso permite pasos grandes, a diferencia del esquema explícito, que exige
dt <= h^2/4 (y en TV, dt <= eps h^2/4, que es muy pequeño).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .discretizacion import Malla


@dataclass
class Resultado:
    u: np.ndarray                 # imagen reconstruida completa
    iteraciones: int = 0
    tiempo: float = 0.0           # segundos de cómputo
    historia: list = field(default_factory=list)  # cambio relativo por paso


def _valor_inicial(malla: Malla, imagen: np.ndarray) -> np.ndarray:
    """Relleno inicial de la máscara con el promedio del dato en su borde."""
    A, rhs = malla.ensamblar(imagen)
    # La suma de cada fila de A es (número de vecinos conocidos)/h^2, y rhs
    # es la suma de esos vecinos entre h^2, así que el cociente es su promedio.
    peso = np.asarray(A.sum(axis=1)).ravel()
    m = peso > 0
    media = rhs[m].sum() / peso[m].sum()
    return np.full(malla.n, media)


# ======================================================================
#  Modelo armónico
# ======================================================================
class SolverArmonico:
    """Inpainting armónico con la matriz factorizada una sola vez.

    La matriz A depende solo de la máscara, así que se factoriza en el
    constructor (etapa por máscara) y cada llamada a `resolver` solo arma el
    lado derecho y hace las dos sustituciones triangulares (etapa por dato).
    Es la versión de la ruta directa del argumento de amortización del
    capítulo V.
    """

    def __init__(self, mascara: np.ndarray, h: float = 1.0):
        t0 = time.perf_counter()
        self.malla = Malla(mascara, h)
        A, _ = self.malla.ensamblar(np.zeros(self.malla.forma))
        self.A = A
        self.lu = spla.splu(A, permc_spec="COLAMD")
        self.tiempo_factorizacion = time.perf_counter() - t0

    def resolver(self, imagen: np.ndarray) -> Resultado:
        t0 = time.perf_counter()
        _, rhs = self.malla.ensamblar(imagen)
        x = self.lu.solve(rhs)
        u = self.malla.a_imagen(x, imagen)
        return Resultado(u=u, iteraciones=1, tiempo=time.perf_counter() - t0)


def armonico(imagen, mascara, h=1.0) -> Resultado:
    """Atajo para un solo dato. El tiempo incluye la factorización."""
    s = SolverArmonico(mascara, h)
    r = s.resolver(imagen)
    r.tiempo += s.tiempo_factorizacion
    return r


# ======================================================================
#  Esquema semi-implícito general
# ======================================================================
def _evolucion(imagen, mascara, conductancia, dt, pasos, h=1.0, tol=0.0,
               x0=None, conductancia_fija=False, masa=None) -> Resultado:
    """Integra m u_t = div(b ∇u) con el esquema semi-implícito.

    masa: imagen con el coeficiente m(z) (por defecto 1). Con m = rho^2 se
    obtiene la evolución del modelo estándar del disco vista en D.

    conductancia(malla, u) -> (cond_h, cond_v), o None para b = 1.
    Se detiene después de `pasos` pasos o cuando el cambio relativo
    ||x^{n+1}-x^n|| / ||x^n|| es menor que tol.
    """
    t0 = time.perf_counter()
    malla = Malla(mascara, h)
    u = np.array(imagen, dtype=float, copy=True)
    x = _valor_inicial(malla, u) if x0 is None else np.asarray(x0, float).copy()
    u = malla.a_imagen(x, u)
    m = np.ones(malla.n) if masa is None else malla.a_vector(masa)
    I = sp.diags(m, format="csc")
    historia = []
    lu = None
    for k in range(pasos):
        if lu is None or not conductancia_fija:
            ch, cv = conductancia(malla, u) if conductancia else (None, None)
            A, rhs = malla.ensamblar(u, ch, cv)
            lu = spla.splu((I + dt * A).tocsc(), permc_spec="COLAMD")
        x_nuevo = lu.solve(m * x + dt * rhs)
        cambio = np.linalg.norm(x_nuevo - x) / max(np.linalg.norm(x), 1e-300)
        historia.append(cambio)
        x = x_nuevo
        u = malla.a_imagen(x, u)
        if cambio < tol:
            break
    return Resultado(u=u, iteraciones=k + 1,
                     tiempo=time.perf_counter() - t0, historia=historia)


# ======================================================================
#  Calor
# ======================================================================
def calor(imagen, mascara, T, dt, h=1.0, x0=None) -> Resultado:
    """u_t = Δu hasta el tiempo T, con Euler implícito.

    La matriz no cambia, así que se factoriza una sola vez. Cuando T crece,
    la solución converge a la armónica.
    """
    pasos = max(1, int(round(T / dt)))
    return _evolucion(imagen, mascara, None, dt, pasos, h, x0=x0,
                      conductancia_fija=True)


# ======================================================================
#  Perona–Malik
# ======================================================================
def conductancia_pm(K, rho=None):
    """c(|∇u|^2) = 1/(1 + |∇u|^2/K^2), ecuación (pm-c) del capítulo IV.

    Con un campo rho, el umbral es K rho(z). Es el modelo estándar del disco
    visto en D (Proposición pm-trasplantado leída al revés).
    """
    def cond(malla, u):
        g2h, g2v = malla.gradiente_en_aristas(u)
        if rho is None:
            Kh = Kv = K
        else:
            rh, rv = malla.promedio_en_aristas(rho)
            Kh, Kv = K * rh, K * rv
        return 1.0 / (1.0 + g2h / Kh**2), 1.0 / (1.0 + g2v / Kv**2)
    return cond


def perona_malik(imagen, mascara, K, T, dt, h=1.0, x0=None, rho=None) -> Resultado:
    """u_t = div(c ∇u). Con rho, rho^2 u_t = div(c(|∇u|^2/(K rho)^2) ∇u)."""
    pasos = max(1, int(round(T / dt)))
    return _evolucion(imagen, mascara, conductancia_pm(K, rho), dt, pasos, h,
                      x0=x0, masa=None if rho is None else rho**2)


# ======================================================================
#  Variación total
# ======================================================================
def conductancia_tv(eps, rho=None):
    """1/|∇u|_eps con |∇u|_eps = sqrt(eps^2 + |∇u|^2).

    Con un campo rho, rho/sqrt((rho eps)^2 + |∇u|^2), que es la variación total
    ponderada por rho (la TV estándar del disco vista en D).
    """
    def cond(malla, u):
        g2h, g2v = malla.gradiente_en_aristas(u)
        if rho is None:
            return 1.0 / np.sqrt(eps**2 + g2h), 1.0 / np.sqrt(eps**2 + g2v)
        rh, rv = malla.promedio_en_aristas(rho)
        return (rh / np.sqrt((rh * eps)**2 + g2h),
                rv / np.sqrt((rv * eps)**2 + g2v))
    return cond


def variacion_total(imagen, mascara, eps, T, dt, h=1.0, x0=None) -> Resultado:
    """Flujo u_t = div(∇u/|∇u|_eps) hasta el tiempo T."""
    pasos = max(1, int(round(T / dt)))
    return _evolucion(imagen, mascara, conductancia_tv(eps), dt, pasos, h, x0=x0)


def variacion_total_estacionaria(imagen, mascara, eps, h=1.0, tol=1e-6,
                                 max_iter=500, x0=None, rho=None) -> Resultado:
    """Minimizador de la TV regularizada con el dato fijo fuera de D.

    Método de difusividad retardada (Vogel y Oman), que consiste en resolver
    A(u^k) u^{k+1} = rhs(u^k) hasta que el cambio relativo sea menor que tol.
    Es el esquema semi-implícito con dt = infinito.
    """
    t0 = time.perf_counter()
    malla = Malla(mascara, h)
    cond = conductancia_tv(eps, rho)
    u = np.array(imagen, dtype=float, copy=True)
    x = _valor_inicial(malla, u) if x0 is None else np.asarray(x0, float).copy()
    u = malla.a_imagen(x, u)
    historia = []
    for k in range(max_iter):
        A, rhs = malla.ensamblar(u, *cond(malla, u))
        x_nuevo = spla.spsolve(A.tocsc(), rhs)
        cambio = np.linalg.norm(x_nuevo - x) / max(np.linalg.norm(x), 1e-300)
        historia.append(cambio)
        x = x_nuevo
        u = malla.a_imagen(x, u)
        if cambio < tol:
            break
    return Resultado(u=u, iteraciones=k + 1,
                     tiempo=time.perf_counter() - t0, historia=historia)
