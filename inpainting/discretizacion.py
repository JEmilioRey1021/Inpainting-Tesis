"""
Discretización sobre la malla de píxeles.

Idea central
------------
Todos los modelos de la tesis (armónico, calor, Perona–Malik, TV) se escriben
en forma de divergencia,

    u_t = div( b · grad u ),

con b = 1 (armónico y calor), b = c(|grad u|^2) (Perona–Malik) o
b = 1/|grad u|_eps (variación total). Por eso basta UNA rutina que, dada una
"conductancia" b en cada arista entre píxeles vecinos, arme la matriz del
operador. Es la versión discreta de la Proposición de operadores en forma de
divergencia del capítulo V.

Convenciones
------------
- La imagen es un arreglo 2D `u` de tamaño (n_filas, n_columnas).
- La máscara `D` es un arreglo booleano del mismo tamaño. True = píxel dañado
  (incógnita), False = píxel conocido (dato).
- El espaciado de la malla es h. Para imágenes se usa h = 1. Para las pruebas
  de convergencia se usa h = longitud del dominio / número de píxeles.
- En el borde de la imagen se usa reflexión (condición de Neumann), aunque en
  la práctica las máscaras no tocan el borde.

Discretización
--------------
Para cada píxel p en la máscara y cada uno de sus 4 vecinos q,

    div(b grad u)(p)  ≈  (1/h^2) Σ_q  b_pq (u_q - u_p),

donde b_pq es la conductancia en la arista entre p y q. Con b = 1 es el
laplaciano de 5 puntos. Si q es conocido, u_q pasa al lado derecho, y así se
impone la condición de Dirichlet del inpainting.

Esto da el sistema   A u_D = rhs,   donde
    (A u)_p = (1/h^2) Σ_q b_pq (u_p - u_q)       (solo q incógnitos)
    rhs_p   = (1/h^2) Σ_q b_pq u_q               (solo q conocidos)
de modo que  div(b grad u) = rhs - A u  en los píxeles de la máscara.
A es simétrica, y si b > 0 y cada componente de D toca algún píxel conocido,
es definida positiva.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Malla:
    """Indexación de las incógnitas de una máscara."""

    def __init__(self, mascara: np.ndarray, h: float = 1.0):
        D = np.asarray(mascara, dtype=bool)
        if D.ndim != 2:
            raise ValueError("La máscara debe ser un arreglo 2D.")
        if not D.any():
            raise ValueError("La máscara está vacía.")
        if D.all():
            raise ValueError("La máscara cubre toda la imagen, no hay datos.")
        self.D = D
        self.h = float(h)
        self.forma = D.shape
        # índice lineal de cada incógnita, -1 para píxeles conocidos
        self.indice = -np.ones(D.shape, dtype=np.int64)
        self.filas, self.cols = np.nonzero(D)
        self.n = self.filas.size
        self.indice[self.filas, self.cols] = np.arange(self.n)
        self._aristas = self._construir_aristas()

    def _construir_aristas(self):
        """Aristas (p, q) con al menos un extremo en la máscara.

        Se devuelven por separado las horizontales y las verticales, como
        pares de coordenadas planas en la imagen.
        """
        nf, nc = self.forma
        D = self.D
        # horizontales: (i, j) -- (i, j+1)
        hi, hj = np.nonzero(D[:, :-1] | D[:, 1:])
        # verticales: (i, j) -- (i+1, j)
        vi, vj = np.nonzero(D[:-1, :] | D[1:, :])
        return (hi, hj), (vi, vj)

    # ------------------------------------------------------------------
    def a_vector(self, u: np.ndarray) -> np.ndarray:
        """Valores de u en las incógnitas."""
        return u[self.filas, self.cols].copy()

    def a_imagen(self, x: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Inserta las incógnitas x en una copia de la imagen base."""
        u = np.array(base, dtype=float, copy=True)
        u[self.filas, self.cols] = x
        return u

    # ------------------------------------------------------------------
    def ensamblar(self, u: np.ndarray, cond_h=None, cond_v=None):
        """Arma A y rhs para div(b grad u) con conductancias dadas.

        Parámetros
        ----------
        u : imagen completa (se usan solo los valores conocidos).
        cond_h, cond_v : conductancias en las aristas horizontales y
            verticales, en el mismo orden que `self._aristas`. Si son None,
            se toma b = 1 (laplaciano).

        Devuelve
        --------
        A : matriz dispersa CSC de tamaño n x n, simétrica.
        rhs : vector de tamaño n.
        """
        (hi, hj), (vi, vj) = self._aristas
        nf, nc = self.forma
        if cond_h is None:
            cond_h = np.ones(hi.size)
        if cond_v is None:
            cond_v = np.ones(vi.size)

        p = np.concatenate([self.indice[hi, hj], self.indice[vi, vj]])
        q = np.concatenate([self.indice[hi, hj + 1], self.indice[vi + 1, vj]])
        up = np.concatenate([u[hi, hj], u[vi, vj]])
        uq = np.concatenate([u[hi, hj + 1], u[vi + 1, vj]])
        b = np.concatenate([cond_h, cond_v]) / self.h**2

        filas, cols, vals = [], [], []
        rhs = np.zeros(self.n)

        # p incógnito: contribuye a la diagonal de p
        mp = p >= 0
        filas.append(p[mp]); cols.append(p[mp]); vals.append(b[mp])
        # q incógnito: contribuye a la diagonal de q
        mq = q >= 0
        filas.append(q[mq]); cols.append(q[mq]); vals.append(b[mq])
        # ambos incógnitos: términos fuera de la diagonal
        mpq = mp & mq
        filas.append(p[mpq]); cols.append(q[mpq]); vals.append(-b[mpq])
        filas.append(q[mpq]); cols.append(p[mpq]); vals.append(-b[mpq])
        # uno conocido: pasa al lado derecho
        solo_p = mp & ~mq
        np.add.at(rhs, p[solo_p], b[solo_p] * uq[solo_p])
        solo_q = mq & ~mp
        np.add.at(rhs, q[solo_q], b[solo_q] * up[solo_q])

        A = sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
            shape=(self.n, self.n),
        )
        return A, rhs

    # ------------------------------------------------------------------
    def gradiente_en_aristas(self, u: np.ndarray):
        """|grad u|^2 en el punto medio de cada arista.

        En una arista horizontal entre (i,j) y (i,j+1), la derivada en x es la
        diferencia a través de la arista, y la derivada en y es el promedio de
        las diferencias centradas en los dos píxeles. Lo análogo para las
        verticales. Se usa reflexión en el borde de la imagen.
        """
        (hi, hj), (vi, vj) = self._aristas
        h = self.h
        U = np.pad(u, 1, mode="edge")  # U[i+1, j+1] = u[i, j]

        # horizontales
        ux = (U[hi + 1, hj + 2] - U[hi + 1, hj + 1]) / h
        uy = (U[hi + 2, hj + 1] - U[hi, hj + 1]
              + U[hi + 2, hj + 2] - U[hi, hj + 2]) / (4 * h)
        g2_h = ux**2 + uy**2

        # verticales
        uy = (U[vi + 2, vj + 1] - U[vi + 1, vj + 1]) / h
        ux = (U[vi + 1, vj + 2] - U[vi + 1, vj]
              + U[vi + 2, vj + 2] - U[vi + 2, vj]) / (4 * h)
        g2_v = ux**2 + uy**2
        return g2_h, g2_v

    # ------------------------------------------------------------------
    def promedio_en_aristas(self, campo: np.ndarray):
        """Promedio de un campo de la imagen en los dos extremos de cada arista."""
        (hi, hj), (vi, vj) = self._aristas
        return (0.5 * (campo[hi, hj] + campo[hi, hj + 1]),
                0.5 * (campo[vi, vj] + campo[vi + 1, vj]))
