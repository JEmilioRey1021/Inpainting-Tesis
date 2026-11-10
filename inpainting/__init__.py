"""Inpainting por EDP: ruta directa y ruta conforme (tesis de Emilio Reyes, UVG)."""

from .discretizacion import Malla
from .directa import (
    Resultado,
    SolverArmonico,
    armonico,
    calor,
    perona_malik,
    variacion_total,
    variacion_total_estacionaria,
)
from .sc import MapeoSC
from .conforme import RutaConforme
