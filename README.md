# Código de la tesis: inpainting por EDP, ruta directa frente a ruta conforme

## Estructura

```
inpainting/
  discretizacion.py   Malla de píxeles y ensamblaje de div(b ∇u) para cualquier conductancia b
  directa.py          Ruta directa: armónico, calor, Perona–Malik, variación total
  sc.py               Mapeo de Schwarz–Christoffel del disco al polígono (psi, phi, cuadraturas)
  conforme.py         Ruta conforme: polígono, mapeo, dato en el círculo, disco, regreso
verificacion/
  verificar_directa.py    7 pruebas de la ruta directa
  verificar_conforme.py   5 pruebas de la ruta conforme
experimentos/
  demo_directa.py         Los cuatro modelos sobre una imagen real
  comparar_rutas.py       Primera comparación entre las dos rutas
resultados/               Figuras y salidas de texto
NOTAS_METODOLOGIA.md      Decisiones que deben quedar en el capítulo VI
```

## Uso

```
pip install -r requirements.txt
python -m verificacion.verificar_directa
python -m verificacion.verificar_conforme      # unos 3 minutos
python -m experimentos.comparar_rutas          # unos 2 minutos
```

Desde Python:

```python
from inpainting import SolverArmonico, RutaConforme
sd = SolverArmonico(mascara)      # etapa por máscara de la ruta directa (factorización)
rc = RutaConforme(mascara)        # etapa por máscara de la ruta conforme (mapeo)
u_dir = sd.resolver(imagen).u     # etapa por dato
u_conf = rc.resolver(imagen).u
u_tv_disco = rc.resolver_disco(imagen, "tv", eps=1e-2).u   # TV estándar en el disco
```
