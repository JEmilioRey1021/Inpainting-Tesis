# Notas para el capítulo VI (Metodología)

Decisiones tomadas en el código que el capítulo VI tiene que explicar y justificar.

## Ruta directa

1. Discretización en forma de divergencia. Los cuatro modelos son u_t = div(b ∇u). La
   matriz se arma con conductancias b en las aristas entre píxeles vecinos (4 vecinos).
   Con b = 1 es el laplaciano de 5 puntos. Es la versión discreta de prop:divergencia.
2. Dirichlet. Los píxeles conocidos vecinos de la máscara pasan al lado derecho. La matriz
   es simétrica y definida positiva (prueba 1).
3. Gradiente en las aristas. Diferencia a través de la arista y promedio de diferencias
   centradas en la dirección transversal.
4. Esquema semi-implícito para PM y TV. (I + dt A(u^n)) u^{n+1} = u^n + dt rhs(u^n).
   Estable para todo dt (matriz M). El calor usa Euler implícito.
5. TV estacionaria por difusividad retardada (Vogel y Oman), con |∇u|_eps.
6. TV con el dato FIJO fuera de la máscara, no con el término de fidelidad de Chan–Shen.
   Razón, el mapeo solo traslada la frontera y no la franja exterior (obs:franja-conforme).
7. Relleno inicial de las evoluciones con el promedio del dato en el borde. Consecuencia
   observada, en un disco ese relleno no tiene componente en la primera función propia
   (fórmula de Green), y el calor converge con la tasa del segundo valor propio.
8. Factorización LU dispersa (SuperLU, ordenamiento COLAMD) una vez por máscara.

## Ruta conforme

1. Polígono. Curva de nivel 0.9 de la máscara dilatada un píxel, simplificada con
   Douglas–Peucker (tolerancia 0.8 px). Pasa a una décima de píxel de los centros del primer
   anillo de píxeles conocidos, que son los datos de Dirichlet de la ruta directa.
   - Compromiso medido: con tolerancia 0.3 el error con dato exacto baja de 3e-3 a 1.5e-5,
     pero el polígono pasa de 36 a 184 vértices y el mapeo de 7 s a más de 5 minutos.
2. Componentes. Cada componente conexa de la máscara dilatada tiene su propio mapeo.
   Si tiene huecos (no es simplemente conexa) o el problema de parámetros no converge, se
   resuelve con la ruta directa y se informa.
3. Centro conforme. Entre los píxeles con distancia a la frontera de al menos el 90 % de la
   máxima, el más cercano al centroide. Escoger el máximo sin más falla en las rayas, porque
   el máximo se alcanza en toda la línea media y un extremo aglomera exponencialmente al otro.
4. Problema de parámetros. Se fija psi(0) = centro y w_{n-1} = 1. Incógnitas, los n-1
   espaciamientos libres (parametrizados con softmax para que sean positivos). Ecuaciones,
   n-3 cocientes de lados y psi(0) = centro. Levenberg–Marquardt.
5. Cuadratura de SC. Gauss–Jacobi en el tramo que sale de un prevértice, y luego tramos
   Gauss–Legendre de longitud menor que la mitad de la distancia a la singularidad más
   cercana (Driscoll y Trefethen).
6. Inversa phi. Tabla de psi en una malla polar refinada hacia el círculo y hacia los
   prevértices, valor inicial por vecino más cercano, Newton con integración desde el
   iterado anterior.
7. Coeficientes de Fourier por cuadratura graduada, NO por FFT. G(t) = g(psi(e^{it})) tiene
   singularidades |t - t_k|^alpha_k en los prevértices. Con muestras equiespaciadas el
   aliasing contamina todos los coeficientes y el error es algebraico incluso en el
   interior (1e-3). Con paneles graduados el interior llega a 1e-11 (prueba 2).
8. Evaluación híbrida. Serie de Fourier (Horner) si |phi(z)| <= 0.95. Integral de Poisson
   con paneles graduados a escala 1 - |w| alrededor de arg(w) si |phi(z)| > 0.95, porque la
   serie necesita del orden de 1/(1 - |w|) términos y en las rayas hay |w| = 0.99999.
9. Modelos no lineales. Se resuelve el modelo ESTÁNDAR en una malla cartesiana del disco con
   el mismo código de la ruta directa, y se regresa por interpolación bilineal en phi(z).
   K, eps y T se convierten con el factor típico lambda = mediana de |psi'|.

## Resultados ya obtenidos que conviene citar

- Orden 2 del laplaciano discreto (prueba directa 2).
- Separación mínima de prevértices en rectángulos = 8 e^{-pi L/2}, pendiente medida -1.5707
  contra -pi/2. Nuestra implementación converge hasta L = 13 y falla desde L = 14 (separación
  2e-9), antes del límite teórico de la doble precisión (L cerca de 23).
- Costo por dato a 256 x 256. La ruta directa ya factorizada es más barata que la conforme
  por dato, no solo en la etapa por máscara. Ver resultados/comparacion_rutas.txt.

## Primera comparación (resultados/comparacion_rutas.txt)

- Armónico. Las dos rutas dan el mismo PSNR (diferencias de 0.1 dB, y de 0.7 dB en la raya
  L=8). La diferencia entre ellas es pequeña en el interior y se concentra cerca de la
  frontera, donde el dato real de la imagen no es suave.
- Costo. Etapa por máscara, factorización 1 a 8 ms contra mapeo 1.5 a 4 s. Etapa por dato,
  ruta directa 0.2 a 1 ms contra ruta conforme 6 a 14 ms. A este tamaño la ruta conforme no
  amortiza nunca, porque también es más cara por dato.
- Raya L=16. El problema de parámetros no converge y la componente se resuelve por la ruta
  directa (predicción del capítulo V sobre aglomeración).
- Modelos no invariantes. CUIDADO, el resultado de la mancha grande (diferencia unas 3 veces
  la del armónico) NO es evidencia limpia de P2. Con daños pequeños
  (experimentos/demo_danos_pequenos.py) el armónico resuelto con la misma malla cartesiana
  del disco da 16.6 dB, igual que TV en el disco (16.7 dB), mientras que el armónico por la
  serie da 21.0 dB. La pérdida viene de la malla uniforme del disco, que no resuelve los arcos
  aglomerados (separaciones de 1e-5 en los rayones), no de la no invariancia.
- Pendiente para medir P2 limpiamente. (a) Comparar el modelo en el disco contra el armónico
  resuelto con la misma malla del disco, para cancelar el error de discretización. (b) Usar una
  malla polar graduada hacia los prevértices.

## Daños realistas (resultados/demo_danos_pequenos.png)

- 5 manchas de radio 4 a 8 px y 3 rayones de 4 px de ancho sobre la foto de 512 x 512.
  Armónico directa 21.6 dB, conforme 21.0 dB, TV directa 22.9 dB, PM directa 22.5 dB.
- Las 8 componentes tuvieron mapeo. Las diferencias entre rutas se concentran en las puntas
  de los rayones, donde se aglomeran los prevértices.
- Tensión central para la discusión. El inpainting por EDP funciona en daños delgados o
  pequeños, y los daños delgados y alargados son justo donde la ruta conforme sufre por la
  aglomeración.

## P2 medida sin error de malla (experimentos/medir_p2.py, resultados/p2.txt)

- Idea. Resolver el modelo ESTÁNDAR en el disco y regresar con phi equivale, por
  prop:divergencia, a resolver en D un modelo con el campo rho = lambda |phi'|
  (lambda = mediana de |psi'|). TV estándar del disco = div(rho ∇u / sqrt((rho eps)^2+|∇u|^2)) = 0.
  PM estándar del disco = rho^2 u_t = div(c(|∇u|^2/(K rho)^2) ∇u). Se resuelve en la MISMA
  malla de píxeles que la ruta directa, así que la diferencia es pura no invariancia.
  Para el armónico la diferencia es cero por construcción, porque el laplaciano no ve rho.
- rho varía mucho dentro de una misma componente. En los rayones va de 0.002 a 134.
- Daños pequeños (512 x 512). La diferencia media en el 10 % de píxeles con |grad log rho|
  más alto es unas 12 veces la del 50 % más bajo en TV (0.0275 contra 0.0022) y unas 20 veces
  en PM. Spearman 0.28 (TV) y 0.40 (PM). Esto SÍ apoya P2.
- Mancha grande. Efecto débil (Spearman 0.10 a 0.18), porque rho varía poco.
- Raya L=8. TV pierde 7 dB en el disco. En PM la correlación sale negativa, porque en la raya
  casi todos los píxeles tienen |grad log rho| alto y el rango es pequeño.
- Matiz importante para la discusión. TV solo difiere donde hay un borde de la imagen que
  cruza la máscara (resultados/p2_daños_pequeños.png). En zonas planas los dos modelos dan lo
  mismo aunque rho varíe. La formulación precisa de P2 es que la diferencia crece con
  |grad log rho| DONDE la solución tiene bordes, lo cual es exactamente lo que dice
  cor:tv-no-invariante (la curvatura cambia por grad log|psi'| · n).
- Comprobación de la equivalencia (mancha grande, TV). Malla cartesiana del disco de 100, 200
  y 400 px contra TV con rho, diferencia 0.019, 0.014, 0.014. Contra la ruta directa, 0.030,
  0.032, 0.032. El disco se acerca a la versión con rho y se aleja de la directa, pero se
  estanca en 0.014 por el error de borde de la malla cartesiana del disco. Apoya la
  equivalencia sin demostrarla numéricamente al cien por ciento.

## Costo según la resolución (experimentos/costo_escala.py, resultados/costo_escala.txt)

- Misma escena a 128, 256, 512, 1024 px, M = 4N términos.
- Mancha a 1024 (53785 px en D). Factorización 0.27 s, mapeo 27 s. Por dato, directa 29 ms,
  conforme 284 ms.
- Exponentes (tiempo ~ píxeles^p). Mancha, factorización 1.15, dato directa 1.08, mapeo 0.72,
  dato conforme 1.15. Raya, 0.91, 0.87, 0.77, 0.93.
- Conclusión. El costo por dato de la ruta conforme crece con el mismo exponente que el de la
  directa y es 8 a 30 veces mayor, así que la ruta conforme no amortiza a ninguna resolución
  probada. El mapeo crece más lento que la factorización (0.72 contra 1.15), pero parte unas
  100 veces más caro y no lo alcanza en ningún tamaño realista. P3 se confirma en su forma
  fuerte para nuestra implementación.
- Razón teórica para la discusión. Por dato, la conforme cuesta del orden de
  (píxeles) x (términos), y los términos deben crecer como el perímetro en píxeles, es decir,
  como raíz de los píxeles, lo que da del orden de n^1.5. La directa factorizada cuesta del
  orden de nnz(L), que en 2D con disección anidada es del orden de n log n. En el rango medido
  el exponente de la conforme salió 1.15, no 1.5, probablemente porque a estos tamaños pesan
  los costos fijos de numpy, pero en ningún caso hay un régimen asintótico favorable.

## Banco de pruebas (experimentos/banco.py, resultados/banco_resumen.txt, banco.csv, banco_rayas.png)

- 5 imágenes (camarógrafo, astronauta, café, luna, gato) a 256 x 256. Familias, manchas
  (6 elipses de radio 4 a 8), rayas de 4 px de ancho con L = 2..14, y texto ("Tesis UVG").
- Armónico. La conforme queda 0.4 dB debajo de la directa en manchas y 1.3 a 1.8 dB en
  rayas. La brecha NO crece de forma sistemática con L, así que no la causa la aglomeración.
  La causa es la discretización de máscaras muy delgadas (4 px de ancho, 4 incógnitas a lo
  ancho). Evidencia, en costo_escala.txt la misma raya a 128, 256, 512, 1024 px da brechas de
  0.46, 0.22, 0.10, 0.05 dB, que se reducen a la mitad cada vez que se duplica la resolución,
  es decir, error O(h). Bajar la tolerancia del polígono de 0.8 a 0.2 px solo recupera 0.3 a
  0.8 dB y multiplica el tiempo del mapeo por 10 a 20.
- TV. El modelo estándar del disco pierde de 0.5 a 2.2 dB frente a la directa en las rayas,
  más con L entre 8 y 14. En manchas casi no cambia (incluso +0.2 dB).
- Aglomeración. La separación mínima sigue 8 exp(-pi L_ef/2) con L_ef = (4L+2)/6, la razón de
  aspecto del polígono que rodea la raya dilatada. La raya oblicua de la imagen del astronauta
  se aparta (su polígono escalonado tiene más vértices) y falla desde L = 12.
- Componentes sin mapeo. 0 de 29 en manchas, 1 de 5 en L = 12 y L = 14, y 35 de 45 en texto
  (letras con huecos como la e o la o, y trazos delgados). La ruta conforme casi no es
  aplicable a texto superpuesto, que es uno de los daños típicos del inpainting.
- Costo. Mapeo 0.3 a 4.8 s en rayas (crece con L), 2.7 s en manchas, 38 s en texto, frente a
  1 a 3 ms de factorización. Por dato, la conforme es 9 a 26 veces más lenta.

## Cambio de robustez (10 nov.)

- En la computadora de Emilio (Windows, Python 3.11.9, NumPy 2.1.3, SciPy 1.15.3) el problema
  de parámetros fallaba en casos que aquí convergían (por ejemplo la mancha a 128 px). La
  convergencia de UN intento de Levenberg–Marquardt dependía de la versión de SciPy.
- Arreglo en sc.py. Se prueban dos valores iniciales (ángulos desde el centro y longitudes de
  los lados) y dos métodos (lm y trf), y se queda el mejor. Los residuos con NaN o inf se
  reemplazan por un valor grande.
- Con el arreglo, y con SciPy 1.15.3, el banco da 0 de 29 manchas sin mapeo, 1 de 5 en L=12,
  0 de 5 en L=14 y 20 de 45 componentes de texto sin mapeo (antes 35 de 45). Los rectángulos
  siguen convergiendo hasta L=13 y fallando desde L=14.
- Los números del banco que se citen en la tesis deben salir de la corrida final en la
  computadora de Emilio.
