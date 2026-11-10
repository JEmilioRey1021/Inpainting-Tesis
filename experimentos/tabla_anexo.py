"""
Cuadros por imagen del anexo C a partir de resultados/banco.csv.

    python -m experimentos.tabla_anexo

Escribe resultados/anexo_banco.tex, con un cuadro por imagen, listo para
copiar en el anexo C de la tesis.
"""

from __future__ import annotations

import csv
import os

SALIDA = os.path.join(os.path.dirname(__file__), "..", "resultados")


def fila(r):
    if r["familia"] == "rayas":
        nombre = f"Raya, $L={r['parametro']}$"
    else:
        nombre = {"manchas": "Manchas", "texto": "Texto"}[r["familia"]]
    total = int(r["comp_mapeo"]) + int(r["comp_directa"])
    vals = [float(r[k]) for k in ("psnr_arm_dir", "psnr_arm_conf", "dif_arm",
                                  "psnr_tv_dir", "psnr_tv_disco", "dif_tv")]
    return (f"    {nombre} & {int(r['pixeles'])} & {vals[0]:.2f} & {vals[1]:.2f} & "
            f"{vals[2]:.4f} & {vals[3]:.2f} & {vals[4]:.2f} & {vals[5]:.4f} & "
            f"{r['comp_directa']} de {total} \\\\")


def main():
    ruta = os.path.join(SALIDA, "banco.csv")
    try:
        with open(ruta, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
    except UnicodeDecodeError:
        # banco.csv escrito en Windows con la codificación del sistema
        with open(ruta, encoding="cp1252") as f:
            filas = list(csv.DictReader(f))
    imagenes = list(dict.fromkeys(r["imagen"] for r in filas))
    partes = []
    for img in imagenes:
        etiqueta = (img.replace("á", "a").replace("é", "e").replace("ó", "o")
                    .replace("í", "i").replace("ú", "u"))
        partes.append("\n".join([
            r"\begin{table}[ht]",
            r"  \centering",
            rf"  \caption{{Banco de pruebas, imagen {img}.}}",
            rf"  \label{{tab:anexo-{etiqueta}}}",
            r"  \small",
            r"  \begin{tabular}{@{}lrccccccc@{}}",
            r"    \toprule",
            r"    & & \multicolumn{3}{c}{Armónico} & \multicolumn{3}{c}{Variación total} & \\",
            r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"    Máscara & Píxeles & Directa & Conforme & Dif. & Directa & Disco & Dif. & Sin mapeo \\",
            r"    \midrule",
            *[fila(r) for r in filas if r["imagen"] == img],
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \propia",
            r"\end{table}",
        ]))
    with open(os.path.join(SALIDA, "anexo_banco.tex"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(partes) + "\n")
    print("Escrito resultados/anexo_banco.tex")


if __name__ == "__main__":
    main()