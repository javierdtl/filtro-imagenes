# Filtros de imagen por convolución

Programa en Python que aplica un kernel de convolución a una imagen en escala de grises: desenfoque, detección de bordes o enfoque. Empezó como práctica del grado de Ingeniería Matemática (UCM) en 2017 y lo he revisado en 2026. La versión original recorría la imagen píxel a píxel; esta hace las cuentas con NumPy.

| Original | `blur` |
|---|---|
| ![original](examples/sample.png) | ![blur](examples/sample_blur.png) |
| **`edges`** | **`sharpen`** |
| ![edges](examples/sample_edges.png) | ![sharpen](examples/sample_sharpen.png) |

## Instalación y uso

Necesita Python 3, NumPy y Pillow.

```
pip install -r requirements.txt
python image_filter.py examples/sample.png --kernel edges
```

El resultado se guarda como `examples/sample_edges.png`. Opciones:

- `-k`, `--kernel`: filtro a aplicar, `blur` (por defecto), `gaussian`, `edges` o `sharpen`.
- `-o`, `--output`: nombre del archivo de salida.
- `--show`: abre la imagen original y la filtrada al terminar.

Las imágenes en color se pasan a gris antes de filtrar.

También se puede usar desde Python con un kernel propio, siempre que sea cuadrado y de lado impar:

```python
from PIL import Image
from image_filter import filter_kernel

kernel = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]   # Sobel horizontal
result = filter_kernel(Image.open("foto.png"), kernel)
result.save("foto_sobel.png")
```

## Cómo funciona

Cada píxel nuevo es la suma de sus vecinos multiplicados por los valores del kernel. Por ejemplo, con un kernel de 3x3 lleno de unos, cada píxel pasa a ser la media de los 9 que lo rodean, y la imagen se suaviza.

En vez de recorrer la imagen píxel a píxel, el programa recorre las casillas del kernel (9 en uno de 3x3). En cada paso suma la imagen entera desplazada y multiplicada por ese valor. Para los bordes, la imagen se amplía repitiendo los píxeles del extremo.

Después hay que volver al rango 0-255:

- Si los valores del kernel suman algo distinto de 0 (`blur`, `gaussian`, `sharpen`), se divide entre esa suma y se recorta lo que quede fuera del rango.
- Si suman 0 (`edges`), se toma el valor absoluto y se reescala del mínimo al máximo.

Con la imagen de ejemplo (320x200 píxeles), la versión de 2017 tardaba algo más de un segundo y esta unas milésimas.

## Tests

`test_image_filter.py` compara el resultado con una versión lenta, píxel a píxel, y prueba algunos casos límite (imagen lisa, imagen en color, kernel no válido).

```
pip install pytest
python -m pytest
```

## Autor

Javier de Torres
