# -*- coding: utf-8 -*-
"""
Applies a convolution kernel (blur, edges, sharpen...) to a greyscale image.

Original version: university assignment
Revised: November 2026.

@author: Javidtl
"""

import argparse

import numpy as np
from PIL import Image


KERNELS = {
    # Average of the 3x3 neighbourhood.
    "blur": [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],
    # Weighted average, smoother than "blur".
    "gaussian": [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]],
    # Laplacian: highlights edges, flat areas go black.
    "edges": [[0, -1, 0],
              [-1, 4, -1],
              [0, -1, 0]],
    # Sharpen: the image plus its edges.
    "sharpen": [[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]],
}


def filter_kernel(img, kernel):
    """
    Filters a greyscale image with a square kernel of odd size and returns
    a new greyscale image (mode "L").
    -----
    Each pixel becomes the sum of its neighbours multiplied by the kernel
    values. If the kernel values add up to something other than 0, the
    result is divided by that sum and clipped to 0-255. If they add up to 0
    (edge detection), the absolute value is rescaled with convert_to_L().
    """
    pixels = np.asarray(img.convert("L"), dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] \
       or kernel.shape[0] % 2 == 0:
        raise ValueError("The kernel must be square with an odd size (3x3, 5x5...)")

    half = kernel.shape[0] // 2
    height, width = pixels.shape

    # Repeat the border pixels outwards so the edges are filtered too.
    padded = np.pad(pixels, half, mode="edge")

    # Loop over the kernel cells instead of the pixels: each step adds the
    # whole image, shifted and multiplied by one kernel value.
    result = np.zeros((height, width))
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            result += kernel[i, j] * padded[i:i + height, j:j + width]

    # Kernels that add up to a non-zero value are normalised and clipped.
    # Edge kernels add up to 0: take the absolute value and rescale.
    total = kernel.sum()
    if total != 0:
        result = np.clip(result / total, 0, 255)
        return Image.fromarray(np.round(result).astype(np.uint8))
    return convert_to_L(np.abs(result))


def convert_to_L(values):
    """
    Rescales an array of any range to 0-255 (the minimum becomes black and
    the maximum white) and returns it as a greyscale image (mode "L").
    """
    minimum, maximum = values.min(), values.max()
    interval = maximum - minimum
    if interval == 0:
        return Image.new("L", (values.shape[1], values.shape[0]), 0)
    scaled = np.round((values - minimum) / interval * 255)
    return Image.fromarray(scaled.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(
        description="Apply a convolution kernel to a greyscale image.")
    parser.add_argument("image", help="input image")
    parser.add_argument("-k", "--kernel", default="blur",
                        choices=sorted(KERNELS),
                        help="filter to apply (default: blur)")
    parser.add_argument("-o", "--output",
                        help="output file (default: <image>_<kernel>.png)")
    parser.add_argument("--show", action="store_true",
                        help="open the original and the result on screen")
    args = parser.parse_args()

    img = Image.open(args.image)
    result = filter_kernel(img, KERNELS[args.kernel])

    output = args.output
    if output is None:
        name = args.image.rsplit(".", 1)[0]
        output = "{0}_{1}.png".format(name, args.kernel)
    result.save(output)
    print("Saved", output)

    if args.show:
        img.convert("L").show()
        result.show()


if __name__ == "__main__":
    main()
