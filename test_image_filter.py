# -*- coding: utf-8 -*-
"""
Tests for image_filter.py. Run them with:  python -m pytest
"""

import numpy as np
import pytest
from PIL import Image

from image_filter import KERNELS, filter_kernel, convert_to_L


def slow_filter(pixels, kernel):
    """Pixel by pixel version, like the original assignment, used as reference."""
    k = len(kernel)
    half = k // 2
    padded = np.pad(pixels.astype(float), half, mode="edge")
    out = np.zeros(pixels.shape)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            out[y, x] = sum(kernel[i][j] * padded[y + i, x + j]
                            for i in range(k) for j in range(k))
    return out


def random_image(seed=0, size=(30, 20)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size[1], size[0])).astype(np.uint8)


@pytest.mark.parametrize("name", ["blur", "gaussian", "sharpen"])
def test_matches_slow_version(name):
    pixels = random_image()
    kernel = KERNELS[name]
    expected = np.clip(slow_filter(pixels, kernel) / np.sum(kernel), 0, 255)
    result = np.asarray(filter_kernel(Image.fromarray(pixels), kernel))
    assert np.abs(result - np.round(expected)).max() == 0


def test_edges_matches_slow_version():
    pixels = random_image(1)
    expected = np.asarray(convert_to_L(np.abs(slow_filter(pixels, KERNELS["edges"]))))
    result = np.asarray(filter_kernel(Image.fromarray(pixels), KERNELS["edges"]))
    assert np.array_equal(result, expected)


def test_flat_image_stays_flat():
    img = Image.new("L", (10, 10), 120)
    assert set(np.asarray(filter_kernel(img, KERNELS["blur"])).ravel()) == {120}
    # Edges on a flat image: everything is 0, and there is no division by zero.
    assert set(np.asarray(filter_kernel(img, KERNELS["edges"])).ravel()) == {0}


def test_colour_image_is_accepted():
    img = Image.new("RGB", (8, 6), (200, 10, 10))
    out = filter_kernel(img, KERNELS["blur"])
    assert out.mode == "L" and out.size == (8, 6)


def test_bad_kernel():
    with pytest.raises(ValueError):
        filter_kernel(Image.new("L", (5, 5)), [[1, 1], [1, 1]])
