# analysis/ssm.py

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def compute_ssm(features, eps=1e-8):
    """
    features: (N, D)
    cosine similarity based SSM
    """
    norm = np.linalg.norm(features, axis=1, keepdims=True) + eps
    X = features / norm
    return np.dot(X, X.T)


def checkerboard_kernel(size):
    """
    size: half window size
    """
    kernel = np.zeros((2 * size, 2 * size))
    kernel[:size, :size] = 1
    kernel[size:, size:] = 1
    kernel[:size, size:] = -1
    kernel[size:, :size] = -1
    return kernel


def compute_novelty_curve(ssm, kernel_size=8):
    """
    Slide checkerboard kernel along diagonal
    """
    kernel = checkerboard_kernel(kernel_size)
    k = kernel_size
    novelty = np.zeros(ssm.shape[0])

    for i in range(k, ssm.shape[0] - k):
        sub = ssm[i - k:i + k, i - k:i + k]
        novelty[i] = np.sum(sub * kernel)

    # normalize
    if novelty.max() > 0:
        novelty /= novelty.max()

    return novelty


def novelty_segmentation(
    times,
    novelty,
    peak_prominence=0.2,
    min_section_length=3.0
):
    """
    times: window start times
    """
    peaks, _ = find_peaks(novelty, prominence=peak_prominence)

    boundaries = [times[p] for p in peaks]

    sections = []
    prev = times[0]

    for b in boundaries:
        if b - prev >= min_section_length:
            sections.append({"start": prev, "end": b})
            prev = b

    # last section
    if times[-1] - prev >= min_section_length:
        sections.append({"start": prev, "end": times[-1]})

    return sections
