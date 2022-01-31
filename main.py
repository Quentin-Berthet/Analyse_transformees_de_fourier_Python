#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import argparse

__authors__ = ["Baptiste Coudray", "Quentin Berthet"]
__date__ = "11.05.2020"
__course__ = "Mathématiques en technologies de l'information"
__description__ = "Les transformées de Fourier"


def w(n):
    power = complex(-2 * np.pi * 1j) / n
    return np.exp(power)


def build_w_matrix(shape: int):
    matrix = np.ones((shape, shape), dtype=np.complex)
    for row in range(1, shape):
        for column in range(1, shape):
            matrix[row, column] = w(shape) ** (row * column)
    return matrix


def tfd(signal):
    w1 = build_w_matrix(signal.shape[0])
    return w1.dot(signal.T)


def tfd2(signal):
    w1 = build_w_matrix(signal.shape[1])
    w2 = build_w_matrix(signal.shape[0])
    return w2.dot(w1.dot(signal.T).T)


def itfd(signal):
    w1 = np.linalg.inv(build_w_matrix(signal.shape[0]))
    return w1.dot(signal.T)


def itfd2(signal):
    w1 = np.linalg.inv(build_w_matrix(signal.shape[1]))
    w2 = np.linalg.inv(build_w_matrix(signal.shape[0]))
    return w2.dot(w1.dot(signal.T).T)


def f(t):
    return 2.3 * np.sin(2 * np.pi * t) + 0.1 * np.sin(10 * np.pi * t)


def plot_f():
    xs = np.arange(0, 1.001, 0.001)
    ys = np.array([f(x) for x in xs])

    plt.figure()
    plt.plot(xs, ys)
    plt.show()

    res = tfd(ys)
    res[-1] = 0
    ys = np.abs(itfd(res))

    plt.figure()
    plt.plot(xs, ys)
    plt.show()


def show_img(img, title, save=False, cmap="viridis"):
    plt.figure()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    if save:
        plt.savefig(f"images_debruitees/{title}.png")
    plt.show()


def surface_plot_3d(image, label):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    tfd2_image = tfd2(image)
    xx, yy = np.mgrid[0:tfd2_image.shape[0], 0:tfd2_image.shape[1]]
    bug = ax.plot_surface(xx, yy, np.abs(tfd2_image), label=label)
    bug._facecolors2d = bug._facecolors3d
    bug._edgecolors2d = bug._edgecolors3d
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"images_debruitees/sp_3d_{label}.png", bbox_inches="tight")
    plt.show()


def remove_noise(image, row_freq, col_freq):
    row_freq_start, row_freq_stop = row_freq
    col_freq_start, col_freq_stop = col_freq
    tfd2_image = tfd2(image)
    tfd2_image[:, col_freq_start:col_freq_stop] = 0
    tfd2_image[row_freq_start:row_freq_stop, :] = 0
    return np.abs(itfd2(tfd2_image)).astype(np.uint16)


def read_image(image, dtype):
    img = imageio.imread(image)
    return img.astype(dtype)


def compress(input_image_path, output_data_path, compression_level):
    input_image = read_image(input_image_path, np.uint8)
    tfd2_input_image = tfd2(input_image)
    compressed_image = np.delete(tfd2_input_image,
                                 np.s_[tfd2_input_image.shape[0] - compression_level:tfd2_input_image.shape[0]], 0)
    compressed_image = np.delete(compressed_image,
                                 np.s_[tfd2_input_image.shape[1] - compression_level:tfd2_input_image.shape[1]], 1)
    compressed_image = np.round(compressed_image, 1)

    with open(output_data_path, "w") as fh:
        fh.write(f"{tfd2_input_image.shape[0]},{tfd2_input_image.shape[1]}\n")
        fh.write(f"{compression_level}\n")
        for row in compressed_image:
            fh.write(f'{",".join([str(v).strip("()") for v in row])}\n')


def decompress(input_image_path, output_image_path):
    with open(input_image_path, "r") as fh:
        original_shape = np.int0(fh.readline().split(","))
        compression_level = int(fh.readline())
        compressed_image = None
        for line in fh:
            line = np.fromstring(line, dtype=np.complex, sep=',')
            if compressed_image is None:
                compressed_image = line
            else:
                compressed_image = np.vstack((compressed_image, line))
    for _ in range(0, compression_level):
        compressed_image = np.insert(compressed_image, original_shape[0] - compression_level, 0, axis=0)
        compressed_image = np.insert(compressed_image, original_shape[1] - compression_level, 0, axis=1)
    decompressed_image = np.abs(itfd2(compressed_image)).astype(np.uint8)
    imageio.imwrite(output_image_path, decompressed_image)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-pf", "--plot-f", required=False, action='store_true', help="Show the plotting of f(x)")
    ap.add_argument("-c", "--compress", required=False, nargs=3,
                    metavar=('input_file', 'output_file', 'compression_level (0-10)'), help="Compress a PGM file")
    ap.add_argument("-d", "--decompress", required=False, nargs=2, metavar=('input_file', 'output_file'),
                    help="Decompress a file")
    ap.add_argument("-rn", "--remove-noise", required=False, type=str,
                    choices=[str(i) for i in range(1, 17)] + ["cache", "all"],
                    nargs="+")
    args = vars(ap.parse_args())

    if args["plot_f"]:
        plot_f()
    if args["compress"] is not None:
        compression_level = int(args["compress"][2])
        if compression_level < 0:
            compression_level = 0
        elif compression_level > 10:
            compression_level = 10
        compress(args["compress"][0], args["compress"][1], compression_level)
    if args["decompress"] is not None:
        decompress(args["decompress"][0], args["decompress"][1])

    files = {
        "cache": ((142, 778), (100, 572)),  # Turing
        "1": ((150, 870), (110, 690)),  # karl max
        "2": ((200, 1050), (110, 690)),  # emmy noether
        "3": ((28, 340), (75, 535)),  # ada lovelace
        "4": ((40, 470), (90, 660)),  # muhammad idn musa al-khuwarizmi
        "5": ((60, 520), (40, 360)),  # marie curie
        "6": ((170, 910), (100, 700)),  # charles baddage
        "7": ((90, 560), (75, 425)),  # jhon von heumann
        "8": ((100, 590), (65, 365)),  # L'Internationale
        "9": ((250, 1420), (142, 1050)),  # Edsger Dijkstra
        "10": ((135, 766), (105, 665)),  # Nelson Mandela
        "11": ((450, 2565), (330, 1850)),  # Malcolm X
        "12": ((90, 525), (74, 410)),  # Joseph Fourier
        "13": ((60, 340), (95, 515)),  # Grace Hopper
        "14": ((67, 385), (66, 384)),  # Paul Albuquerque
        "15": ((90, 510), (65, 420)),  # Ludwig Boltzmann
        "16": ((135, 805), (110, 665)),  # Nelson Mandela
    }
    if args["remove_noise"] is not None:
        files_choices = args["remove_noise"]
        if "all" in files_choices:
            files_choices = ["cache"] + [str(i) for i in range(1, 17)]
        for f in files_choices:
            filename = f
            image_with_noise = read_image(f"images/{filename}.png", np.uint16)
            surface_plot_3d(image_with_noise, f"{filename}_with_noise")
            image_without_noise = remove_noise(image_with_noise, files[filename][0], files[filename][1])
            surface_plot_3d(image_without_noise, f"{filename}_without_noise")
            show_img(image_without_noise, f"{filename}_without_noise", True, "gray")
