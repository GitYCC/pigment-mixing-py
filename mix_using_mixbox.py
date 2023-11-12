import warnings
warnings.filterwarnings("ignore")

import mixbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint


def draw_color_matrix(m, texts=None, box_size=(0.95, 0.8)):
    height, width = m.shape[0], m.shape[1]
    fig, ax = plt.subplots(figsize=(width, height), layout='constrained')
    ax.set_xlim(-0.5, m.shape[1]+0.5)
    ax.set_ylim(-1 * m.shape[0] -0.5, 0.5)
    ax.axis('off')

    for i, row in enumerate(m):
        for j, c in enumerate(row):
            ax.add_patch(Rectangle((j, - i - 1), box_size[0], box_size[1], color=np.array(c)/255))
            if texts is not None:
                ax.text(j + 0.1, - i - 1 + 0.1, texts[i][j], color='lightgray')
    plt.show()


def decomposite(target, components, k, precision=0.01):
    target = np.array(mixbox.rgb_to_latent(target))
    components = [np.array(mixbox.rgb_to_latent(c)) for c in components]

    def objective(x):
        res = target - np.sum([c * x[i] for i, c in enumerate(components)], axis=0)
        loss = np.inner(res, res)
        return loss

    def optimize(zero_list=None):
        lc_matrix = []
        lc_low = []
        lc_high = []

        # sum of coeff is 1.0
        lc_matrix.append([1.0] * len(components))
        lc_low.append(1.0)
        lc_high.append(1.0)

        # range of coeff
        for i in range(len(components)):
            lc_matrix.append([1. if i == j else 0. for j in range(len(components))])
            lc_low.append(0.)
            if zero_list and i in zero_list:
                lc_high.append(0.)
            else:
                lc_high.append(1.)

        linear_constraint = LinearConstraint(lc_matrix, lc_low, lc_high)

        if zero_list:
            x0 = np.array([
                1 / (len(components) - len(zero_list)) if i not in zero_list else 0.
                for i in range(len(components))])
        else:
            x0 = np.array([1 / len(components)] * len(components))
        res = minimize(objective, x0, method='SLSQP',
                    constraints=[linear_constraint], tol=0.0001)
        return res.x

    coeff = optimize()
    zero_list = np.argpartition(coeff, -k)[:-k].tolist()
    coeff = optimize(zero_list=zero_list)

    second_min = np.min(coeff[coeff > 0.0001])
    while second_min < precision:
        idx = np.argwhere(coeff == second_min)[0]
        zero_list.append(idx)
        coeff = optimize(zero_list=zero_list)
        second_min = np.min(coeff[coeff > 0.0001])

    return np.round(coeff / precision, 0) * precision


def mix_by_coeff(components, coeff):
    latent = np.sum(
        [np.array(mixbox.rgb_to_latent(comp)) * c for comp, c in zip(components, coeff)],
        axis=0)
    return mixbox.latent_to_rgb(latent)


rgb_components = [
    (180, 19, 27), # 313 Azo Red Deep
    (201, 28, 73), # 366 Quinacridone Rose
    (238, 63, 0), # 303 Cadmium Red Light
    (251, 137, 22), # 211 Cadmium Orange
    (255, 192, 0), # 270 Azo Yellow Deep
    (242, 235, 0), # 267 Azo Yellow Lemon
    (198, 155, 1), # 227 Yellow Ochre
    (153, 216, 1), # 617 Yellowish Green
    (1, 61, 49), # 616 Viridian
    (24, 61, 168), # 511 Cobalt Blue
    (14, 145, 213), # 530 Sevres Blue
    (0, 12, 124), # 504 Ultramarine
    (113, 37, 21), # 411 Burnt Sienna
    (43, 37, 25), # 408 Raw Umber
    (244, 240, 240), # 105 Titanium White
    (18, 0, 18), # 536 Violet
    (20, 19, 19), # 702 Lamp Black
]


def main(): 
    mat = []
    texts = []
    for x in range(50):
        h = 0.02 * x
        sv_list = [[0.6, 255], [1.0, 255], [1.0, 200], [1.0, 150]]
        row = []
        text_row = []
        for s, v in sv_list:
            target_rgb = colorsys.hsv_to_rgb(h, s, v)
            row.append(target_rgb)
            text_row.append('target')
    
            coeff = decomposite(target_rgb, rgb_components, k=4)
            rec_rgb = mix_by_coeff(rgb_components, coeff)
            row.append(rec_rgb)
            text_row.append('mixing')
    
            idxes = np.argsort(coeff)
            idxes.shape = (-1,)
            for i in range(4):
                if coeff[idxes[-1 - i]] > 0.0001:
                    row.append(rgb_components[idxes[-1 - i]])
                    text_row.append(f'* {coeff[idxes[-1 - i]]:.2f}')
                else:
                    row.append([255,255,255])
                    text_row.append('')
            row.append([255,255,255])
            text_row.append('')
    
        mat.append(row)
        texts.append(text_row)
    
    draw_color_matrix(np.array(mat), texts=texts)


if __name__ == '__main__':
    main()
