import numpy as np
import pandas as pd


def load_t_matrix():
    path = 'data/reflectance_to_D65_weighted_linear_rgb_matrix.csv'
    matrix = pd.read_csv(path, index_col=0)
    return matrix


def load_color_index():
    path = 'data/color_index.csv'
    df = pd.read_csv(path)
    color_index = {}
    for _, row in df.iterrows():
        sRGB = (row['R'], row['G'], row['B'])
        color_index[sRGB] = row['index']
    return color_index


def load_reflectance():
    path = 'data/reflectance.csv'
    reflectance = pd.read_csv(path, index_col=0)
    return reflectance


def convert_by_gamma_correction(d65_weighted_linear_rgb):
    sRGB = [
        12.92 * v if v < 0.0031308 else 1.055 * (v ** (1/2.4)) - 0.055
        for v in d65_weighted_linear_rgb
    ]
    return sRGB


def denormalize(sRGB):
    return [int(np.round(v * 255)) for v in sRGB]


def get_sRGB_from_reflectance_instance(instance, t_matrix, denormalization=True):
    d65_weighted_linear_rgb = np.matmul(t_matrix.values, instance.T).T
    sRGB = convert_by_gamma_correction(d65_weighted_linear_rgb)
    if denormalization:
        return denormalize(sRGB)
    else:
        return sRGB


def reflectance_mix(instances, ratios):
    result = np.power(instances[0], ratios[0])
    for i in range(1, len(instances)):
        result = result * np.power(instances[i], ratios[i])
    return result


t_matrix = load_t_matrix()
reflectance = load_reflectance()
instance1 = reflectance.loc[0,:].values
instance2 = reflectance.loc[177,:].values
instance = reflectance_mix([instance1, instance2], [0.5, 0.5])
print(get_sRGB_from_reflectance_instance(instance, t_matrix))
