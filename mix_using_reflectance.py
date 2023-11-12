import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



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
    return np.clip([int(np.round(v * 255)) for v in sRGB], 0, 255)


def get_sRGB_from_reflectance_instance(instance, t_matrix, denormalization=True):
    d65_weighted_linear_rgb = np.matmul(t_matrix.values, instance.T).T
    sRGB = convert_by_gamma_correction(d65_weighted_linear_rgb)
    if denormalization:
        return denormalize(sRGB)
    else:
        return sRGB


def reflectance_mix(instances, ratios):
    result = np.power(instances[0], 1. - ratios[0])
    for i in range(1, len(instances)):
        result = result * np.power(instances[i], 1. - ratios[i])
    return result


def rgb2lab(inputColor):

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab


def find_nearest_sRGB(sRGB, candidates):
    lab = np.array(rgb2lab(sRGB))
    lab_candidates = np.array([rgb2lab(sRGB) for sRGB in candidates])
    return np.argmin(np.linalg.norm(lab_candidates - lab, axis=1))


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


def get_gradients(rgb_components, color_index, t_matrix, points=1):
    reversed_color_index = {y: x for x, y in color_index.items()}

    candidates = list(color_index.keys())
    indexes = [find_nearest_sRGB(rgb, candidates) for rgb in rgb_components]
    rgb_components = [
        np.clip(np.array(reversed_color_index[idx]), 0, 255)
        for idx in indexes
    ]

    gradients = []
    for i in range(len(rgb_components) - 1):
        for j in range(i + 1, len(rgb_components)):
            row = []
            row.append(rgb_components[i])
            for k in range(points):
                t = (k + 1) / (points + 1)

                instance1 = reflectance.loc[indexes[i],:].values
                instance2 = reflectance.loc[indexes[j],:].values
                instance = reflectance_mix([instance1, instance2], [t, 1-t])

                sRGB = get_sRGB_from_reflectance_instance(instance, t_matrix)
                row.append(sRGB)
            row.append(rgb_components[j])
            gradients.append(row)
    return np.array(gradients)


def get_gradients(rgb_components, points=1):
    latents = [mixbox.rgb_to_latent(rgb_component) for rgb_component in rgb_components]

    gradients = []
    for i in range(len(rgb_components) - 1):
        for j in range(i + 1, len(rgb_components)):
            row = []
            row.append(rgb_components[i])
            for k in range(points):
                t = (k + 1) / (points + 1)

                mix_latent = (1 - t) * np.array(latents[i]) + t * np.array(latents[j])
                sRGB = mixbox.latent_to_rgb(mix_latent)

                row.append(sRGB)
            row.append(rgb_components[j])
            gradients.append(row)
    return np.array(gradients)


t_matrix = load_t_matrix()
reflectance = load_reflectance()
color_index = load_color_index()

import colorsys

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

gradients = get_gradients(rgb_components, points=3)

order_idx = np.argsort(
    np.apply_along_axis(lambda x: (0 if colorsys.rgb_to_hsv(*(x / 255.).tolist())[2] > 0.5 else 1) + colorsys.rgb_to_hsv(*(x / 255.).tolist())[0],
                        1, gradients[:, 2, :]),
)
order_idx = np.argsort(
    np.apply_along_axis(lambda x: colorsys.rgb_to_hsv(*(x / 255.).tolist())[0],
                        1, gradients[:, 0, :]),
)
gradients = gradients[order_idx]

chuck_size = gradients.shape[0] // 5
m = np.ones((chuck_size, gradients.shape[1] * 5 + 4, 3)) * 255
m[:, 0                         :gradients.shape[1] * 1 + 0, :] = gradients[0           :chuck_size*1, :, :]
m[:, gradients.shape[1] * 1 + 1:gradients.shape[1] * 2 + 1, :] = gradients[chuck_size*1:chuck_size*2, :, :]
m[:, gradients.shape[1] * 2 + 2:gradients.shape[1] * 3 + 2, :] = gradients[chuck_size*2:chuck_size*3, :, :]
m[:, gradients.shape[1] * 3 + 3:gradients.shape[1] * 4 + 3, :] = gradients[chuck_size*3:chuck_size*4, :, :]
m[:, gradients.shape[1] * 4 + 4:gradients.shape[1] * 5 + 4, :] = gradients[chuck_size*4:chuck_size*5, :, :]

draw_color_matrix(m)
