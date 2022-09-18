import numpy as np
from scipy.ndimage import rotate
import os
from PIL import Image
import math
import cv2
import matplotlib.pyplot as plt


def ex1(matricola):
    aux_1 = list(matricola*2 + matricola[:2])
    aux_2 = list(matricola[:2] + matricola*2)
    aux_16 = []
    for i in range(len(aux_1)):
        aux_16.append(abs(int(aux_1[i])-int(aux_2[i])))

    aux_1 = list(matricola + matricola[:2])
    aux_2 = list(matricola[:4] + matricola[2:])
    aux_9 = []
    for i in range(len(aux_1)):
        aux_9.append(abs(int(aux_1[i])-int(aux_2[i])))

    image = np.array(aux_16)
    image = np.reshape(image, (4,4))
    kernel = np.array(aux_9)
    kernel = np.reshape(kernel, (3,3))

    paddings = ["Constant padding: 0", "Constant padding 1", "Reflection padding", "Symmetric padding"]
    matricola_list = list(matricola)
    totale = 0
    for elem in matricola_list:
        totale += int(elem)
    padding_index = (totale%4)
    x_coords_1 = (totale%2)+ 1
    y_coords_1 = (totale%2)+1
    x_coords_2 = (totale%4)
    y_coords_2 = (totale%4)
    if not (x_coords_2 == 0 or x_coords_2 == 3 or y_coords_2 == 0 or y_coords_2 == 3):
        x_coords_2 = 0
    operations = ['Convolution', 'Correlation']
    print("Operation:")
    print(operations[totale%2])
    print("\nImage:")
    print(image)
    print("\nKernel:")
    print(kernel)
    print("\nPaddings:")
    print(paddings[padding_index])
    print("\nFirst Coords")
    print(x_coords_1, y_coords_1)
    print("Second Coords")
    print(x_coords_2, y_coords_2)


def ex2(matricola):
    aux = [int(elem) for elem in matricola]
    totale = 0
    for elem in aux:
        totale += elem
    print("Sigma:")
    print(totale%8 +1)


def ex3(matricola):
    #aux_1 = list(matricola[3:] + matricola[4:] + matricola[3:5] +matricola[:3]+ matricola[:4])
    #aux_2 = list(matricola[6:] + matricola[3:5] + matricola[2:] + matricola[:6]+ matricola[:2])
    aux = list(matricola[3:]*4)
    aux_16 = []
    for elem in aux:
        aux_16.append(int(elem))

    image = np.array(aux_16)
    image = np.reshape(image, (4,4))

    kernel = [[-1,-2,-1],[-1,-2,-1],[-1,-2,-1]]
    kernel = np.array(kernel)
    aux = [int(elem) for elem in matricola]
    totale = 0
    for elem in aux:
        totale += elem
    rotation = 45*(totale%8)
    print("Image:")
    print(rotate(image, angle=rotation, mode='nearest', reshape=False))
    print("\nKernel:")
    print(rotate(kernel, angle=rotation, mode='nearest', reshape=False))


def ex4(matricola):

    #aux = list([['0'] + list(matricola[3:6]) +['0'], ['0']*2 + list(matricola[4]) +[0]*2, ['0']*5, ['0']*2 + list(matricola[4]) +[0]*2, ['0'] + list(matricola[3:6]) +['0']])
    aux = list(['0'] + list(matricola[3:6]) +['0']+ ['0']*2 + list(matricola[4]) +[0]*2 + ['0']*5 + ['0']*2 + list(matricola[4]) +[0]*2 + ['0'] + list(matricola[3:6]) +['0'])
    aux_16 = []
    for elem in aux:
        aux_16.append(int(elem))

    image = np.array(aux_16)
    image = np.reshape(image, (5,5))

    aux = [int(elem) for elem in matricola]
    totale = 0
    for elem in aux:
        totale += elem
    rotation = 45*(totale%8)
    operations = ['Opening', 'Closure']
    if totale%2==0:
        image[2,2] = 1
    image = np.where(image==0, image, 1)
    print("Image:")
    print(rotate(image, angle=rotation, mode='nearest', reshape=False))
    print("\nOperation:")
    print(operations[totale%2])


def ex5(matricola):
    aux = list(['0'] + list(matricola[3:6]) +['0']+ ['0']*2 + list(matricola[4]) +[0]*2 + ['0']*10 + ['0']*2 + list(matricola[4]) +[0]*2)
    aux_16 = []
    for elem in aux:
        aux_16.append(int(elem))

    image = np.array(aux_16)
    image = np.reshape(image, (5,5))

    aux = [int(elem) for elem in matricola]
    totale = 0
    for elem in aux:
        totale += elem
    rotation = 45*(totale%8)
    operations = ['Erosion (3,3)', 'Dilation (3,3)']

    print("Image:")
    print(rotate(image, angle=rotation, mode='nearest', reshape=False))
    print("\nOperation:")
    print(operations[totale%2])


def ex6(matricola):
    aux1 = list(matricola[:4])
    aux2 = list(matricola[4:]+matricola[1])
    aux = []
    for i in range(len(aux1)):
        aux.append(abs(int(aux1[i])-int(aux2[i])))
    vec = np.array(aux)
    print("Upsample by a factor of 2 (resulting in a 1x8 vector) via Bilinear Interpolation")
    print(vec)



def ex7(matricola):
    aux1 = list(matricola)
    aux2 = list(matricola)
    aux2.reverse()
    aux = []
    for i in range(len(aux1)):
        aux.append(abs(int(aux1[i])+int(aux2[i])))
    aux += list(matricola) + list(matricola[4:6])
    mat = [int(elem) for elem in aux]
    image = np.array(mat)
    image = np.reshape(image, (4,4))
    print("Image Mat:")
    print(image)


def ex8(matricola):
    path = "./Photos"
    aux = [int(elem) for elem in matricola]
    totale = 0
    for elem in aux:
        totale += elem
    lista = os.listdir(path)
    idx1 = totale%len(lista)
    idx2 = (totale-3)%len(lista)
    im1 = Image.open(path + '/' + lista[idx1]).convert('L')
    im2 = Image.open(path + '/' + lista[idx2]).convert('L')
    return im1, im2


def ex10(matricola):
    aux = [int(elem) for elem in matricola]
    totale = 0
    for elem in aux:
        totale += elem
    methods = ['HoG', 'Harris']
    print("Features Extractor:\n", methods[totale%2])


def draw_shape(image, shape, dims, color):
    """Draws a shape from the given specs."""
    # Get the center x, y and the size s
    x, y, s = dims
    if shape == 'square':
        image = cv2.rectangle(image, (x - s, y - s),
                              (x + s, y + s), color, -1)
    elif shape == "circle":
        image = cv2.circle(image, (x, y), s, color, -1)
    elif shape == "triangle":
        points = np.array([[(x, y - s),
                            (x - s / math.sin(math.radians(60)), y + s),
                            (x + s / math.sin(math.radians(60)), y + s),
                            ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)
    return image



def ex11(matricola):
    image1 = np.zeros((256,256,3))
    image2 = np.zeros((256,256,3))
    image3 = np.zeros((256,256,3))
    image4 = np.zeros((256,256,3))

    coords_1 = [[64,64], [64, 192], [192, 64], [192,192]]
    coords_2 = [[64,64], [64, 192], [192, 64], [192,192]]
    shapes = ["square", "circle", "triangle"]

    aux = [int(elem) for elem in matricola]
    totale = 0
    for elem in aux:
        totale += elem

    idx1 = totale%4
    idx2 = (totale-1)%4
    if idx1==idx2:
        idx1 = 0
        idx2 = 1
    shapeIdx1 = totale%3
    shapeIdx2 = (totale-1)%3

    dims1 = ((totale%3)+1)*12
    dims2 = ((totale%4)+1)*10

    color = (255,255,255)

    coords1_1 = [coords_1[idx1][0], coords_1[idx1][1], dims1]
    coords1_2 = [coords_2[idx2][0], coords_2[idx2][1], dims2]

    if coords_1[idx1][0]==64:
        coords_1[idx1][0] -= 10
    if coords_1[idx1][1]==64:
        coords_1[idx1][1] -= 10
    if coords_1[idx1][0]==192:
        coords_1[idx1][0] += 10
    if coords_1[idx1][1]==192:
        coords_1[idx1][1] += 10

    if coords_2[idx2][0]==64:
        coords_2[idx2][0] += 10
    if coords_2[idx2][1]==64:
        coords_2[idx2][1] += 10
    if coords_2[idx2][0]==192:
        coords_2[idx2][0] -= 10
    if coords_2[idx2][1]==192:
        coords_2[idx2][1] -= 10

    coords2_1 = [coords_1[idx1][0], coords_1[idx1][1], dims1]
    coords2_2 = [coords_2[idx2][0], coords_2[idx2][1], dims2]

    if coords_1[idx1][0]==54:
        coords_1[idx1][0] -= 10
    if coords_1[idx1][1]==54:
        coords_1[idx1][1] -= 10
    if coords_1[idx1][0]==202:
        coords_1[idx1][0] += 10
    if coords_1[idx1][1]==202:
        coords_1[idx1][1] += 10

    if coords_2[idx2][0]==74:
        coords_2[idx2][0] += 10
    if coords_2[idx2][1]==74:
        coords_2[idx2][1] += 10
    if coords_2[idx2][0]==182:
        coords_2[idx2][0] -= 10
    if coords_2[idx2][1]==182:
        coords_2[idx2][1] -= 10

    coords3_1 = [coords_1[idx1][0], coords_1[idx1][1], dims1]
    coords3_2 = [coords_2[idx2][0], coords_2[idx2][1], dims2]

    image1 = draw_shape(image1, shapes[shapeIdx1], coords1_1, color)
    image1 = draw_shape(image1, shapes[shapeIdx2], coords1_2, color)

    image2 = draw_shape(image2, shapes[shapeIdx1], coords2_1, color)
    image2 = draw_shape(image2, shapes[shapeIdx2], coords2_2, color)

    image3 = draw_shape(image3, shapes[shapeIdx1], coords3_1, color)
    image3 = draw_shape(image3, shapes[shapeIdx2], coords3_2, color)



    image4 = draw_shape(image4, shapes[shapeIdx1], coords1_1, color)
    image4 = draw_shape(image4, shapes[shapeIdx2], coords1_2, color)

    image4 = draw_shape(image4, shapes[shapeIdx1], coords2_1, color)
    image4 = draw_shape(image4, shapes[shapeIdx2], coords2_2, color)

    image4 = draw_shape(image4, shapes[shapeIdx1], coords3_1, color)
    image4 = draw_shape(image4, shapes[shapeIdx2], coords3_2, color)

    image1 = Image.fromarray(image1.astype(np.uint8)).convert('L')
    image2 = Image.fromarray(image2.astype(np.uint8)).convert('L')
    image3 = Image.fromarray(image3.astype(np.uint8)).convert('L')
    image4 = Image.fromarray(image4.astype(np.uint8)).convert('L')
    
    return image1, image2, image3, image4


def ex12(matricola):
    path = "./Distorted"
    aux = [int(elem) for elem in matricola]
    totale = 0
    for elem in aux:
        totale += elem
    lista = os.listdir(path)
    idx1 = totale%len(lista)
    idx2 = (totale-3)%len(lista)
    im1 = Image.open(path + '/' + lista[idx1]).convert('L')
    im2 = Image.open(path + '/' + lista[idx2]).convert('L')
    return im1, im2
