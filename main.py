import gzip
import os
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import csv


def list_files(mnist_path):
    return [join(mnist_path, f) for f in os.listdir(mnist_path) if isfile(join(mnist_path, f))]


def get_images(mnist_path):
    x_train, y_train, x_test, y_test = None, None, None, None

    for f in list_files(mnist_path):
        if 'train-images' in f:
            with gzip.open(f, 'rb') as data:
                _ = int.from_bytes(data.read(4), 'big')
                num_images = int.from_bytes(data.read(4), 'big')
                rows = int.from_bytes(data.read(4), 'big')
                cols = int.from_bytes(data.read(4), 'big')
                train_images = data.read()
                x_train = np.frombuffer(train_images, dtype=np.uint8)
                x_train = x_train.reshape((num_images, rows, cols))
        elif 'train-labels' in f:
            with gzip.open(f, 'rb') as data:
                train_labels = data.read()[8:]
                y_train = np.frombuffer(train_labels, dtype=np.uint8)
        elif 't10k-images' in f:
            with gzip.open(f, 'rb') as data:
                _ = int.from_bytes(data.read(4), 'big')
                num_images = int.from_bytes(data.read(4), 'big')
                rows = int.from_bytes(data.read(4), 'big')
                cols = int.from_bytes(data.read(4), 'big')
                test_images = data.read()
                x_test = np.frombuffer(test_images, dtype=np.uint8)
                x_test = x_test.reshape((num_images, rows, cols))
        elif 't10k-labels' in f:
            with gzip.open(f, 'rb') as data:
                test_labels = data.read()[8:]
                y_test = np.frombuffer(test_labels, dtype=np.uint8)

    return x_train, y_train, x_test, y_test


mnist_path = './Mnist/'

Data = get_images(mnist_path)

# Convertir imágenes de entrenamiento y prueba a CSV
imagesToList = [image.flatten().tolist() for image in Data[0]]  # Entrenamiento
imagesToListTest = [image.flatten().tolist() for image in Data[2]]  # Prueba

csv_filename_train = 'mnist_train_images.csv'
with open(csv_filename_train, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(imagesToList)

csv_filename_test = 'mnist_test_images.csv'
with open(csv_filename_test, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(imagesToListTest)

# Convertir etiquetas de entrenamiento y prueba a CSV
LabelsToListTrain = [[label] for label in Data[1]]  # Entrenamiento
LabelsToListTest = [[label] for label in Data[3]]  # Prueba

csv_filename_labels_train = 'mnist_train_labels.csv'
with open(csv_filename_labels_train, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(LabelsToListTrain)

csv_filename_labels_test = 'mnist_test_labels.csv'
with open(csv_filename_labels_test, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(LabelsToListTest)


def plot_number(image):
    plt.imshow(image)
    plt.show()


# Ejemplo: graficar el número 908 del set de entrenamiento
# plot_number(Data[0][908])
