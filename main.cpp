#include <iostream>
#include "grafo.h"
#include "mnist_vector.h"

int main() {

    vector<int> capas = {784, 128, 10}; // 784 entradas (28x28 imágenes), 128 neuronas ocultas, 10 salidas

    auto red_neuronal = start_graph<>(capas);


    size_t max_imagenes = 300;

    string archivo_imagenes = "D:/Proyecto-final-progra/Normalizacion_mnist/mnist_train_images.csv";
    string archivo_etiquetas = "D:/Proyecto-final-progra/Normalizacion_mnist/mnist_train_labels.csv";
    auto imagenes_entrenamiento = getImages(archivo_imagenes, max_imagenes  );
    auto etiquetas_entrenamiento = getLabels(archivo_etiquetas);

    vector<vector<float>> etiquetas_float;
    for (const auto& etiqueta : etiquetas_entrenamiento) {
        etiquetas_float.emplace_back(etiqueta.begin(), etiqueta.end());
    }

    // Entrenamiento de la red
    float tasa_aprendizaje = 0.01f; // Tasa de aprendizaje
    size_t epocas = 1;            // Número de épocas de entrenamiento

    for (size_t epoca = 0; epoca < epocas; ++epoca) {
        for (size_t i = 0; i < imagenes_entrenamiento.size(); ++i) {

            cout << "Entrenando imagen " << i + 1 << " de " << imagenes_entrenamiento.size() << endl;
            // Actualizar las entradas de la red
            changeEntradas(red_neuronal, imagenes_entrenamiento[i]);

            // Propagación hacia adelante
            frontPropagation(red_neuronal, capas);

            // Backpropagation para ajustar pesos
            backPropagation(red_neuronal, capas, etiquetas_entrenamiento[i], tasa_aprendizaje);
        }
        cout << "Epoca " << epoca + 1 << " completada." << endl;
    }



    size_t max_tests = 100;
    string archivo_imagenes_prueba = "D:/Proyecto-final-progra/Normalizacion_mnist/mnist_test_images.csv";
    string archivo_etiquetas_prueba = "D:/Proyecto-final-progra/Normalizacion_mnist/mnist_test_labels.csv";
    auto imagenes_prueba = getImages(archivo_imagenes_prueba, max_tests);
    auto etiquetas_prueba = getLabels(archivo_etiquetas_prueba);

    size_t correctos = 0;


    for (size_t i = 0; i < imagenes_prueba.size(); ++i) {
        cout << "Testeando imagen " << i + 1 << " de " << imagenes_prueba.size() << endl;
        changeEntradas(red_neuronal, imagenes_prueba[i]);
        frontPropagation(red_neuronal, capas);

        int inicio_salida = accumulate(capas.begin(), capas.end() - 1, 0);
        int prediccion = -1;
        float max_valor = -1;

        for (int j = 0; j < 10; ++j) {
            float valor = red_neuronal.get_vertex(inicio_salida + j);
            if (valor > max_valor) {
                max_valor = valor;
                prediccion = j;
            }
        }
        int etiqueta_real = distance(etiquetas_prueba[i].begin(),
                                     max_element(etiquetas_prueba[i].begin(), etiquetas_prueba[i].end()));
        if (prediccion == etiqueta_real) ++correctos;
    }

    cout << "Precisión: " << (static_cast<float>(correctos) / imagenes_prueba.size()) * 100.0f << "%" << endl;

    return 0;
}
