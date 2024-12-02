#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
vector<vector<float>> getImages(const string& archivo, size_t max_images = SIZE_MAX) {
    cout << "Intentando abrir el archivo: " << archivo << endl;
    ifstream archivo_csv(archivo);

    if (!archivo_csv.is_open()) {
        cerr << "Error al abrir el archivo CSV de imágenes: " << archivo << endl;
        exit(EXIT_FAILURE); // Detener ejecución si no se abre el archivo
    }

    string linea;
    vector<vector<float>> datos;

    size_t count = 0;
    while (getline(archivo_csv, linea) && count < max_images) {
        vector<float> fila;
        istringstream ss(linea);
        string campo;

        while (getline(ss, campo, ',')) {
            float valor = stof(campo) / 255.0f; // Normalizar el valor entre 0 y 1
            fila.push_back(valor);
        }

        datos.push_back(fila);
        count++;
    }

    return datos;
}










/*
vector<vector<float>> getImages(const string& archivo) {
    cout << "Intentando abrir el archivo: " << archivo << endl;
    ifstream archivo_csv(archivo);

    if (!archivo_csv.is_open()) {
        cerr << "Error al abrir el archivo CSV de imagenes: " << archivo << endl;
        exit(EXIT_FAILURE); // Detener ejecución si no se abre el archivo
    }

    string linea;
    vector<vector<float>> datos;

    while (getline(archivo_csv, linea)) {
        vector<float> fila;
        istringstream ss(linea);
        string campo;

        while (getline(ss, campo, ',')) {
            float valor = stof(campo) / 255.0f; // Normalizar el valor entre 0 y 1
            fila.push_back(valor);
        }

        datos.push_back(fila);
    }

    return datos;
}


*/

vector<vector<int>> getLabels(const string& archivo, size_t max_images = SIZE_MAX) {
    cout << "Intentando abrir el archivo: " << archivo << endl;
    ifstream archivo_csv(archivo);

    if (!archivo_csv.is_open()) {
        cerr << "Error al abrir el archivo CSV de etiquetas: " << archivo << endl;
        exit(EXIT_FAILURE); // Detener ejecución si no se abre el archivo
    }

    string linea;
    vector<vector<int>> datos;

    size_t count = 0;
    while (getline(archivo_csv, linea) && count < max_images) {
        vector<int> fila(10, 0); // Crear un vector de 10 posiciones inicializado en 0
        int valor = stoi(linea); // Convertir la etiqueta en entero
        if (valor >= 0 && valor < 10) {
            fila[valor] = 1; // Asignar 1 en la posición correspondiente
        } else {
            cerr << "Etiqueta inválida encontrada: " << valor << endl;
            exit(EXIT_FAILURE);
        }

        datos.push_back(fila);
        count++;
    }

    return datos;
}
