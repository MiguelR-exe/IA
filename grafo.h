#include <vector>
#include <forward_list>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <stack>
#include <queue>
#include <unordered_set>
#include <random>
#include <numeric>
#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>

using namespace std;


float funcion_activacion(float sumatoria){
    return 1 / (1 + exp(-sumatoria));
}

float funcion_activacion_derivada(float x) {
    float sig = funcion_activacion(x);
    return sig * (1.0 - sig);
}



random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dis(0.0, 1.0f);



template <typename KeyType, typename ValueType, typename WeightType>
class graph {
    using AdjancentType = pair<size_t, WeightType>;
    using ItemType = pair<size_t, ValueType>;
    using AdjancentListType = forward_list<AdjancentType>;

    unordered_map<KeyType, ItemType> vertices;
    vector<AdjancentListType> buckets;

public:
    graph() = default;

    bool push_vertex(KeyType key, ValueType value) {
       if (vertices.emplace(key, make_pair(buckets.size(), value)).second) {
            buckets.emplace_back();
            return true;
        }
        return false;
    }

    void push_edge(const tuple<KeyType, KeyType, WeightType>& edge) {

        auto [u,v,w] = edge;

        if (vertices.find(u) == vertices.end()) push_vertex(u, ValueType{});
        if (vertices.find(v) == vertices.end()) push_vertex(v, ValueType{});

        auto idx_u = vertices[u].first;
        auto idx_v = vertices[v].first;

        buckets[idx_u].emplace_front(idx_v, w);
        buckets[idx_v].emplace_front(idx_u, w);
    }

    size_t size() const {
        return vertices.size();
    }

    ValueType get_vertex(KeyType key) const {
        auto it = vertices.find(key);
        return it != vertices.end() ? it->second.second : ValueType{};
    }

    bool update_vertex(KeyType key, ValueType newValue) {
        auto it = vertices.find(key);
        if (it != vertices.end()) {
            it->second.second = newValue;
            return true;
        }
        return false;
    }

    bool update_edge_weight(size_t u, size_t v, WeightType new_weight) {
        auto it_u = vertices.find(u);
        auto it_v = vertices.find(v);
        if (it_u != vertices.end() && it_v != vertices.end()) {
            size_t idx_u = it_u->second.first;
            size_t idx_v = it_v->second.first;
            for (auto& adj : buckets[idx_u]) {
                if (adj.first == idx_v) {
                    adj.second = new_weight;
                    return true;
                }
            }
        }
        return false;
    }

    WeightType get_edge(KeyType u, KeyType v) const {
        auto it_u = vertices.find(u);
        auto it_v = vertices.find(v);
        if (it_u != vertices.end() && it_v != vertices.end()) {
            size_t idx_u = it_u->second.first;
            size_t idx_v = it_v->second.first;

            auto it = find_if(buckets[idx_u].begin(), buckets[idx_u].end(),
                                   [idx_v](const auto& adj) { return adj.first == idx_v; });
            if (it != buckets[idx_u].end()) return it->second;
        }
        return numeric_limits<WeightType>::infinity();
    }
};

template<typename T = int, typename U = float, typename V = float>
void give_vertex(graph<T, U, V>& g1, int n_vertex) {
    for (int i = 0; i < n_vertex; i++) {
        T key = static_cast<T>(i);      // Asegurar que 'i' sea del tipo KeyType
        U value = static_cast<U>(dis(gen)); // Asegurar que el valor sea del tipo ValueType
        g1.push_vertex(key, value);
    }
}

template<typename T=int,typename U=float, typename V=float>
void give_weight_and_relations(graph<T, U, V>& g, const vector<int>& capas) {
    int n_capas = capas.size();
    for (int i = 0; i < n_capas - 1; ++i) {
        int start = accumulate(capas.begin(), capas.begin() + i, 0);
        int end = start + capas[i];

        int start_next = end;
        int end_next = start_next + capas[i + 1];

        for (int j = start; j < end; ++j) {
            for (int k = start_next; k < end_next; ++k) {
                g.push_edge({j, k, dis(gen)});
            }
        }
    }
}

template<typename T = int, typename U = float, typename V = float>
void changeEntradas(graph<T, U, V>& g, const vector<float>& entradas) {
    for (size_t i = 0; i < entradas.size(); ++i) {
        g.update_vertex(i, entradas[i]);
    }
}

template<typename T = int, typename U = float, typename V = float>
void frontPropagation(graph<T, U, V>& g, const vector<int>& capas) {
    int n_capas = capas.size();
    for (int i = 0; i < n_capas - 1; ++i) {
        int start = accumulate(capas.begin(), capas.begin() + i, 0);
        int end = start + capas[i];

        int start_next = end;
        int end_next = start_next + capas[i + 1];

        for (int k = start_next; k < end_next; ++k) {
            float sumatoria = 0.0f;
            for (int j = start; j < end; ++j) {
                sumatoria += g.get_edge(j, k) * g.get_vertex(j);
            }
            g.update_vertex(k, funcion_activacion(sumatoria));
        }
    }
}

template<typename T = int, typename U = float, typename V = float>
void backPropagation(graph<T, U, V> &g1, vector<int> capas, const vector<int> &salidas_esperadas, float learning_rate) {
    int n_capas = capas.size();

    int start_salida = accumulate(capas.begin(), capas.end() - 1, 0);
    int end_salida = start_salida + capas.back();

    vector<float> errores(g1.size(), 0.0f);

    for (int i = start_salida; i < end_salida; ++i) {
        float salida_actual = g1.get_vertex(i);
        errores[i] = (salida_actual - salidas_esperadas[i - start_salida]) * funcion_activacion_derivada(salida_actual);
    }

    for (int i = n_capas - 2; i >= 0; --i) {
        int start = accumulate(capas.begin(), capas.begin() + i, 0);
        int end = start + capas[i];
        int start_next = accumulate(capas.begin(), capas.begin() + i + 1, 0);
        int end_next = start_next + capas[i + 1];

        for (int j = start; j < end; ++j) {
            float gradiente = 0.0f;

            for (int k = start_next; k < end_next; ++k) {
                float peso = g1.get_edge(j, k);
                gradiente += errores[k] * peso;
                float nuevo_peso = peso - learning_rate * errores[k] * g1.get_vertex(j);
                g1.update_edge_weight(j, k, nuevo_peso);
            }
            errores[j] = gradiente * funcion_activacion_derivada(g1.get_vertex(j));
        }
    }
    for (int i = 0; i < g1.size(); ++i) {
        g1.update_vertex(i, g1.get_vertex(i) - learning_rate * errores[i]);
    }
}

template<typename T = int, typename U = float, typename V = float>
graph<T, U, V> start_graph(const vector<int>& capas) {
    graph<T, U, V> g;
    give_vertex(g, accumulate(capas.begin(), capas.end(), 0));
    give_weight_and_relations(g, capas);
    return g;
}
