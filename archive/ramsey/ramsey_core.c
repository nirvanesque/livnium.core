/*
 * Ramsey Core - C-accelerated bitset-based clique checking and batch operations
 * 
 * This module provides fast, M5-optimized operations for Ramsey number search:
 * - Bitset-based edge coloring representation
 * - Bitwise clique checking (AND/POPCOUNT)
 * - Batch validation of thousands of omcubes
 * 
 * Designed to plug into the existing Python/Livnium meta-physics layer.
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#define MAX_VERTICES 128  // Support up to 128 vertices (adjustable)
#define MAX_BITSET_SIZE (MAX_VERTICES * MAX_VERTICES)  // n×n adjacency matrix
#define BITSET_WORDS ((MAX_BITSET_SIZE + 63) / 64)  // 64-bit words needed

// Bitset structure for adjacency matrix
typedef struct {
    uint64_t words[BITSET_WORDS];
} bitset_t;

// Initialize bitset to zero
static inline void bitset_init(bitset_t *bs) {
    memset(bs->words, 0, sizeof(bs->words));
}

// Set bit at position i
static inline void bitset_set(bitset_t *bs, int i) {
    bs->words[i / 64] |= (1ULL << (i % 64));
}

// Check if bit at position i is set
static inline bool bitset_get(const bitset_t *bs, int i) {
    return (bs->words[i / 64] & (1ULL << (i % 64))) != 0;
}

// Count set bits (POPCOUNT)
static inline int bitset_popcount(const bitset_t *bs) {
    int count = 0;
    for (int i = 0; i < BITSET_WORDS; i++) {
        count += __builtin_popcountll(bs->words[i]);
    }
    return count;
}

// AND two bitsets, store result in dest
static inline void bitset_and(bitset_t *dest, const bitset_t *a, const bitset_t *b) {
    for (int i = 0; i < BITSET_WORDS; i++) {
        dest->words[i] = a->words[i] & b->words[i];
    }
}

// Recursive helper for clique finding
static bool find_clique_recursive(const bitset_t *adj, int n, int k,
                                  const bitset_t *candidates, int candidate_count,
                                  int clique_size, int *clique) {
    if (clique_size == k) {
        return true;
    }
    if (clique_size + candidate_count < k) {
        return false;
    }
    
    // Try each candidate
    for (int i = 0; i < n; i++) {
        if (!bitset_get(candidates, i)) continue;
        
        // Check if i is connected to all vertices in current clique
        bool can_add = true;
        for (int j = 0; j < clique_size; j++) {
            if (!bitset_get(adj, clique[j] * n + i)) {
                can_add = false;
                break;
            }
        }
        
        if (can_add) {
            clique[clique_size] = i;
            
            // Build new candidate set (neighbors of i that are also candidates)
            bitset_t new_candidates;
            bitset_init(&new_candidates);
            bitset_t i_neighbors;
            bitset_init(&i_neighbors);
            
            for (int v = 0; v < n; v++) {
                if (bitset_get(adj, i * n + v)) {
                    bitset_set(&i_neighbors, v);
                }
            }
            
            bitset_and(&new_candidates, candidates, &i_neighbors);
            int new_count = bitset_popcount(&new_candidates);
            
            if (find_clique_recursive(adj, n, k, &new_candidates, new_count,
                                     clique_size + 1, clique)) {
                return true;
            }
        }
    }
    return false;
}

// Check if a clique of size k exists in the graph
// Uses bitwise operations for fast checking
static bool has_clique_bitset(const bitset_t *adj, int n, int k, int *clique_out) {
    if (k == 0) return false;
    if (k == 1) {
        // Any vertex is a 1-clique
        clique_out[0] = 0;
        return true;
    }
    
    // For each starting vertex
    for (int start = 0; start < n; start++) {
        bitset_t neighbors;
        bitset_init(&neighbors);
        
        // Get neighbors of start vertex
        for (int v = 0; v < n; v++) {
            if (v != start && bitset_get(adj, start * n + v)) {
                bitset_set(&neighbors, v);
            }
        }
        
        int neighbor_count = bitset_popcount(&neighbors);
        if (neighbor_count < k - 1) continue;
        
        // Recursive clique finding with bitsets
        int current_clique[MAX_VERTICES];
        current_clique[0] = start;
        
        if (find_clique_recursive(adj, n, k, &neighbors, neighbor_count, 1, current_clique)) {
            memcpy(clique_out, current_clique, k * sizeof(int));
            return true;
        }
    }
    
    return false;
}

// Python function: check_ramsey_coloring
// Input: numpy array of edge colors (shape: [num_edges], dtype: uint8)
//        0 = red, 1 = blue, 255 = uncolored
// Returns: (has_clique: bool, clique_vertices: array of int32)
static PyObject *check_ramsey_coloring(PyObject *self, PyObject *args) {
    PyArrayObject *edge_colors;
    int n, k;
    
    if (!PyArg_ParseTuple(args, "Oii", &edge_colors, &n, &k)) {
        return NULL;
    }
    
    if (n > MAX_VERTICES) {
        PyErr_Format(PyExc_ValueError, "n=%d exceeds MAX_VERTICES=%d", n, MAX_VERTICES);
        return NULL;
    }
    
    // Build red and blue adjacency bitsets
    bitset_t red_adj, blue_adj;
    bitset_init(&red_adj);
    bitset_init(&blue_adj);
    
    int num_edges = n * (n - 1) / 2;
    uint8_t *colors = (uint8_t *)PyArray_DATA(edge_colors);
    
    int edge_idx = 0;
    for (int u = 0; u < n; u++) {
        for (int v = u + 1; v < n; v++) {
            if (edge_idx >= num_edges) break;
            uint8_t color = colors[edge_idx];
            
            if (color == 0) {  // Red
                bitset_set(&red_adj, u * n + v);
                bitset_set(&red_adj, v * n + u);
            } else if (color == 1) {  // Blue
                bitset_set(&blue_adj, u * n + v);
                bitset_set(&blue_adj, v * n + u);
            }
            // 255 (uncolored) is ignored
            
            edge_idx++;
        }
    }
    
    // Check for cliques
    int clique[MAX_VERTICES];
    bool found = false;
    
    if (has_clique_bitset(&red_adj, n, k, clique)) {
        found = true;
    } else if (has_clique_bitset(&blue_adj, n, k, clique)) {
        found = true;
    }
    
    // Return result
    if (found) {
        npy_intp dims[1] = {k};
        PyArrayObject *clique_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
        int32_t *clique_data = (int32_t *)PyArray_DATA(clique_array);
        memcpy(clique_data, clique, k * sizeof(int32_t));
        return Py_BuildValue("(OO)", Py_True, (PyObject *)clique_array);
    } else {
        Py_INCREF(Py_None);
        return Py_BuildValue("(OO)", Py_False, Py_None);
    }
}

// Python function: batch_check_ramsey_colorings
// Input: numpy array of edge colorings (shape: [batch_size, num_edges], dtype: uint8)
// Returns: numpy array of bools (shape: [batch_size]) indicating validity
static PyObject *batch_check_ramsey_colorings(PyObject *self, PyObject *args) {
    PyArrayObject *edge_colorings;
    int n, k;
    
    if (!PyArg_ParseTuple(args, "Oii", &edge_colorings, &n, &k)) {
        return NULL;
    }
    
    if (n > MAX_VERTICES) {
        PyErr_Format(PyExc_ValueError, "n=%d exceeds MAX_VERTICES=%d", n, MAX_VERTICES);
        return NULL;
    }
    
    npy_intp *dims = PyArray_DIMS(edge_colorings);
    int batch_size = dims[0];
    int num_edges = n * (n - 1) / 2;
    
    // Create output array
    npy_intp out_dims[1] = {batch_size};
    PyArrayObject *results = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_BOOL);
    bool *results_data = (bool *)PyArray_DATA(results);
    
    uint8_t *colorings = (uint8_t *)PyArray_DATA(edge_colorings);
    
    // Process each coloring in batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Build red and blue adjacency bitsets
        bitset_t red_adj, blue_adj;
        bitset_init(&red_adj);
        bitset_init(&blue_adj);
        
        uint8_t *colors = colorings + batch_idx * num_edges;
        
        int edge_idx = 0;
        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                if (edge_idx >= num_edges) break;
                uint8_t color = colors[edge_idx];
                
                if (color == 0) {  // Red
                    bitset_set(&red_adj, u * n + v);
                    bitset_set(&red_adj, v * n + u);
                } else if (color == 1) {  // Blue
                    bitset_set(&blue_adj, u * n + v);
                    bitset_set(&blue_adj, v * n + u);
                }
                
                edge_idx++;
            }
        }
        
        // Check for cliques (we only need to know if valid, not the clique itself)
        int dummy_clique[MAX_VERTICES];
        bool has_red_clique = has_clique_bitset(&red_adj, n, k, dummy_clique);
        bool has_blue_clique = has_clique_bitset(&blue_adj, n, k, dummy_clique);
        
        results_data[batch_idx] = !(has_red_clique || has_blue_clique);
    }
    
    return (PyObject *)results;
}

// Method definitions
static PyMethodDef RamseyCoreMethods[] = {
    {"check_ramsey_coloring", check_ramsey_coloring, METH_VARARGS,
     "Check if a single Ramsey coloring is valid (no monochromatic k-clique).\n"
     "Args:\n"
     "  edge_colors: numpy array [num_edges] of uint8 (0=red, 1=blue, 255=uncolored)\n"
     "  n: number of vertices\n"
     "  k: clique size to avoid\n"
     "Returns:\n"
     "  (has_clique: bool, clique_vertices: array or None)"},
    
    {"batch_check_ramsey_colorings", batch_check_ramsey_colorings, METH_VARARGS,
     "Check validity of a batch of Ramsey colorings.\n"
     "Args:\n"
     "  edge_colorings: numpy array [batch_size, num_edges] of uint8\n"
     "  n: number of vertices\n"
     "  k: clique size to avoid\n"
     "Returns:\n"
     "  numpy array [batch_size] of bool (True = valid, False = invalid)"},
    
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef ramsey_core_module = {
    PyModuleDef_HEAD_INIT,
    "ramsey_core",
    "C-accelerated Ramsey number search operations using bitsets",
    -1,
    RamseyCoreMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_ramsey_core(void) {
    import_array();
    return PyModule_Create(&ramsey_core_module);
}

