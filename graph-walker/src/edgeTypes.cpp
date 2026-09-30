#include <algorithm>
#include <random>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "edgeTypes.hpp"
#include "threading.hpp"

namespace py = pybind11;

// CSR columns are sorted by destination (see wander.to_csr_tensor).
static void append_csr_matches(
    uint32_t prev,
    uint32_t curr,
    const uint32_t *indptr,
    const uint32_t *indices,
    const uint32_t *data,
    uint32_t dir_not_loop,
    bool homogeneous,
    std::vector<uint32_t> &types,
    std::vector<uint32_t> &dirs)
{
    const uint32_t start = indptr[prev];
    const uint32_t end = indptr[prev + 1];
    const uint32_t *first = indices + start;
    const uint32_t *last = indices + end;
    const uint32_t *lo = std::lower_bound(first, last, curr);
    const uint32_t dir = (curr == prev) ? 2u : dir_not_loop;
    for (const uint32_t *p = lo; p != last && *p == curr; ++p)
    {
        types.push_back(homogeneous ? 0u : data[p - indices]);
        dirs.push_back(dir);
    }
}

static void assign_match(
    uint32_t *edge_type,
    uint32_t *backwards,
    size_t slot,
    const std::vector<uint32_t> &types,
    const std::vector<uint32_t> &dirs,
    std::mt19937 &generator)
{
    const size_t n = types.size();
    if (n == 0)
    {
        edge_type[slot] = static_cast<uint32_t>(-1);
        backwards[slot] = static_cast<uint32_t>(-1);
        return;
    }
    size_t idx = 0;
    if (n > 1)
    {
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        idx = dist(generator);
    }
    edge_type[slot] = types[idx];
    backwards[slot] = dirs[idx];
}

std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>> parseEdgeTypesAndDirections(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indptr_edge_type,
    py::array_t<uint32_t> _indices_edge_type,
    py::array_t<uint32_t> _data_edge_type,
    py::array_t<uint32_t> _indptr_edge_type_transposed,
    py::array_t<uint32_t> _indices_edge_type_transposed,
    py::array_t<uint32_t> _data_edge_type_transposed,
    size_t seed,
    bool homogeneous)
{
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    uint32_t *indptr_edge_type = (uint32_t *)_indptr_edge_type.request().ptr;
    uint32_t *indices_edge_type = (uint32_t *)_indices_edge_type.request().ptr;
    uint32_t *data_edge_type = (uint32_t *)_data_edge_type.request().ptr;
    uint32_t *indptr_edge_type_transposed = (uint32_t *)_indptr_edge_type_transposed.request().ptr;
    uint32_t *indices_edge_type_transposed = (uint32_t *)_indices_edge_type_transposed.request().ptr;
    uint32_t *data_edge_type_transposed = (uint32_t *)_data_edge_type_transposed.request().ptr;

    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    py::array_t<uint32_t> _edge_type({shape, walkLen});
    uint32_t *edge_type = (uint32_t *)_edge_type.request().ptr;
    py::array_t<uint32_t> _backwards({shape, walkLen});
    uint32_t *backwards = (uint32_t *)_backwards.request().ptr;

    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        std::vector<uint32_t> types;
        std::vector<uint32_t> dirs;
        types.reserve(4);
        dirs.reserve(4);
        for (size_t k = 0; k < walkLen; k++)
        {
            const size_t slot = i * walkLen + k;
            const uint32_t value = walks[slot];
            const bool restart = restarts[slot];
            if (k > 0 && !restart)
            {
                const uint32_t prev = walks[i * walkLen + k - 1];
                types.clear();
                dirs.clear();
                append_csr_matches(
                    prev, value, indptr_edge_type, indices_edge_type, data_edge_type,
                    0, homogeneous, types, dirs);
                append_csr_matches(
                    prev, value, indptr_edge_type_transposed, indices_edge_type_transposed,
                    data_edge_type_transposed, 1, homogeneous, types, dirs);
                assign_match(edge_type, backwards, slot, types, dirs, generator);
            }
            else
            {
                edge_type[slot] = static_cast<uint32_t>(-1);
                backwards[slot] = static_cast<uint32_t>(-1);
            }
        }
    }
    PARALLEL_FOR_END();

    return {_edge_type, _backwards};
}

std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>> parseEdgeTypesAndDirectionsNeighbors(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors,
    py::array_t<uint32_t> _indptr_edge_type,
    py::array_t<uint32_t> _indices_edge_type,
    py::array_t<uint32_t> _data_edge_type,
    py::array_t<uint32_t> _indptr_edge_type_transposed,
    py::array_t<uint32_t> _indices_edge_type_transposed,
    py::array_t<uint32_t> _data_edge_type_transposed,
    size_t seed,
    bool homogeneous)
{
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info neighborsBuf = _neighbors.request();
    bool *neighbors = (bool *)neighborsBuf.ptr;

    uint32_t *indptr_edge_type = (uint32_t *)_indptr_edge_type.request().ptr;
    uint32_t *indices_edge_type = (uint32_t *)_indices_edge_type.request().ptr;
    uint32_t *data_edge_type = (uint32_t *)_data_edge_type.request().ptr;
    uint32_t *indptr_edge_type_transposed = (uint32_t *)_indptr_edge_type_transposed.request().ptr;
    uint32_t *indices_edge_type_transposed = (uint32_t *)_indices_edge_type_transposed.request().ptr;
    uint32_t *data_edge_type_transposed = (uint32_t *)_data_edge_type_transposed.request().ptr;

    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    py::array_t<uint32_t> _edge_type({shape, walkLen});
    uint32_t *edge_type = (uint32_t *)_edge_type.request().ptr;
    py::array_t<uint32_t> _backwards({shape, walkLen});
    uint32_t *backwards = (uint32_t *)_backwards.request().ptr;

    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        size_t predecessor = static_cast<size_t>(-1);
        std::vector<uint32_t> types;
        std::vector<uint32_t> dirs;
        types.reserve(4);
        dirs.reserve(4);

        for (size_t k = 0; k < walkLen; k++)
        {
            const size_t slot = i * walkLen + k;
            const uint32_t value = walks[slot];
            const bool restart = restarts[slot];
            const bool neighbor = neighbors[slot];

            if (restart || k == 0)
            {
                edge_type[slot] = static_cast<uint32_t>(-1);
                backwards[slot] = static_cast<uint32_t>(-1);
                predecessor = value;
            }
            else
            {
                types.clear();
                dirs.clear();
                append_csr_matches(
                    static_cast<uint32_t>(predecessor), value,
                    indptr_edge_type, indices_edge_type, data_edge_type,
                    0, homogeneous, types, dirs);
                append_csr_matches(
                    static_cast<uint32_t>(predecessor), value,
                    indptr_edge_type_transposed, indices_edge_type_transposed,
                    data_edge_type_transposed, 1, homogeneous, types, dirs);
                assign_match(edge_type, backwards, slot, types, dirs, generator);
                if (!neighbor)
                {
                    predecessor = value;
                }
            }
        }
    }
    PARALLEL_FOR_END();

    return {_edge_type, _backwards};
}
