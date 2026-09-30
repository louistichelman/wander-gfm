#pragma once

#include <cstdint>
#include <random>

/// Uniform neighbor sample in CSR row ``[start, end)``.
/// When ``skip_prev`` and ``deg >= 2``, never returns ``prev`` (non-backtracking).
inline uint32_t sample_uniform_neighbor(
    const uint32_t *indices,
    uint32_t start,
    uint32_t end,
    bool skip_prev,
    uint32_t prev,
    std::mt19937 &gen)
{
    const uint32_t deg = end - start;
    if (deg == 0)
    {
        return prev;
    }
    std::uniform_int_distribution<uint32_t> dist(0, deg - 1);
    if (!skip_prev || deg == 1)
    {
        return indices[start + dist(gen)];
    }
    for (int attempt = 0; attempt < 16; ++attempt)
    {
        const uint32_t nxt = indices[start + dist(gen)];
        if (nxt != prev)
        {
            return nxt;
        }
    }
    std::uniform_int_distribution<uint32_t> dist_skip(0, deg - 2);
    uint32_t pick = dist_skip(gen);
    for (uint32_t z = start; z < end; ++z)
    {
        if (indices[z] == prev)
        {
            continue;
        }
        if (pick == 0)
        {
            return indices[z];
        }
        --pick;
    }
    return indices[start];
}
