#pragma once

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

/// Spawn/join worker threads for one parallel_for, then let them exit.
/// A process-lifetime pool (Aug 31) kept 16–128 threads alive during GPU
/// eval and delayed CUDA launches (~12 s/batch CoDEx vs ~1 s on 30 Aug).
///
/// Do not spawn hardware_concurrency() (often 128–256 on the node) into a
/// Slurm cgroup of 16 CPUs — that spawn tax was the leftover walks gap.
/// Cap to GRAPH_WALKER_NUM_THREADS, else OMP_NUM_THREADS,
/// else SLURM_CPUS_PER_TASK, else the process CPU affinity.

static unsigned read_positive_env(const char *name)
{
    const char *v = std::getenv(name);
    if (v == nullptr || *v == '\0')
    {
        return 0;
    }
    char *end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || n <= 0)
    {
        return 0;
    }
    return static_cast<unsigned>(n);
}

static unsigned affinity_count()
{
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
    {
        const int n = CPU_COUNT(&set);
        if (n > 0)
        {
            return static_cast<unsigned>(n);
        }
    }
#endif
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 8u : hw;
}

static unsigned choose_n_workers(unsigned nb_elements)
{
    unsigned n = read_positive_env("GRAPH_WALKER_NUM_THREADS");
    if (n == 0)
    {
        n = read_positive_env("OMP_NUM_THREADS");
    }
    if (n == 0)
    {
        n = read_positive_env("SLURM_CPUS_PER_TASK");
    }
    if (n == 0)
    {
        n = affinity_count();
    }
    if (n < 1)
    {
        n = 1;
    }
    if (nb_elements > 0 && n > nb_elements)
    {
        n = nb_elements;
    }
    return n;
}

/// @param[in] nb_elements : size of your for loop
/// @param[in] functor(start, end) :
/// your function processing a sub chunk of the for loop.
/// "start" is the first index to process (included) until the index "end"
/// (excluded)
/// @code
///     for(int i = start; i < end; ++i)
///         computation(i);
/// @endcode
/// @param use_threads : enable / disable threads.
static void parallel_for(unsigned nb_elements,
                         std::function<void(int start, int end)> functor,
                         bool use_threads = true)
{
    if (nb_elements == 0)
    {
        return;
    }

    const unsigned nb_threads = choose_n_workers(nb_elements);
    if (!use_threads || nb_threads <= 1)
    {
        functor(0, static_cast<int>(nb_elements));
        return;
    }

    const unsigned batch_size = nb_elements / nb_threads;
    const unsigned batch_remainder = nb_elements % nb_threads;

    std::vector<std::thread> my_threads;
    my_threads.reserve(nb_threads);
    for (unsigned i = 0; i < nb_threads; ++i)
    {
        const int start = static_cast<int>(i * batch_size);
        my_threads.emplace_back(functor, start, start + static_cast<int>(batch_size));
    }

    const int rem_start = static_cast<int>(nb_threads * batch_size);
    if (batch_remainder > 0)
    {
        functor(rem_start, rem_start + static_cast<int>(batch_remainder));
    }

    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}

#define PARALLEL_FOR_BEGIN(nb_elements) parallel_for(nb_elements, [&](int start, int end){ for(int i = start; i < end; ++i)
#define PARALLEL_FOR_END() \
    })
