#pragma once
#include "tensors.hpp"
#include <vector>

// -----------------------------------------------------------------------------
// permutations
// this components's purpose is to provide utilities used in
// construction of the element table
// -----------------------------------------------------------------------------

// Permutations counters

int count_eq_permutations(int const num_dims, int const limit);

int count_eq_permutations_multi(fk::vector<int> const &levels,
                                int const num_dims, int const limit);

int count_leq_permutations(int const num_dims, int const limit);

int count_leq_permutations_multi(fk::vector<int> const &levels,
                                 int const num_dims, int const limit);

int count_max_permutations(int const num_dims, int const limit);

// Permutations builders

fk::matrix<int>
get_eq_permutations(int const num_dims, int const limit, bool const order_by_n);

fk::matrix<int> get_leq_permutations(int const num_dims, int const limit,
                                     bool const order_by_n);

fk::matrix<int> get_max_permutations(int const num_dims, int const limit,
                                     bool const last_index_decreasing);

using list_set = std::vector<fk::vector<int>>;

// Index counter

int count_leq_max_indices(list_set lists, int const num_dims, int const max_sum,
                          int const max_val);

// Index finder

fk::matrix<int> get_leq_max_indices(list_set lists, int const num_dims,
                                    int const max_sum, int const max_val);
