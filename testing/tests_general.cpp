
//-----------------------------------------------------------------------------
//
// This file generates the main() for the Catch2 tests.
//
// We compile it here separately to cut down on recompilation times for the main
// software components.
//
//-----------------------------------------------------------------------------

#include "tests_general.hpp"
#include <string>
#include <vector>

std::array<int, 2> get_rank_info()
{
#ifdef ASGARD_USE_MPI
  int num_ranks;
  auto status = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  assert(status == 0);
  int my_rank;
  status = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  assert(status == 0);
  return {my_rank, num_ranks};
#else
  return {0, 1};
#endif
}

options make_options(std::vector<std::string> const arguments)
{
  std::vector<char *> argv;
  argv.push_back(const_cast<char *>("asgard"));
  for (const auto &arg : arguments)
  {
    argv.push_back(const_cast<char *>(arg.data()));
  }
  argv.push_back(nullptr);
  return options(argv.size() - 1, argv.data());
}
