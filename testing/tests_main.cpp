#define CATCH_CONFIG_RUNNER
#include "../src/distribution.hpp"
#include "catch.hpp"

int main(int argc, char *argv[])
{

#ifdef ASGARD_USE_MPI
  auto status = MPI_Init(NULL, NULL);
  std::cout << status << '\n';
  //assert(status == 0);
#endif
  int const result = Catch::Session().run(argc, argv);
#ifdef ASGARD_USE_MPI
  status = MPI_Finalize();

  //std::cout << "yo" << '\n';
  //assert(status == 0);
#endif

  return result;
}
