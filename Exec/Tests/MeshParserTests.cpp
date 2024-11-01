#include <cstdlib>
#include <iostream>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "EBGeometry.hpp"

using namespace EBGeometry;

TEST_CASE("CLEAN_STL_ASCII")
{
  const char* tmp = std::getenv("EBGEOMETRY_HOME");

  std::string EBGeometryHome = "invalid";
  if (tmp != NULL) {
    EBGeometryHome = std::string(tmp);
  }

  const std::string folder = EBGeometryHome + "/Meshes/Clean/STL/ASCII";

  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    std::cout << "Parsing ASCII file into DCEL mesh: " << entry.path() << std::endl;

    auto mesh = MeshParser::readIntoDCEL<short>(entry.path());

    CHECK(mesh->isManifold() == true);
    CHECK(EBGEOMETRY_ASSERTION_FAILURES == 0);

    mesh->freeMem();

    delete mesh;
  }
}

TEST_CASE("CLEAN_STL_BINARY")
{
  const char* tmp = std::getenv("EBGEOMETRY_HOME");

  std::string EBGeometryHome = "invalid";
  if (tmp != NULL) {
    EBGeometryHome = std::string(tmp);
  }

  const std::string folder = EBGeometryHome + "/Meshes/Clean/STL/Binary";

  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    std::cout << "Parsing binary file into DCEL mesh: " << entry.path() << std::endl;

    auto mesh = MeshParser::readIntoDCEL<short>(entry.path());

    CHECK(mesh->isManifold() == true);
    CHECK(EBGEOMETRY_ASSERTION_FAILURES == 0);

    mesh->freeMem();

    delete mesh;
  }
}
