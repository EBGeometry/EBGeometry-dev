#include <cstdlib>
#include <iostream>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "EBGeometry.hpp"

using namespace EBGeometry;

const char*       tmp            = std::getenv("EBGEOMETRY_HOME");
const std::string EBGeometryHome = (tmp == nullptr) ? "invalid" : std::string(tmp);
const std::string cleanSTLASCII  = EBGeometryHome + "/Meshes/Clean/STL/ASCII";
const std::string cleanSTLBinary = EBGeometryHome + "/Meshes/Clean/STL/Binary";
const std::string cleanPLYASCII  = EBGeometryHome + "/Meshes/Clean/PLY/ASCII";
const std::string cleanPLYBinary = EBGeometryHome + "/Meshes/Clean/PLY/Binary";

TEST_CASE("CLEAN_STL_ASCII_SOUP")
{
  for (const auto& entry : std::filesystem::directory_iterator(cleanSTLASCII)) {
    std::cout << "Parsing ASCII file into polygon soup: " << entry.path() << std::endl;

    auto soup = MeshParser::readIntoSoup<short>(entry.path());

    CHECK(EBGEOMETRY_ASSERTION_FAILURES == 0);
  }
}

TEST_CASE("CLEAN_STL_BINARY_SOUP")
{
  for (const auto& entry : std::filesystem::directory_iterator(cleanSTLBinary)) {
    std::cout << "Parsing binary file into polygon soup: " << entry.path() << std::endl;

    auto soup = MeshParser::readIntoSoup<short>(entry.path());

    CHECK(EBGEOMETRY_ASSERTION_FAILURES == 0);
  }
}

TEST_CASE("CLEAN_STL_ASCII_DCEL")
{
  for (const auto& entry : std::filesystem::directory_iterator(cleanSTLASCII)) {
    std::cout << "Parsing ASCII file into DCEL mesh: " << entry.path() << std::endl;

    auto mesh = MeshParser::readIntoDCEL<short>(entry.path());

    CHECK(mesh->isManifold() == true);
    CHECK(EBGEOMETRY_ASSERTION_FAILURES == 0);

    mesh->freeMem();

    delete mesh;
  }
}

TEST_CASE("CLEAN_STL_BINARY_DCEL")
{
  for (const auto& entry : std::filesystem::directory_iterator(cleanSTLBinary)) {
    std::cout << "Parsing binary file into DCEL mesh: " << entry.path() << std::endl;

    auto mesh = MeshParser::readIntoDCEL<short>(entry.path());

    CHECK(mesh->isManifold() == true);
    CHECK(EBGEOMETRY_ASSERTION_FAILURES == 0);

    mesh->freeMem();

    delete mesh;
  }
}

TEST_CASE("CLEAN_PLY_ASCII_SOUP")
{
  for (const auto& entry : std::filesystem::directory_iterator(cleanPLYASCII)) {
    std::cout << "Parsing ASCII file into polygon soup: " << entry.path() << std::endl;

    auto soup = MeshParser::readIntoSoup<short>(entry.path());

    CHECK(EBGEOMETRY_ASSERTION_FAILURES == 0);
  }
}

TEST_CASE("CLEAN_PLY_ASCII_DCEL")
{
  for (const auto& entry : std::filesystem::directory_iterator(cleanPLYASCII)) {
    std::cout << "Parsing ASCII file into DCEL mesh: " << entry.path() << std::endl;

    auto mesh = MeshParser::readIntoDCEL<short>(entry.path());

    CHECK(mesh->isManifold() == true);
    CHECK(EBGEOMETRY_ASSERTION_FAILURES == 0);

    mesh->freeMem();

    delete mesh;
  }
}
