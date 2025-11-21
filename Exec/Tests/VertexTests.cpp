// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

TEST_CASE("Vertex_BasisVectors")
{

  const auto x0 = Vec3(-1, -1, 0.);
  const auto x1 = Vec3(+1, -1, 0);
  const auto x2 = Vec3(+1, +1, 0);
  const auto x3 = Vec3(-1, +1, 0);
  const auto x4 = Vec3(0, 0, +1);

  constexpr int numVertices = 5;
  constexpr int numEdges    = 16;
  constexpr int numFaces    = 4;

  auto vertices = new EBGeometry::DCEL::Vertex<>[numVertices];
  auto edges    = new EBGeometry::DCEL::Edge<>[numEdges];
  auto faces    = new EBGeometry::DCEL::Face<>[numFaces];

  vertices[0] = DCEL::Vertex<>(x0, Vec3::zero(), 0);
  vertices[0] = DCEL::Vertex<>(x1, Vec3::zero(), 1);
  vertices[0] = DCEL::Vertex<>(x2, Vec3::zero(), 2);
  vertices[0] = DCEL::Vertex<>(x3, Vec3::zero(), 3);
  vertices[0] = DCEL::Vertex<>(x4, Vec3::zero(), 4);

  delete[] vertices;
  delete[] edges;
  delete[] faces;
}
