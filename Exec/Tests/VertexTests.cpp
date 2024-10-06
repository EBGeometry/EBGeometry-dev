#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

// These tests use a hard-coded pyramid DCEL watertight mesh given by five vertices and five faces.
//
//  v0 = (-1, -1,  0)
//  v1 = (+1, -1,  0)
//  v2 = (+1, +1,  0)
//  v3 = (-1, +1,  0)
//  v4 = ( 0,  0, +1)
//
//  There are five faces (with outward normal vectors) spanned by the following vertices:
//
//  f0 = v0, v1, v2, v3
//  f1 = v0, v1, v4
//  f2 = v1, v2, v4
//  f3 = v2, v3, v4
//  f4 = v3, v0, v4
//
//  Edges are spanned as follows:
//  e0,  e1,  e2,  e3 span the inside of f0
//  e4,  e5,  e6      span the inside of f1
//  e7,  e8,  e9      span the inside of f2
//  e10, e11, e12     span the inside of f3
//  e13, e14, e15     span the inside of f4

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

  vertices[0].define(x0, Vec3::zero(), &edges[0]);
  vertices[1].define(x1, Vec3::zero(), &edges[1]);
  vertices[2].define(x2, Vec3::zero(), &edges[2]);
  vertices[3].define(x3, Vec3::zero(), &edges[3]);
  vertices[4].define(x4, Vec3::zero(), &edges[0]);

  delete[] vertices;
  delete[] edges;
  delete[] faces;
}
