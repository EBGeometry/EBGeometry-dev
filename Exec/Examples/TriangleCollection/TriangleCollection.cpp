// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   TriangleCollection.cpp
 * @brief  Generates N=20000 triangles in both AoS and SoA layouts,
 *         then measures performance of signed distance evaluation.
 * @author Robert Marskar 
 */

// Std includes
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

// EBGeometry includes
#include "EBGeometry.hpp"

int
main()
{
  using namespace EBGeometry;
  using MetaData = int;

  constexpr int N = 20000;

  // Random generator for sampling positions inside the unit cube.
  std::mt19937                         rng(12345);
  std::uniform_real_distribution<Real> dist(Real(0), Real(1));

  // Allocate AoS and SoA storage
  std::vector<Triangle<MetaData>> trianglesAoS;
  trianglesAoS.reserve(N);

  std::vector<Vec3>     soa_tn(N);
  std::vector<Vec3>     soa_vx1(N);
  std::vector<Vec3>     soa_vx2(N);
  std::vector<Vec3>     soa_vx3(N);
  std::vector<Vec3>     soa_vn1(N);
  std::vector<Vec3>     soa_vn2(N);
  std::vector<Vec3>     soa_vn3(N);
  std::vector<Vec3>     soa_en1(N);
  std::vector<Vec3>     soa_en2(N);
  std::vector<Vec3>     soa_en3(N);
  std::vector<MetaData> soa_metadata(N);

  // Populate both AoS and SoA with identical triangle data
  for (int i = 0; i < N; ++i) {
    // Random vertices in the unit cube
    Vec3 vx[3] = {Vec3(dist(rng), dist(rng), dist(rng)), Vec3(dist(rng), dist(rng), dist(rng)), Vec3(dist(rng), dist(rng), dist(rng))};

    // Simple fixed normal (these are mostly placeholders).
    Vec3 tn(Real(0), Real(0), Real(1));
    Vec3 vn[3] = {tn, tn, tn};
    Vec3 en[3] = {Vec3(Real(1), Real(0), Real(0)), Vec3(Real(0), Real(1), Real(0)), Vec3(Real(0), Real(0), Real(1))};

    MetaData md = i;

    trianglesAoS.emplace_back(tn, vx, vn, en, md);

    soa_tn[i]       = tn;
    soa_vx1[i]      = vx[0];
    soa_vx2[i]      = vx[1];
    soa_vx3[i]      = vx[2];
    soa_vn1[i]      = vn[0];
    soa_vn2[i]      = vn[1];
    soa_vn3[i]      = vn[2];
    soa_en1[i]      = en[0];
    soa_en2[i]      = en[1];
    soa_en3[i]      = en[2];
    soa_metadata[i] = md;
  }

  // Build AoS TriangleCollection
  Span<const Triangle<MetaData>>                spanAoS(trianglesAoS.data(), static_cast<int>(trianglesAoS.size()));
#if 0
  TriangleCollection<MetaData, LayoutType::AoS> collectionAoS(spanAoS);

  // Build SoA TriangleCollection
  Span<const Vec3>     tnSpan(soa_tn.data(), N);
  Span<const Vec3>     vx1Span(soa_vx1.data(), N);
  Span<const Vec3>     vx2Span(soa_vx2.data(), N);
  Span<const Vec3>     vx3Span(soa_vx3.data(), N);
  Span<const Vec3>     vn1Span(soa_vn1.data(), N);
  Span<const Vec3>     vn2Span(soa_vn2.data(), N);
  Span<const Vec3>     vn3Span(soa_vn3.data(), N);
  Span<const Vec3>     en1Span(soa_en1.data(), N);
  Span<const Vec3>     en2Span(soa_en2.data(), N);
  Span<const Vec3>     en3Span(soa_en3.data(), N);
  Span<const MetaData> metaSpan(soa_metadata.data(), N);

  TriangleCollection<MetaData, LayoutType::SoA>
    collectionSoA(tnSpan, vx1Span, vx2Span, vx3Span, vn1Span, vn2Span, vn3Span, en1Span, en2Span, en3Span, metaSpan);

  // Query point
  Vec3 query(dist(rng), dist(rng), dist(rng));

  // Time AoS version
  auto                                      t0         = std::chrono::steady_clock::now();
  Real                                      dAoS       = collectionAoS.value(query);
  auto                                      t1         = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> elapsedAoS = t1 - t0;

  // Time SoA version
  auto                                      t2         = std::chrono::steady_clock::now();
  Real                                      dSoA       = collectionSoA.value(query);
  auto                                      t3         = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> elapsedSoA = t3 - t2;

  // Print results
  std::cout << "Number of triangles: " << N << "\n";
  std::cout << "Query point: (" << query[0] << ", " << query[1] << ", " << query[2] << ")\n";
  std::cout << "AoS  distance: " << dAoS << "  (" << elapsedAoS.count() << " ms)\n";
  std::cout << "SoA  distance: " << dSoA << "  (" << elapsedSoA.count() << " ms)\n";

  double ratio = (elapsedAoS.count() > 0.0) ? (elapsedSoA.count() / elapsedAoS.count()) : 0.0;

  std::cout << "SoA/AoS runtime ratio: " << ratio << "\n";
#endif
  return 0;
}
