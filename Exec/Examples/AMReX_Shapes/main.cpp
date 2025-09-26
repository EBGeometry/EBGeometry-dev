// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// AMReX includes
#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

// Our include
#include "../../../EBGeometry.hpp"

using namespace amrex;

using ImplicitFunction = EBGeometry::ImplicitFunction;
using Vec3             = EBGeometry::Vec3;

/*!
  @brief This is just an EBGeometry-exposed implicit function usable with AMReX.
  AMReX.
*/
class EBGeometryIF
{
public:
  /*!
    @brief Full constructor.
    @param[in] a_impFunc  Implicit function
    @param[in] a_flipSign Hook for swapping inside/outside.
  */
  EBGeometryIF(ImplicitFunction* a_impFunc)
  {
    m_impFunc = a_impFunc;
  }

  /*!
    @brief Copy constructor.
    @param[in] a_other Other ImplicitFunction.
  */
  EBGeometryIF(const EBGeometryIF& a_other)
  {
    this->m_impFunc = a_other.m_impFunc;
  }

  /*!
    @brief AMReX's implicit function definition. EBGeometry sign is opposite to AMReX'
  */
  Real
  operator()(AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
  {
    return -m_impFunc->value(Vec3(x, y, z));
  };

  /*!
    @brief Also an AMReX implicit function implementation
  */
  inline Real
  operator()(const RealArray& p) const noexcept
  {
    return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
  }

protected:
  /*!
    @brief EBGeometry implicit function.
  */
  ImplicitFunction* m_impFunc;
};

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);

  int n_cell          = 128;
  int max_grid_size   = 32;
  int whichGeom       = 0;
  int num_coarsen_opt = 0;

  std::string filename;

  // read parameters
  ParmParse pp;
  pp.query("n_cell", n_cell);
  pp.query("max_grid_size", max_grid_size);
  pp.query("which_geom", whichGeom);
  pp.query("num_coarsen_opt", num_coarsen_opt);

  Geometry geom;
  RealBox  rb;

  ImplicitFunction* func;
  if (whichGeom == 0) { // Sphere.
    rb = RealBox({-1, -1, -1}, {1, 1, 1});

    func = EBGeometry::createImpFunc<EBGeometry::SphereSDF, EBGeometry::MemoryLocation::Host>(Vec3::zero(), 0.25);
  }

  EBGeometryIF sdf(func);

  // Below here is only AMReX codes
  Array<int, AMREX_SPACEDIM> is_periodic{false, false, false};
  Geometry::Setup(&rb, 0, is_periodic.data());
  Box domain(IntVect(0), IntVect(n_cell - 1));
  geom.define(domain);

  auto gshop = EB2::makeShop(sdf);
  EB2::Build(gshop, geom, 0, 0, true, true, num_coarsen_opt);

  // Put some data
  MultiFab mf;
  {
    BoxArray boxArray(geom.Domain());
    boxArray.maxSize(max_grid_size);
    DistributionMapping dm{boxArray};

    std::unique_ptr<EBFArrayBoxFactory> factory =
      amrex::makeEBFabFactory(geom, boxArray, dm, {2, 2, 2}, EBSupport::full);

    mf.define(boxArray, dm, 1, 0, MFInfo(), *factory);
    mf.setVal(1.0);
  }

  EB_WriteSingleLevelPlotfile("plt", mf, {"rho"}, geom, 0.0, 0);

  amrex::Finalize();
}
