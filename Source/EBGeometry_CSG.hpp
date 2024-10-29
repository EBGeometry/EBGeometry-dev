/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_CSG.hpp
  @brief  Declaration of a CSG operations for implicit functions. 
  @author Robert Marskar
*/

#ifndef EBGeometry_CSG
#define EBGeometry_CSG

// Our includes
#include "EBGeometry_AnalyticDistanceFunctions.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_ImplicitFunction.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {

  /*!
    @brief CSG union. Computes the minimum value of all input primitives.
    @details User inputs the implicit functions.
  */
  class UnionIF : public ImplicitFunction
  {
  public:
    /*!
      @brief Full constructor. Computes the CSG union
      @param[in] a_implicitFunctions List of primitives
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    UnionIF(const ImplicitFunction* const a_f1, const ImplicitFunction* const a_f2) noexcept
    {
      m_f1 = a_f1;
      m_f2 = a_f2;
    }

    /*!
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    virtual ~UnionIF() noexcept
    {}

    /*!
      @brief Implicit function for a union.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept override
    {
      return EBGeometry::min(m_f1->value(a_point), m_f2->value(a_point));
    }

  protected:
    /*!
      @brief First implicit function
    */
    const ImplicitFunction* m_f1;

    /*!
      @brief Second implicit function
    */
    const ImplicitFunction* m_f2;
  };

  /*!
    @brief Factory method for building UnionIF objects on the host or device.
  */
  class UnionIFFactory : public ImplicitFunctionFactory<UnionIF>
  {
  public:
    EBGEOMETRY_GPU_HOST
    UnionIFFactory(ImplicitFunction** a_f1, ImplicitFunction** a_f2) noexcept
    {
      m_f1 = a_f1;
      m_f2 = a_f2;
    };

    EBGEOMETRY_GPU_HOST
    [[nodiscard]] virtual std::shared_ptr<UnionIF>
    buildOnHost() const noexcept override
    {
      return std::make_shared<UnionIF>(*m_f1, *m_f2);
    }

#ifdef EBGEOMETRY_ENABLE_GPU
    /*!
      @brief Build implicit function on the device
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] virtual UnionIF**
    buildOnDevice() const noexcept override
    {
      UnionIF** csgUnion = allocateImplicitFunctionOnDevice<UnionIF>();

      createImplicitFunctionOnDevice<<<1, 1>>>(csgUnion, m_f1, m_f2);

      return csgUnion;
    };
#endif

  protected:
    ImplicitFunction** m_f1;

    ImplicitFunction** m_f2;
  };

} // namespace EBGeometry

#endif
