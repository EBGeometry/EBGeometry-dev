/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/**
 * @file   EBGeometry_SFC.hpp
 * @brief  Declaration of various space-filling curves
 * @author Robert Marskar
 */

#ifndef EBGeometry_SFC
#define EBGeometry_SFC

// Std includes
#include <cstdint>

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_Macros.hpp"

// Our includes
namespace EBGeometry::SFC {

  /**
   * @brief Alias for SFC code
   */
  using Code = uint64_t;

  /**
   * @brief Maximum available bits. Using same number = 21 for all dimensions (strictly speaking, we could use 32 bits in 2D and all 64 bits in 1D).
   */
  static constexpr unsigned int ValidBits = 21;

  /**
   * @brief Maximum permitted span along any spatial coordinate.
   */
  static constexpr Code ValidSpan = (static_cast<uint64_t>(1) << ValidBits) - 1;

  /**
   * @brief Simple 3D cell index for usage with SFC codes.
   */
  class Index
  {
  public:
    /**
     * @brief Default constructor. Sets the zero index.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Index() noexcept = default;

    /**
     * @brief Full constructor. Create a cell index.
     * @param[in] x Index in x-direction
     * @param[in] y Index in y-direction
     * @param[in] z Index in z-direction	
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Index(unsigned int x, unsigned int y, unsigned int z) noexcept;

    /**
     * @brief Copy constructor.
     * @param[in] a_index Other index
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Index(const Index& a_index) noexcept = default;

    /**
     * @brief Move constructor.
     * @param[in] a_index Other index
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Index(Index&& a_index) noexcept = default;

    /**
     * @brief Destructor.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~Index() noexcept = default;

    /**
     * @brief Copy assignment.
     * @param[in] a_index Other index
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Index&
    operator=(const Index& a_index) noexcept = default;

    /**
     * @brief Move assignment.
     * @param[in, out] a_index Other index
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Index&
    operator=(Index&& a_index) noexcept = default;

    /**
     * @brief Get the index
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    unsigned int
    operator[](int a_dir) const noexcept;

  protected:
    /**
     * @brief Logical cell index.
     */
    unsigned int m_indices[3]{0, 0, 0};
  };

#if __cplusplus >= 202002L
  /**
   * @brief Encodable SFC concept -- class must have a static function static uint64_t encode(const Index&). This is the main interface for SFCs
   */
  template <typename S>
  concept Encodable = requires(const SFC::Index& point, const SFC::Code code) {
    { S::encode(point) } -> std::same_as<SFC::Code>;
    { S::decode(code) } -> std::same_as<SFC::Index>;
  };
#endif

  /**
   * @brief Implementation of the Morton SFC
   */
  struct Morton
  {
    /**
     * @brief Encode an input point into a Morton index with a 64-bit representation.
     * @param[in] a_point
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static uint64_t
    encode(const Index& a_point) noexcept;

    /**
     * @brief Decode the 64-bit Morton code into an Index.
     * @param[in] a_code Morton code
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static Index
    decode(const uint64_t& a_code) noexcept;

  protected:
    /**
     * @brief Mask for magic-bits encoding of 3D Morton code
     */
    static constexpr uint_fast64_t Mask_64[6]{
      0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff, 0x100f00f00f00f00f, 0x10c30c30c30c30c3, 0x1249249249249249};
  };

  /**
   * @brief Implementation of a nested index SFC.
   * @details The SFC is encoded by the code = i + j * N + k * N * N in 3D, where i,j,k are the block indices.
   */
  struct Nested
  {
    /**
     * @brief Encode the input point into the SFC code.
     * @param[in] a_point
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static uint64_t
    encode(const Index& a_point) noexcept;

    /**
     * @brief Decode the 64-bit SFC code into an Index.
     * @param[in] a_code SFC code.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static Index
    decode(const uint64_t& a_code) noexcept;
  };
} // namespace EBGeometry::SFC

#if __cplusplus >= 202002L
static_assert(EBGeometry::SFC::Encodable<EBGeometry::SFC::Morton>);
static_assert(EBGeometry::SFC::Encodable<EBGeometry::SFC::Nested>);
#endif

#include "EBGeometry_SFCImplem.hpp" // NOLINT

#endif
