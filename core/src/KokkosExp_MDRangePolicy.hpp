/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP
#define KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP

#include <initializer_list>

#include<impl/KokkosExp_Host_IterateTile.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel.hpp>

#if defined( __CUDACC__ ) && defined( KOKKOS_HAVE_CUDA )
#include<Cuda/KokkosExp_Cuda_IterateTile.hpp>
#endif


namespace Kokkos { namespace Experimental { namespace Impl { 

// ------------------------------------------------------------------ //
// Tags
struct BeginTag {};
struct EndTag {};
struct TileTag {};

// ConstArray
template <typename Type, int Rank, typename Tag = void >
struct ConstArray
{
  using type = Type;
  using tag  = Tag;

  template <typename... Args>
  static constexpr
  typename std::enable_if< (sizeof...(Args) < Rank-1), ConstArray>::type
  fill(Type const & value, Args const &... args)
  {
    return fill(value, value, args...);
  }

  template <typename... Args>
  static constexpr
  typename std::enable_if< (sizeof...(Args) == Rank-1), ConstArray>::type
  fill(Type const & value, Args const &... args)
  {
    return ConstArray{value, args...};
  }


  static constexpr int rank = Rank;
  static constexpr int size() { return Rank; }

  template <int I>
  constexpr Type get() const
  {
    static_assert( (I < Rank) && (I >= 0), "Error: Index out of range");
    return m_values[I];
  }

  constexpr Type operator[](int i) const
  { return m_values[i]; }

  template <typename... Values>
  constexpr ConstArray( Values &&... values )
    : m_values{ static_cast<Type>(values)... }
  {
    static_assert( sizeof...(Values) == Rank
                 , "Error: Number of arguments not equal to the Rank" );
  }

  Type m_values[Rank];
};

// are_integrals
template <typename... Args>
struct are_integrals;

template <typename Arg, typename... Args>
struct are_integrals<Arg, Args...>
  : public std::integral_constant<bool, (std::is_integral<Arg>::value && are_integrals<Args...>::value) >
{};

template <typename Arg>
struct are_integrals<Arg>
  : public std::is_integral<Arg>
{};

template <>
struct are_integrals<>
  : public std::false_type
{};


template < typename ... Args >
struct common_index_type {
  using type = typename std::remove_cv< typename std::common_type< Args... >::type >::type;
  static_assert( std::is_integral<type>::value, "MDRange Error: Common argument type is not an integral type" );
};


} } } //end  namespace Kokkos namespace Experimental namespace Impl 



// ------------------------------------------------------------------ //

namespace Kokkos { namespace Experimental {

// ------------------------------------------------------------------ //

// Begin
template <typename... Args>
inline constexpr
Impl::ConstArray< typename std::common_type<Args...>::type, sizeof...(Args), Impl::BeginTag >
Begin(Args &&... args)
{
//  static_assert( Impl::are_integrals<Args...>::value , "Error: Argument not an integral type" );
//  using return_type = Impl::ConstArray< typename std::common_type<Args...>::type, sizeof...(Args), Impl::BeginTag >;
  using return_type = Impl::ConstArray< typename Impl::common_index_type<Args...>::type, sizeof...(Args), Impl::BeginTag >;
  return return_type( std::forward<Args>(args)... );
}

// End
template <typename... Args>
inline constexpr
Impl::ConstArray< typename std::common_type<Args...>::type, sizeof...(Args), Impl::EndTag >
End(Args &&... args)
{
//  static_assert( std::is_integral< common_type >::value , "Error: Argument not an integral type" );
//  using return_type = Impl::ConstArray< typename std::common_type<Args...>::type, sizeof...(Args), Impl::EndTag >;
  using return_type = Impl::ConstArray< typename Impl::common_index_type<Args...>::type, sizeof...(Args), Impl::EndTag >;
  return return_type( std::forward<Args>(args)... );
}

// Tile
template <typename... Args>
inline constexpr
Impl::ConstArray< typename std::common_type<Args...>::type, sizeof...(Args), Impl::TileTag >
Tile(Args &&... args)
{
//  static_assert( Impl::are_integrals<Args...>::value , "Error: Argument not an integral type" );
//  using return_type = Impl::ConstArray< typename std::common_type<Args...>::type, sizeof...(Args), Impl::TileTag >;
  using return_type = Impl::ConstArray< typename Impl::common_index_type<Args...>::type, sizeof...(Args), Impl::TileTag >;
  return return_type( std::forward<Args>(args)... );
}


enum class Iterate
{
  Default, // Default for the device
  Left,    // Left indices stride fastest
  Right,   // Right indices stride fastest
};

template <typename ExecSpace>
struct default_outer_direction
{
  using type = Iterate;
  static constexpr Iterate value = Iterate::Right;
};

template <typename ExecSpace>
struct default_inner_direction
{
  using type = Iterate;
  static constexpr Iterate value = Iterate::Right;
};


// Iteration Pattern
template < unsigned N
         , Iterate OuterDir = Iterate::Default
         , Iterate InnerDir = Iterate::Default
         >
struct Rank
{
  static_assert( N != 0u, "Kokkos Error: rank 0 undefined");
  static_assert( N != 1u, "Kokkos Error: rank 1 is not a multi-dimensional range");
  static_assert( N  < 7u, "Kokkos Error: Unsupported rank...");

  using iteration_pattern = Rank<N, OuterDir, InnerDir>;

  static constexpr int rank = N;
  static constexpr Iterate outer_direction = OuterDir;
  static constexpr Iterate inner_direction = InnerDir;
};


// multi-dimensional iteration pattern
// ImplMDRangePolicy will be returned by the constexor MDRangePolicy functions
// Traits template parameter and inheritance is actually Kokkos::Impl::PolicyTraits< stuff >
template <typename Begin, typename End, typename Tile, typename Traits>
struct ImplMDRangePolicy
  : public Traits
{
  static_assert( std::is_same< typename Begin::tag, Impl::BeginTag>::value, "ImplMDRangePolicy Error: Not a Begin" );
  static_assert( std::is_same< typename End::tag, Impl::EndTag>::value, "ImplMDRangePolicy Error: Not a End" );
  static_assert( std::is_same< typename Tile::tag, Impl::TileTag>::value, "ImplMDRangePolicy Error: Not a Tile" );

  using traits = Traits; //Kokkos::Impl::PolicyTraits<Properties ...>;
  using range_policy = RangePolicy< typename traits::execution_space
                                  , typename traits::schedule_type
                                  , typename traits::work_tag
                                  , typename traits::index_type
                                  , typename traits::iteration_pattern
                                  >;

  using impl_range_policy = RangePolicy< typename traits::execution_space
                                       , typename traits::schedule_type
                                       , typename traits::index_type
                                       > ;

  static_assert( !std::is_same<typename traits::iteration_pattern,void>::value
               , "Kokkos Error: MD iteration pattern not defined" );

  using iteration_pattern   = typename traits::iteration_pattern;
  using work_tag            = typename traits::work_tag;

  static constexpr int rank = iteration_pattern::rank;

  static constexpr int outer_direction = static_cast<int> (
      (iteration_pattern::outer_direction != Iterate::Default)
    ? iteration_pattern::outer_direction
    : default_outer_direction< typename traits::execution_space>::value );

  static constexpr int inner_direction = static_cast<int> (
      iteration_pattern::inner_direction != Iterate::Default
    ? iteration_pattern::inner_direction
    : default_inner_direction< typename traits::execution_space>::value ) ;


  // Ugly ugly workaround intel 14 not handling scoped enum correctly
  static constexpr int Right = static_cast<int>( Iterate::Right );
  static constexpr int Left  = static_cast<int>( Iterate::Left );

  using index_type  = typename traits::index_type;
  using array_index_type = long;
//  using point_type  = Kokkos::Array<array_index_type,rank>; //was index_type
//  using tile_type   = Kokkos::Array<array_index_type,rank>;
  using point_type  = Begin; //will remove point_type once compiling...
  using begin_type  = Begin;
  using end_type    = End;
  using tile_type   = Tile;
  // If point_type or tile_type is not templated on a signed integral type (if it is unsigned), 
  // then if user passes in intializer_list of runtime-determined values of 
  // signed integral type that are not const will receive a compiler error due 
  // to an invalid case for implicit conversion - 
  // "conversion from integer or unscoped enumeration type to integer type that cannot represent all values of the original, except where source is a constant expression whose value can be stored exactly in the target type"
  // This would require the user to either pass a matching index_type parameter
  // as template parameter to the MDRangePolicy or static_cast the individual values

  ImplMDRangePolicy( begin_type const& lower, end_type const& upper, tile_type const& tile, traits const& traits_input = traits{} )
    : m_lower(lower)
    , m_upper(upper)
    , m_tile(tile)
    , m_num_tiles(1)
  {
    // Host
    if ( true
       #if defined(KOKKOS_HAVE_CUDA)
         && !std::is_same< typename traits::execution_space, Kokkos::Cuda >::value
       #endif
       )
    {
      index_type span;
      for (int i=0; i<rank; ++i) {
        span = upper[i] - lower[i];
        if ( m_tile[i] <= 0 ) {
          if (  (inner_direction == Right && (i < rank-1))
              || (inner_direction == Left && (i > 0)) )
          {
            m_tile[i] = 2;
          }
          else {
            m_tile[i] = span;
          }
        }
        m_tile_end[i] = static_cast<index_type>((span + m_tile[i] -1) / m_tile[i]);
        m_num_tiles *= m_tile_end[i];
      }
    }
    #if defined(KOKKOS_HAVE_CUDA)
    else // Cuda
    {
      index_type span;
      for (int i=0; i<rank; ++i) {
        span = upper[i] - lower[i];
        if ( m_tile[i] <= 0 ) {
          // TODO: determine what is a good default tile size for cuda
          // may be rank dependent
          if (  (inner_direction == Right && (i < rank-1))
              || (inner_direction == Left && (i > 0)) )
          {
            m_tile[i] = 2;
          }
          else {
            m_tile[i] = 16;
          }
        }
        m_tile_end[i] = static_cast<index_type>((span + m_tile[i] - 1) / m_tile[i]);
        m_num_tiles *= m_tile_end[i];
      }
      index_type total_tile_size_check = 1;
      for (int i=0; i<rank; ++i) {
        total_tile_size_check *= m_tile[i];
      }
      if ( total_tile_size_check > 1024 ) {
        printf(" Tile dimensions exceed Cuda limits\n");
        Kokkos::abort(" Cuda ExecSpace Error: MDRange tile dims exceed maximum number of threads per block - choose smaller tile dims");
        //Kokkos::Impl::throw_runtime_exception( " Cuda ExecSpace Error: MDRange tile dims exceed maximum number of threads per block - choose smaller tile dims");
      }
    }
    #endif
  }

  begin_type m_lower;
  end_type m_upper;
  tile_type m_tile;
  tile_type m_tile_end;
  /*
  point_type m_lower;
  point_type m_upper;
  tile_type  m_tile;
  point_type m_tile_end;
  */
  index_type m_num_tiles;
};

// MDRangePolicy function - returns an ImplMDRangePolicy

template <typename BeginType, typename EndType, typename TileType, typename Traits>
constexpr 
/*
typename std::enable_if< ( std::is_same< typename BeginType::Tag, Impl::BeginTag>::value 
                        && std::is_same< typename EndType::Tag, Impl::EndTag>::value 
                        && std::is_same< typename TileType::Tag, Impl::TileTag>::value )
                       , ImplMDRangePolicy<BeginType, EndType, TileType, Traits>
                       >::type
*/
ImplMDRangePolicy<BeginType, EndType, TileType, Traits>
MDRangePolicy(BeginType const& begin, EndType const& end, TileType const& tile, Traits const& traits)
{
  return ImplMDRangePolicy<BeginType, EndType, TileType, Traits> {begin, end, tile};
}

template <typename BeginType, typename EndType, typename TileType>
constexpr 
typename std::enable_if< ( std::is_same< typename BeginType::Tag, Impl::BeginTag>::value 
                        && std::is_same< typename EndType::Tag, Impl::EndTag>::value 
                        && std::is_same< typename TileType::Tag, Impl::TileTag>::value )
                       , ImplMDRangePolicy<BeginType, EndType, TileType, Kokkos::Impl::PolicyTraits<> >
                       >::type
MDRangePolicy(BeginType const& begin, EndType const& end, TileType const& tile)
{
  using traits_type = Kokkos::Impl::PolicyTraits<>;
  return ImplMDRangePolicy<BeginType, EndType, TileType, traits_type> {begin, end, tile};
}

//template <typename EndType, typename... Traits>
template <typename EndType, typename Traits>
constexpr
typename std::enable_if< std::is_same< typename EndType::tag, Impl::EndTag>::value
                       , ImplMDRangePolicy< Impl::ConstArray<int,EndType::rank, Impl::BeginTag>, EndType, Impl::ConstArray<int, EndType::rank, Impl::TileTag>, Traits>
                       >::type
MDRangePolicy(EndType const& end, Traits const& traits)
{
  using return_type = ImplMDRangePolicy< Impl::ConstArray<int,EndType::rank, Impl::BeginTag>, EndType, Impl::ConstArray<int, EndType::rank, Impl::TileTag>, Traits>;
  using begin_type  = Impl::ConstArray<int, EndType::rank, Impl::BeginTag>;
  using tile_type  = Impl::ConstArray<int, EndType::rank, Impl::TileTag>;
  return return_type{ begin_type::fill(0), end, tile_type::fill(1) };
}

//template <typename EndType, typename TileType, typename... Traits>
template <typename EndType, typename TileType, typename Traits>
constexpr
typename std::enable_if< (  std::is_same< typename EndType::tag, Impl::EndTag>::value
                         && std::is_same< typename TileType::tag, Impl::TileTag>::value )
                       , ImplMDRangePolicy< Impl::ConstArray<int,EndType::rank, Impl::BeginTag>, EndType, TileType, Traits>
                       >::type
MDRangePolicy(EndType const& end, TileType const& tile, Traits const& traits)
{
  using return_type = ImplMDRangePolicy< Impl::ConstArray<int,EndType::rank, Impl::BeginTag>, EndType, TileType, Traits>;
  using array_type  = Impl::ConstArray<int, EndType::rank, Impl::BeginTag>;
  return return_type{ array_type::fill(0), end, tile };
}


//template <typename BeginType, typename EndType, typename... Traits>
template <typename BeginType, typename EndType, typename Traits>
constexpr
typename std::enable_if< (  std::is_same< typename BeginType::tag, Impl::BeginTag>::value
                         && std::is_same< typename EndType::tag, Impl::EndTag>::value )
                       , ImplMDRangePolicy< BeginType, EndType, Impl::ConstArray<int, EndType::rank, Impl::TileTag>, Traits>
                       >::type
MDRangePolicy(BeginType const& begin, EndType const& end, Traits const& traits)
{
  using return_type = ImplMDRangePolicy< BeginType, EndType, Impl::ConstArray<int, EndType::rank, Impl::TileTag>, Traits>;
  using array_type  = Impl::ConstArray<int, EndType::rank, Impl::TileTag>;
  return return_type{ begin, end, array_type::fill(1) };
}

//no traits passed in
template <typename BeginType, typename EndType, typename TileType>
constexpr 
ImplMDRangePolicy<BeginType, EndType, TileType, Kokkos::Impl::PolicyTraits<> >
MDRangePolicy(BeginType const& begin, EndType const& end, TileType const& tile)
{
  using traits_type = Kokkos::Impl::PolicyTraits<>;
  return ImplMDRangePolicy<BeginType, EndType, TileType, traits_type> {begin, end, tile};
}

template <typename EndType>
constexpr
typename std::enable_if< std::is_same< typename EndType::tag, Impl::EndTag>::value
                       , ImplMDRangePolicy< Impl::ConstArray<int,EndType::rank, Impl::BeginTag>, EndType, Impl::ConstArray<int, EndType::rank, Impl::TileTag>, Kokkos::Impl::PolicyTraits<> >
                       >::type
MDRangePolicy(EndType const& end)
{
  using traits_type = Kokkos::Impl::PolicyTraits<>;
  using return_type = ImplMDRangePolicy< Impl::ConstArray<int,EndType::rank, Impl::BeginTag>, EndType, Impl::ConstArray<int, EndType::rank, Impl::TileTag>, traits_type>;
  using begin_type  = Impl::ConstArray<int, EndType::rank, Impl::BeginTag>;
  using tile_type   = Impl::ConstArray<int, EndType::rank, Impl::TileTag>;
  return return_type{ begin_type::fill(0), end, tile_type::fill(1) };
}

template <typename EndType, typename TileType>
constexpr
typename std::enable_if< (  std::is_same< typename EndType::tag, Impl::EndTag>::value
                         && std::is_same< typename TileType::tag, Impl::TileTag>::value )
                       , ImplMDRangePolicy< Impl::ConstArray<int,EndType::rank, Impl::BeginTag>, EndType, TileType, Kokkos::Impl::PolicyTraits<> >
                       >::type
MDRangePolicy(EndType const& end, TileType const& tile)
{
  using traits_type = Kokkos::Impl::PolicyTraits<>;
  using return_type = ImplMDRangePolicy< Impl::ConstArray<int, EndType::rank, Impl::BeginTag>, EndType, TileType, traits_type>;
  using array_type  = Impl::ConstArray<int, EndType::rank, Impl::BeginTag>;
  return return_type{ array_type::fill(0), end, tile };
}


template <typename BeginType, typename EndType>
constexpr
typename std::enable_if< (  std::is_same< typename BeginType::tag, Impl::BeginTag>::value
                         && std::is_same< typename EndType::tag, Impl::EndTag>::value )
                       , ImplMDRangePolicy< BeginType, EndType, Impl::ConstArray<int, EndType::rank, Impl::TileTag>, Kokkos::Impl::PolicyTraits<> >
                       >::type
MDRangePolicy(BeginType const& begin, EndType const& end)
{
  using traits_type = Kokkos::Impl::PolicyTraits<>;
  using return_type = ImplMDRangePolicy< BeginType, EndType, Impl::ConstArray<int, EndType::rank, Impl::TileTag>, traits_type >;
  using array_type  = Impl::ConstArray<int, EndType::rank, Impl::TileTag>;
  return return_type{ begin, end, array_type::fill(1) };
}


// ------------------------------------------------------------------ //

// ------------------------------------------------------------------ //
//md_parallel_for
// ------------------------------------------------------------------ //
template <typename MDRange, typename Functor, typename Enable = void>
void md_parallel_for( MDRange const& range
                    , Functor const& f
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::MDFunctor<MDRange, Functor, void> g(range, f);

  //using range_policy = typename MDRange::range_policy;
  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_for( range_policy(0, range.m_num_tiles).set_chunk_size(1), g, str );
}

template <typename MDRange, typename Functor>
void md_parallel_for( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::MDFunctor<MDRange, Functor, void> g(range, f);

  //using range_policy = typename MDRange::range_policy;
  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_for( range_policy(0, range.m_num_tiles).set_chunk_size(1), g, str );
}

// Cuda specialization
#if defined( __CUDACC__ ) && defined( KOKKOS_HAVE_CUDA )
template <typename MDRange, typename Functor>
void md_parallel_for( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag> closure(range, f);
  closure.execute();
}

template <typename MDRange, typename Functor>
void md_parallel_for( MDRange const& range
                    , Functor const& f
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag> closure(range, f);
  closure.execute();
}
#endif
// ------------------------------------------------------------------ //

// ------------------------------------------------------------------ //
//md_parallel_reduce
// ------------------------------------------------------------------ //
template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce( MDRange const& range
                    , Functor const& f
                    , ValueType & v
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::MDFunctor<MDRange, Functor, ValueType> g(range, f, v);

  //using range_policy = typename MDRange::range_policy;
  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_reduce( str, range_policy(0, range.m_num_tiles).set_chunk_size(1), g, v );
}

template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , ValueType & v
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::MDFunctor<MDRange, Functor, ValueType> g(range, f, v);

  //using range_policy = typename MDRange::range_policy;
  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_reduce( str, range_policy(0, range.m_num_tiles).set_chunk_size(1), g, v );
}

// Cuda - parallel_reduce not implemented yet
/*
template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce( MDRange const& range
                    , Functor const& f
                    , ValueType & v
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag> closure(range, f, v);
  closure.execute();
}

template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , ValueType & v
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag> closure(range, f, v);
  closure.execute();
}
*/

}} // namespace Kokkos::Experimental

#endif //KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP

