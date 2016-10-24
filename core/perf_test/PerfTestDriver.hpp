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

#include <iostream>
#include <string>

// mfh 06 Jun 2013: This macro doesn't work like one might thing it
// should.  It doesn't take the template parameter DeviceType and
// print its actual type name; it just literally prints out
// "DeviceType".  I've worked around this below without using the
// macro, so I'm commenting out the macro to avoid compiler complaints
// about an unused macro.

// #define KOKKOS_MACRO_IMPL_TO_STRING( X ) #X
// #define KOKKOS_MACRO_TO_STRING( X )  KOKKOS_MACRO_IMPL_TO_STRING( X )

//------------------------------------------------------------------------

namespace Test {

enum { NUMBER_OF_TRIALS = 5 };

template< class DeviceType , class LayoutType >
void run_test_mdrange( int exp_beg , int exp_end, const char deviceTypeName[], int range_offset = 0,  int tile_offset = 0 )
// exp_beg = 6 => 2^6 = 64 is starting range length
{
  std::string label_mdrange ;
  label_mdrange.append( "\"MDRange< double , " );
  label_mdrange.append( deviceTypeName );
  label_mdrange.append( " >\"" );

  std::string label_range_col2 ;
  label_range_col2.append( "\"RangeCol2< double , " );
  label_range_col2.append( deviceTypeName );
  label_range_col2.append( " >\"" );

  std::string label_range_col_all ;
  label_range_col_all.append( "\"RangeColAll< double , " );
  label_range_col_all.append( deviceTypeName );
  label_range_col_all.append( " >\"" );

  if ( std::is_same<LayoutType, Kokkos::LayoutRight>::value) {
    std::cout << "Performance tests for MDRange Layout Right" << std::endl;
  } else {
    std::cout << "Performance tests for MDRange Layout Left\n" << std::endl;
  }


#define MDRANGE_PERFORMANCE_OUTPUT_VERBOSE 0
  for (int i = exp_beg ; i < exp_end ; ++i) {
    const int range_length = (1<<i) + range_offset;

    std::cout << "--------------------------------------------------------------\n"
      << "MDRange Test:  range bounds: " << range_length << " , " << range_length << " , " << range_length 
      << "\n--------------------------------------------------------------"
      << std::endl;

    int t0_min = 0, t1_min = 0, t2_min = 0;
    double seconds_min = 0.0;

    // Test 1: The MDRange in full
    {
    int t0 = 1, t1 = 1, t2 = 1;
    int counter = 1;
#if !defined(KOKKOS_HAVE_CUDA)
    int min_bnd = 8;
    int tfast = range_length;
#else
    int min_bnd = 2;
    int tfast = 32;
#endif
    while ( tfast >= min_bnd ) {
      int tmid = min_bnd;
      while ( tmid < tfast ) { 
        t0 = min_bnd;
        t1 = tmid;
        t2 = tfast;
        int t2_rev = min_bnd;
        int t1_rev = tmid;
        int t0_rev = tfast;

#if defined(KOKKOS_HAVE_CUDA)
        //Note: Product of tile sizes must be < 1024 for Cuda
        if ( t0*t1*t2 > 1024 ) {
          printf("  Exceeded Cuda tile limits; onto next range set\n\n");
          break;
        }
#endif

        // Run 1 with tiles LayoutRight style
        const double seconds_1 = MultiDimRangePerf3D< DeviceType , double , LayoutType >::test_multi_index(range_length,range_length,range_length, t0, t1, t2) ;

#if MDRANGE_PERFORMANCE_OUTPUT_VERBOSE
        std::cout << label_mdrange
          << " , " << t0 << " , " << t1 << " , " << t2
          << " , " << seconds_1
          << std::endl ;
#endif

        if ( counter == 1 ) {
          seconds_min = seconds_1;
          t0_min = t0;
          t1_min = t1;
          t2_min = t2;
        } 
        else {
          if ( seconds_1 < seconds_min ) 
          { 
            seconds_min = seconds_1; 
            t0_min = t0;
            t1_min = t1;
            t2_min = t2;
          }
        }

        // Run 2 with tiles LayoutLeft style
        const double seconds_1rev = MultiDimRangePerf3D< DeviceType , double , LayoutType >::test_multi_index(range_length,range_length,range_length, t0_rev, t1_rev, t2_rev) ;

#if MDRANGE_PERFORMANCE_OUTPUT_VERBOSE
        std::cout << label_mdrange
          << " , " << t0_rev << " , " << t1_rev << " , " << t2_rev
          << " , " << seconds_1rev
          << std::endl ;
#endif

        if ( seconds_1rev < seconds_min ) 
        { 
          seconds_min = seconds_1rev; 
          t0_min = t0_rev;
          t1_min = t1_rev;
          t2_min = t2_rev;
        }

        ++counter;
        tmid <<= 1;
      } //end inner while
      tfast >>=1;
    } //end outer while

    std::cout << "\n"
      << "--------------------------------------------------------------\n"
      << label_mdrange
      << "\n Min values "
      << "\n Range length per dim (3D): " << range_length
      << "\n TileDims:  " << t0_min << " , " << t1_min << " , " << t2_min
      << "\n Min time: " << seconds_min
      << "\n---------------------------------------------------------------"
      << std::endl ;
    } //end scope

#if !defined(KOKKOS_HAVE_CUDA)
  int t0c_min = 0, t1c_min = 0, t2c_min = 0;
  double seconds_min_c = 0.0;
  int counter = 1;
  {
    int min_bnd = 8;
    // Test 1_c: MDRange with 0 for 'inner' tile dim; this case will utilize the full span in that direction, should be similar to Collapse<2>
    if ( std::is_same<LayoutType, Kokkos::LayoutRight>::value ) {
      for ( unsigned int T0 = min_bnd; T0 < range_length; T0<<=1 ) {
        for ( unsigned int T1 = min_bnd; T1 < range_length; T1<<=1 ) {
          const double seconds_c = MultiDimRangePerf3D< DeviceType , double , LayoutType >::test_multi_index(range_length,range_length,range_length, T0, T1, 0) ;

#if MDRANGE_PERFORMANCE_OUTPUT_VERBOSE
          std::cout << " MDRange LR with '0' tile - collapse-like \n"
          << label_mdrange
          << " , " << T0 << " , " << T1 << " , " << range_length
          << " , " << seconds_c
          << std::endl ;
#endif

          t2c_min = range_length;
          if ( counter == 1 ) {
            seconds_min_c = seconds_c;
            t0c_min = T0;
            t1c_min = T1;
          } 
          else {
            if ( seconds_c < seconds_min_c ) 
            { 
              seconds_min = seconds_c; 
              t0c_min = T0;
              t1c_min = T1;
            }
          }
          ++counter;
        }
      }
    }
    else {
      for ( unsigned int T1 = min_bnd; T1 <= range_length; T1<<=1 ) {
        for ( unsigned int T2 = min_bnd; T2 <= range_length; T2<<=1 ) {
          const double seconds_c = MultiDimRangePerf3D< DeviceType , double , LayoutType >::test_multi_index(range_length,range_length,range_length, 0, T1, T2) ;

#if defined( MDRANGE_PERFORMANCE_OUTPUT_VERBOSE )
          std::cout << " MDRange LL with '0' tile - collapse-like \n"
          << label_mdrange
          << " , " <<range_length << " < " << T1 << " , " << T2
          << " , " << seconds_c
          << std::endl ;
#endif


          t0c_min = range_length;
          if ( counter == 1 ) {
            seconds_min_c = seconds_c;
            t1c_min = T1;
            t2c_min = T2;
          } 
          else {
            if ( seconds_c < seconds_min_c ) 
            { 
              seconds_min = seconds_c; 
              t1c_min = T1;
              t2c_min = T2;
            }
          }
          ++counter;
        }
      }
    }
  } //end scope test 2
#endif


    // Test 2: RangePolicy Collapse2 style
    const double seconds_2 = MultiDimRangePerf3D_Collapse< DeviceType , double , LayoutType >::test_index_collapse(range_length,range_length,range_length) ;
    std::cout << label_range_col2
      << " , " << range_length
      << " , " << seconds_2
      << std::endl ;


    // Test 3: RangePolicy Collapse all style
    const double seconds_3 = MultiDimRangePerf3D_CollapseAll< DeviceType , double , LayoutType >::test_collapse_all(range_length,range_length,range_length) ;
    std::cout << label_range_col_all
      << " , " << range_length
      << " , " << seconds_3
      << "\n"
      << std::endl ;

    // Compare fastest times... will never be collapse all
    if ( seconds_min < seconds_min_c ) {
      if ( seconds_min < seconds_2 ) {
        std::cout << "--------------------------------------------------------------\n"
          << " Fastest run: MDRange tiled\n"
          << " Time: " << seconds_min
          << " Difference: " << seconds_2 - seconds_min
          << " Other times: \n"
          << "   MDrange Collapse type: " << seconds_min_c << "\n"
          << "   Collapse2 Range Policy: " << seconds_2 << "\n"
          << "\n--------------------------------------------------------------"
          << "\n\n"
          << std::endl;
      }
      else if ( seconds_min > seconds_2 ) {
        std::cout << " Fastest run: Collapse2 RangePolicy\n"
          << " Time: " << seconds_2
          << " Difference: " << seconds_min - seconds_2
          << " Other times: \n"
          << "   MDrange Tiled: " << seconds_min << "\n"
          << "   MDrange Collapse type: " << seconds_min_c << "\n"
          << "\n--------------------------------------------------------------"
          << "\n\n"
          << std::endl;
      }
    }
    else if ( seconds_min > seconds_min_c ) {
      if ( seconds_min_c < seconds_2 ) {
        std::cout << "--------------------------------------------------------------\n"
          << " Fastest run: MDRange Collapse type\n"
          << " Time: " << seconds_min_c
          << " Difference: " << seconds_2 - seconds_min_c
          << " Other times: \n"
          << "   MDrange Tiled: " << seconds_min << "\n"
          << "   Collapse2 Range Policy: " << seconds_2 << "\n"
          << "\n--------------------------------------------------------------"
          << "\n\n"
          << std::endl;
      }
      else if ( seconds_min_c > seconds_2 ) {
        std::cout << " Fastest run: Collapse2 RangePolicy\n"
          << " Time: " << seconds_2
          << " Difference: " << seconds_min_c - seconds_2
          << " Other times: \n"
          << "   MDrange Tiled: " << seconds_min << "\n"
          << "   MDrange Collapse type: " << seconds_min_c << "\n"
          << "\n--------------------------------------------------------------"
          << "\n\n"
          << std::endl;
      }
    } // end else if

  } //end for

#undef MDRANGE_PERFORMANCE_OUTPUT_VERBOSE

}


template< class DeviceType >
void run_test_hexgrad( int exp_beg , int exp_end, const char deviceTypeName[] )
{
  std::string label_hexgrad ;
  label_hexgrad.append( "\"HexGrad< double , " );
  // mfh 06 Jun 2013: This only appends "DeviceType" (literally) to
  // the string, not the actual name of the device type.  Thus, I've
  // modified the function to take the name of the device type.
  //
  //label_hexgrad.append( KOKKOS_MACRO_TO_STRING( DeviceType ) );
  label_hexgrad.append( deviceTypeName );
  label_hexgrad.append( " >\"" );

  for (int i = exp_beg ; i < exp_end ; ++i) {
    double min_seconds = 0.0 ;
    double max_seconds = 0.0 ;
    double avg_seconds = 0.0 ;

    const int parallel_work_length = 1<<i;

    for ( int j = 0 ; j < NUMBER_OF_TRIALS ; ++j ) {
      const double seconds = HexGrad< DeviceType >::test(parallel_work_length) ;

      if ( 0 == j ) {
        min_seconds = seconds ;
        max_seconds = seconds ;
      }
      else {
        if ( seconds < min_seconds ) min_seconds = seconds ;
        if ( seconds > max_seconds ) max_seconds = seconds ;
      }
      avg_seconds += seconds ;
    }
    avg_seconds /= NUMBER_OF_TRIALS ;

    std::cout << label_hexgrad
      << " , " << parallel_work_length
      << " , " << min_seconds
      << " , " << ( min_seconds / parallel_work_length )
      << std::endl ;
  }
}

template< class DeviceType >
void run_test_gramschmidt( int exp_beg , int exp_end, const char deviceTypeName[] )
{
  std::string label_gramschmidt ;
  label_gramschmidt.append( "\"GramSchmidt< double , " );
  // mfh 06 Jun 2013: This only appends "DeviceType" (literally) to
  // the string, not the actual name of the device type.  Thus, I've
  // modified the function to take the name of the device type.
  //
  //label_gramschmidt.append( KOKKOS_MACRO_TO_STRING( DeviceType ) );
  label_gramschmidt.append( deviceTypeName );
  label_gramschmidt.append( " >\"" );

  for (int i = exp_beg ; i < exp_end ; ++i) {
    double min_seconds = 0.0 ;
    double max_seconds = 0.0 ;
    double avg_seconds = 0.0 ;

    const int parallel_work_length = 1<<i;

    for ( int j = 0 ; j < NUMBER_OF_TRIALS ; ++j ) {
      const double seconds = ModifiedGramSchmidt< double , DeviceType >::test(parallel_work_length, 32 ) ;

      if ( 0 == j ) {
        min_seconds = seconds ;
        max_seconds = seconds ;
      }
      else {
        if ( seconds < min_seconds ) min_seconds = seconds ;
        if ( seconds > max_seconds ) max_seconds = seconds ;
      }
      avg_seconds += seconds ;
    }
    avg_seconds /= NUMBER_OF_TRIALS ;

    std::cout << label_gramschmidt
      << " , " << parallel_work_length
      << " , " << min_seconds
      << " , " << ( min_seconds / parallel_work_length )
      << std::endl ;
  }
}

}

