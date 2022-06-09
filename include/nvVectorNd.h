/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
// A Customized version of NVIDIA's
// Template math library for common 3D functionality
//
// nvVectorNd.h - 2-vector, 3-vector, and 4-vector templates and utilities
//
// This code is in part derived from glh, a cross platform glut helper library.
// The copyright for glh follows this notice.
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

/*
    Copyright (c) 2000 Cass Everitt
    Copyright (c) 2000 NVIDIA Corporation
    All rights reserved.

    Redistribution and use in source and binary forms, with or
    without modification, are permitted provided that the following
    conditions are met:

     * Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer.

     * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution.

     * The names of contributors to this software may not be used
       to endorse or promote products derived from this software
       without specific prior written permission.

       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
       ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
       LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
       FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
       REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
       INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
       BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
       LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
       CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
       LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
       ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
       POSSIBILITY OF SUCH DAMAGE.

    Cass Everitt - cass@r3.nu
*/

#ifndef NV_VECTOR_H
#define NV_VECTOR_H

namespace nv_ext {

    template<class T, unsigned int D = 2>
    class Vec2;

    template<class T, unsigned int D = 3>
    class Vec3;

    template<class T, unsigned int D = 4>
    class Vec4;

    //////////////////////////////////////////////////////////////////////
    //
    // Vec - template class for N-tuple vector
    //
    //////////////////////////////////////////////////////////////////////
    template<class T, unsigned int D>
    class Vec {
    public:

        typedef T value_type;

        CUDAFUNCTION int size() const {
            return D;
        }

        ////////////////////////////////////////////////////////
        //
        //  Constructors
        //
        ////////////////////////////////////////////////////////

        // Default/scalar constructor
        CUDAFUNCTION Vec(const T &t = T()) {
            for (int i = 0; i < size(); i++) {
                _array[i] = t;
            }
        }

        // Construct from array
        CUDAFUNCTION Vec(const T *tp) {
            for (int i = 0; i < size(); i++) {
                _array[i] = tp[i];
            }
        }

        // Construct from explicit values
        CUDAFUNCTION Vec(const T v0, const T v1) {
            _array[0] = v0;
            _array[1] = v1;
//            x = v0;
//            y = v1;
        }

        CUDAFUNCTION explicit Vec(const Vec3<T> &u) {
            for (int i = 0; i < size(); i++) {
                _array[i] = u._array[i];
            }
        }

        CUDAFUNCTION explicit Vec(const Vec4<T> &u) {
            for (int i = 0; i < size(); i++) {
                _array[i] = u._array[i];
            }
        }

        CUDAFUNCTION const T *get_value() const {
            return _array;
        }

        CUDAFUNCTION Vec2<T> &set_value(const T *rhs) {
            for (int i = 0; i < size(); i++) {
                _array[i] = rhs[i];
            }

            return *this;
        }

        // indexing operators
        CUDAFUNCTION T &operator[](int i) {
            return _array[i];
        }

        CUDAFUNCTION const T &operator[](int i) const {
            return _array[i];
        }

        // type-cast operators
        CUDAFUNCTION operator T *() {
            return _array;
        }

        CUDAFUNCTION operator const T *() const {
            return _array;
        }

        ////////////////////////////////////////////////////////
        //
        //  Math operators
        //
        ////////////////////////////////////////////////////////

        // scalar multiply assign
        friend CUDAFUNCTION Vec<T, D> &operator*=(Vec<T, D> &lhs, T d) {
            for (int i = 0; i < lhs.size(); i++) {
                lhs._array[i] *= d;
            }

            return lhs;
        }

        // component-wise vector multiply assign
        friend CUDAFUNCTION Vec<T, D> &operator*=(Vec<T, D> &lhs, const Vec<T, D> &rhs) {
            for (int i = 0; i < lhs.size(); i++) {
                lhs._array[i] *= rhs[i];
            }

            return lhs;
        }

        // scalar divide assign
        friend CUDAFUNCTION Vec<T, D> &operator/=(Vec<T, D> &lhs, T d) {
            if (d == 0) {
                return lhs;
            }

            for (int i = 0; i < lhs.size(); i++) {
                lhs._array[i] /= d;
            }

            return lhs;
        }

        // component-wise vector divide assign
        friend CUDAFUNCTION Vec<T, D> &operator/=(Vec<T, D> &lhs, const Vec2<T> &rhs) {
            for (int i = 0; i < lhs.size(); i++) {
                lhs._array[i] /= rhs._array[i];
            }

            return lhs;
        }

        // component-wise vector add assign
        friend CUDAFUNCTION Vec<T, D> &operator+=(Vec<T, D> &lhs, const Vec2<T> &rhs) {
            for (int i = 0; i < lhs.size(); i++) {
                lhs._array[i] += rhs._array[i];
            }

            return lhs;
        }

        // component-wise vector subtract assign
        friend CUDAFUNCTION Vec<T, D> &operator-=(Vec<T, D> &lhs, const Vec2<T> &rhs) {
            for (int i = 0; i < lhs.size(); i++) {
                lhs._array[i] -= rhs._array[i];
            }

            return lhs;
        }

        // unary negate
        friend CUDAFUNCTION Vec<T, D> operator-(const Vec<T, D> &rhs) {
            Vec<T, D> rv;

            for (int i = 0; i < rhs.size(); i++) {
                rv._array[i] = -rhs._array[i];
            }

            return rv;
        }

        // vector add
        friend CUDAFUNCTION Vec<T, D> operator+(const Vec<T, D> &lhs, const Vec<T, D> &rhs) {
            Vec<T, D> rt(lhs);
            return rt += rhs;
        }

        // vector subtract
        friend CUDAFUNCTION Vec<T, D> operator-(const Vec<T, D> &lhs, const Vec<T, D> &rhs) {
            Vec<T, D> rt(lhs);
            return rt -= rhs;
        }

        // scalar multiply
        friend CUDAFUNCTION Vec<T, D> operator*(const Vec<T, D> &lhs, T rhs) {
            Vec<T, D> rt(lhs);
            return rt *= rhs;
        }

        // scalar multiply
        friend CUDAFUNCTION Vec<T, D> operator*(T lhs, const Vec<T, D> &rhs) {
            Vec<T, D> rt(lhs);
            return rt *= rhs;
        }

        // vector component-wise multiply
        friend CUDAFUNCTION Vec<T, D> operator*(const Vec<T, D> &lhs, const Vec<T, D> &rhs) {
            Vec<T, D> rt(lhs);
            return rt *= rhs;
        }

        // scalar multiply
        friend CUDAFUNCTION Vec<T, D> operator/(const Vec<T, D> &lhs, T rhs) {
            Vec<T, D> rt(lhs);
            return rt /= rhs;
        }

        // vector component-wise multiply
        friend CUDAFUNCTION Vec<T, D> operator/(const Vec<T, D> &lhs, const Vec<T, D> &rhs) {
            Vec<T, D> rt(lhs);
            return rt /= rhs;
        }

        ////////////////////////////////////////////////////////
        //
        //  Comparison operators
        //
        ////////////////////////////////////////////////////////

        // equality
        friend CUDAFUNCTION bool operator==(const Vec<T, D> &lhs, const Vec<T, D> &rhs) {
            bool r = true;

            for (int i = 0; i < lhs.size(); i++) {
                r &= lhs._array[i] == rhs._array[i];
            }

            return r;
        }

        // inequality
        friend CUDAFUNCTION bool operator!=(const Vec<T, D> &lhs, const Vec<T, D> &rhs) {
            bool r = true;

            for (int i = 0; i < lhs.size(); i++) {
                r &= lhs._array[i] != rhs._array[i];
            }

            return r;
        }

        //_data intentionally left public to allow vec2.x
        union {
//                struct
//                {
//                    T x,y;          // standard names for components
//                };
//                struct
//                {
//                    T s,t;          // standard names for components
//                };
            T _array[D];     // array access
        };
    };

//    template<typename T> Vec<T, 2> vec2;
//    template<typename T> Vec<T, 3> vec3;
//    template<typename T> Vec<T, 4> vec4;

//    template vec2<typename T, 2> vec2;
//    template vec3<typename T, 3> vec3;
//    template vec4<typename T, 4> vec4;

    typedef Vec<unsigned char, 2> Vec2b;
    typedef Vec<unsigned char, 3> Vec3b;
    typedef Vec<unsigned char, 4> Vec4b;

    typedef Vec<short, 2> Vec2s;
    typedef Vec<short, 3> Vec3s;
    typedef Vec<short, 4> Vec4s;

    typedef Vec<ushort, 2> Vec2w;
    typedef Vec<ushort, 3> Vec3w;
    typedef Vec<ushort, 4> Vec4w;

    typedef Vec<int, 2> Vec2i;
    typedef Vec<int, 3> Vec3i;
    typedef Vec<int, 4> Vec4i;
    typedef Vec<int, 6> Vec6i;
    typedef Vec<int, 8> Vec8i;

    typedef Vec<float, 2> Vec2f;
    typedef Vec<float, 3> Vec3f;
    typedef Vec<float, 4> Vec4f;
    typedef Vec<float, 6> Vec6f;

    typedef Vec<double, 2> Vec2d;
    typedef Vec<double, 3> Vec3d;
    typedef Vec<double, 4> Vec4d;
    typedef Vec<double, 6> Vec6d;

    ////////////////////////////////////////////////////////////////////////////////
    //
    // Generic vector operations
    //
    ////////////////////////////////////////////////////////////////////////////////

    // compute the dot product of two vectors
    template<class T>
    inline CUDAFUNCTION typename T::value_type dot(const T &lhs, const T &rhs) {
        typename T::value_type r = 0;

        for (int i = 0; i < lhs.size(); i++) {
            r += lhs._array[i] * rhs._array[i];
        }

        return r;
    }

    // return the length of the provided vector
    template<class T>
    inline CUDAFUNCTION typename T::value_type length(const T &vec) {
        typename T::value_type r = 0;

        for (int i = 0; i < vec.size(); i++) {
            r += vec._array[i] * vec._array[i];
        }

        return typename T::value_type(sqrt(r));
    }

    // return the squared norm
    template<class T>
    inline CUDAFUNCTION typename T::value_type square_norm(const T &vec) {
        typename T::value_type r = 0;

        for (int i = 0; i < vec.size(); i++) {
            r += vec._array[i] * vec._array[i];
        }

        return r;
    }

    // return the normalized version of the vector
    template<class T>
    inline CUDAFUNCTION T normalize(const T &vec) {
        typename T::value_type sum(0);
        T r;

        for (int i = 0; i < vec.size(); i++) {
            sum += vec._array[i] * vec._array[i];
        }

        sum = typename T::value_type(sqrt(sum));

        if (sum > 0)
            for (int i = 0; i < vec.size(); i++) {
                r._array[i] = vec._array[i] / sum;
            }

        return r;
    }

    // In VC8 : min and max are already defined by a #define...
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

    //componentwise min
    template<class T>
    inline CUDAFUNCTION T min(const T &lhs, const T &rhs) {
        T rt;

        for (int i = 0; i < lhs.size(); i++) {
            rt._array[i] = std::min(lhs._array[i], rhs._array[i]);
        }

        return rt;
    }

    // component-wise max
    template<class T>
    inline CUDAFUNCTION T max(const T &lhs, const T &rhs) {
        T rt;

        for (int i = 0; i < lhs.size(); i++) {
            rt._array[i] = std::max(lhs._array[i], rhs._array[i]);
        }

        return rt;
    }


};

#endif
