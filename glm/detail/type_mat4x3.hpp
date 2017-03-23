///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2014 G-Truc Creation (www.g-truc.net)
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.
///
/// @ref core
/// @file glm/core/type_mat4x3.hpp
/// @date 2006-08-04 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_mat4x3
#define glm_core_type_mat4x3

#include "../fwd.hpp"
#include "type_vec3.hpp"
#include "type_vec4.hpp"
#include "type_mat.hpp"
#include <limits>

namespace glm{
namespace detail
{
	template <typename T, precision P>
	struct tmat4x3
	{
		enum ctor{_null};
		typedef T value_type;
		typedef std::size_t size_type;
		typedef tvec3<T, P> col_type;
		typedef tvec4<T, P> row_type;
		typedef tmat4x3<T, P> type;
		typedef tmat3x4<T, P> transpose_type;

		GLM_FUNC_DECL GLM_CONSTEXPR length_t length() const;

	private:
		// Data 
		col_type value[4];

	public:
		// Constructors
		GLM_FUNC_DECL tmat4x3();
		GLM_FUNC_DECL tmat4x3(tmat4x3<T, P> const & m);
		template <precision Q>
		GLM_FUNC_DECL tmat4x3(tmat4x3<T, Q> const & m);

		GLM_FUNC_DECL explicit tmat4x3(
			ctor Null);
		GLM_FUNC_DECL explicit tmat4x3(
			T GLM_REFERENCE x);
		GLM_FUNC_DECL tmat4x3(
			T GLM_REFERENCE x0, T GLM_REFERENCE y0, T GLM_REFERENCE z0,
			T GLM_REFERENCE x1, T GLM_REFERENCE y1, T GLM_REFERENCE z1,
			T GLM_REFERENCE x2, T GLM_REFERENCE y2, T GLM_REFERENCE z2,
			T GLM_REFERENCE x3, T GLM_REFERENCE y3, T GLM_REFERENCE z3);
		GLM_FUNC_DECL tmat4x3(
			col_type GLM_REFERENCE v0,
			col_type GLM_REFERENCE v1,
			col_type GLM_REFERENCE v2,
			col_type GLM_REFERENCE v3);

		//////////////////////////////////////
		// Conversions

		template <
			typename X1, typename Y1, typename Z1,
			typename X2, typename Y2, typename Z2,
			typename X3, typename Y3, typename Z3,
			typename X4, typename Y4, typename Z4>
		GLM_FUNC_DECL tmat4x3(
			X1 GLM_REFERENCE x1, Y1 GLM_REFERENCE y1, Z1 GLM_REFERENCE z1,
			X2 GLM_REFERENCE x2, Y2 GLM_REFERENCE y2, Z2 GLM_REFERENCE z2,
			X3 GLM_REFERENCE x3, Y3 GLM_REFERENCE y3, Z3 GLM_REFERENCE z3,
			X4 GLM_REFERENCE x4, Y4 GLM_REFERENCE y4, Z4 GLM_REFERENCE z4);
			
		template <typename V1, typename V2, typename V3, typename V4>
		GLM_FUNC_DECL tmat4x3(
			tvec3<V1, P> GLM_REFERENCE v1,
			tvec3<V2, P> GLM_REFERENCE v2,
			tvec3<V3, P> GLM_REFERENCE v3,
			tvec3<V4, P> GLM_REFERENCE v4);

		// Matrix conversions
		template <typename U, precision Q>
		GLM_FUNC_DECL explicit tmat4x3(tmat4x3<U, Q> GLM_REFERENCE m);
			
		GLM_FUNC_DECL explicit tmat4x3(tmat2x2<T, P> GLM_REFERENCE x);
		GLM_FUNC_DECL explicit tmat4x3(tmat3x3<T, P> GLM_REFERENCE x);
		GLM_FUNC_DECL explicit tmat4x3(tmat4x4<T, P> GLM_REFERENCE x);
		GLM_FUNC_DECL explicit tmat4x3(tmat2x3<T, P> GLM_REFERENCE x);
		GLM_FUNC_DECL explicit tmat4x3(tmat3x2<T, P> GLM_REFERENCE x);
		GLM_FUNC_DECL explicit tmat4x3(tmat2x4<T, P> GLM_REFERENCE x);
		GLM_FUNC_DECL explicit tmat4x3(tmat4x2<T, P> GLM_REFERENCE x);
		GLM_FUNC_DECL explicit tmat4x3(tmat3x4<T, P> GLM_REFERENCE x);

		// Accesses
		GLM_FUNC_DECL col_type & operator[](size_type i);
		GLM_FUNC_DECL col_type GLM_REFERENCE operator[](size_type i) const;

		// Unary updatable operators
		GLM_FUNC_DECL tmat4x3<T, P> & operator=  (tmat4x3<T, P> GLM_REFERENCE m);
		template <typename U>
		GLM_FUNC_DECL tmat4x3<T, P> & operator=  (tmat4x3<U, P> GLM_REFERENCE m);
		template <typename U>
		GLM_FUNC_DECL tmat4x3<T, P> & operator+= (U s);
		template <typename U>
		GLM_FUNC_DECL tmat4x3<T, P> & operator+= (tmat4x3<U, P> GLM_REFERENCE m);
		template <typename U>
		GLM_FUNC_DECL tmat4x3<T, P> & operator-= (U s);
		template <typename U>
		GLM_FUNC_DECL tmat4x3<T, P> & operator-= (tmat4x3<U, P> GLM_REFERENCE m);
		template <typename U>
		GLM_FUNC_DECL tmat4x3<T, P> & operator*= (U s);
		template <typename U>
		GLM_FUNC_DECL tmat4x3<T, P> & operator/= (U s);

		//////////////////////////////////////
		// Increment and decrement operators

		GLM_FUNC_DECL tmat4x3<T, P> & operator++ ();
		GLM_FUNC_DECL tmat4x3<T, P> & operator-- ();
		GLM_FUNC_DECL tmat4x3<T, P> operator++(int);
		GLM_FUNC_DECL tmat4x3<T, P> operator--(int);
	};

	// Binary operators
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator+ (
		tmat4x3<T, P> GLM_REFERENCE m,
		T GLM_REFERENCE s);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator+ (
		tmat4x3<T, P> GLM_REFERENCE m1,
		tmat4x3<T, P> GLM_REFERENCE m2);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator- (
		tmat4x3<T, P> GLM_REFERENCE m,
		T GLM_REFERENCE s);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator- (
		tmat4x3<T, P> GLM_REFERENCE m1,
		tmat4x3<T, P> GLM_REFERENCE m2);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator* (
		tmat4x3<T, P> GLM_REFERENCE m,
		T GLM_REFERENCE s);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator* (
		T GLM_REFERENCE s,
		tmat4x3<T, P> GLM_REFERENCE m);

	template <typename T, precision P>
	GLM_FUNC_DECL typename tmat4x3<T, P>::col_type operator* (
		tmat4x3<T, P> GLM_REFERENCE m,
		typename tmat4x3<T, P>::row_type GLM_REFERENCE v);

	template <typename T, precision P>
	GLM_FUNC_DECL typename tmat4x3<T, P>::row_type operator* (
		typename tmat4x3<T, P>::col_type GLM_REFERENCE v,
		tmat4x3<T, P> GLM_REFERENCE m);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat2x3<T, P> operator* (
		tmat4x3<T, P> GLM_REFERENCE m1,
		tmat2x4<T, P> GLM_REFERENCE m2);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat3x3<T, P> operator* (
		tmat4x3<T, P> GLM_REFERENCE m1,
		tmat3x4<T, P> GLM_REFERENCE m2);
		
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator* (
		tmat4x3<T, P> GLM_REFERENCE m1,
		tmat4x4<T, P> GLM_REFERENCE m2);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator/ (
		tmat4x3<T, P> GLM_REFERENCE m,
		T GLM_REFERENCE s);

	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> operator/ (
		T GLM_REFERENCE s,
		tmat4x3<T, P> GLM_REFERENCE m);

	// Unary constant operators
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x3<T, P> const operator- (
		tmat4x3<T, P> GLM_REFERENCE m);

}//namespace detail
}//namespace glm

#ifndef GLM_EXTERNAL_TEMPLATE
#include "type_mat4x3.inl"
#endif //GLM_EXTERNAL_TEMPLATE

#endif//glm_core_type_mat4x3
