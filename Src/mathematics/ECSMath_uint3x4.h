﻿#pragma once

//=============================================================================
// ECS MATH 
//=============================================================================
namespace ecs
{
	namespace math
	{
		//=========================================================
		// math-uint3x4
		//=========================================================
		/// <summary>Return the uint4x3 transpose of a uint3x4 matrix.</summary>
		/// <param name="v">Value to transpose.</param>
		/// <returns>Transposed value.</returns>
		uint4x3 transpose(const uint3x4& v);

		/// <summary>Returns a uint hash code of a uint3x4 matrix.</summary>
		/// <param name="v">Matrix value to hash.</param>
		/// <returns>uint hash of the argument.</returns>
		uint hash(const uint3x4& v);

		/// <summary>
		/// Returns a uint3 vector hash code of a uint3x4 matrix.
		/// When multiple elements are to be hashes together, it can more efficient to calculate and combine wide hash
		/// that are only reduced to a narrow uint hash at the very end instead of at every step.
		/// </summary>
		/// <param name="v">Matrix value to hash.</param>
		/// <returns>uint3 hash of the argument.</returns>
		uint3 hashwide(const uint3x4& v);
	}

    // https://github.com/Unity-Technologies/Unity.Mathematics/tree/master/src/Unity.Mathematics/uint3x4.gen.cs
#pragma pack(push,1)
	struct uint3x4
	{
	public:
		static const uint3x4		zero;       /// <summary>uint3x4 zero value.</summary>
		union {
			struct
			{
				uint3 c0;	/// <summary>Column 0 of the matrix.</summary>
				uint3 c1;	/// <summary>Column 1 of the matrix.</summary>
				uint3 c2;	/// <summary>Column 2 of the matrix.</summary>
				uint3 c3;	/// <summary>Column 2 of the matrix.</summary>
			};
			uint3		data[4];
		};

	public:
		//=========================================================
		// CONSTRUCTOR
		//=========================================================
        uint3x4() : c0(0), c1(0), c2(0), c3(0) {}
		uint3x4(const uint3& _c0, const uint3& _c1, const uint3& _c2, const uint3& _c3) { c0 = _c0;	c1 = _c1;	c2 = _c2;	c3 = _c3; }

        /// <summary>Constructs a uint3x4 matrix from 12 uint values given in row-major order.</summary>
        /// mXY : The matrix at row X, column Y will be set to this value		
        uint3x4(uint m00, uint m01, uint m02, uint m03,
                uint m10, uint m11, uint m12, uint m13,
                uint m20, uint m21, uint m22, uint m23)
        {
            c0 = uint3(m00, m10, m20);
            c1 = uint3(m01, m11, m21);
            c2 = uint3(m02, m12, m22);
            c3 = uint3(m03, m13, m23);
        }

        uint3x4(bool v);
		uint3x4(uint v)		            {   c0 = v;             c1 = v;             c2 = v;             c3 = v;             }
		uint3x4(int v)          		{   c0 = (uint3)v;      c1 = (uint3)v;      c2 = (uint3)v;      c3 = (uint3)v;      }
		uint3x4(float v)				{	c0 = (uint3)v;      c1 = (uint3)v;      c2 = (uint3)v;      c3 = (uint3)v;      }
		uint3x4(double v)				{	c0 = (uint3)v;		c1 = (uint3)v;		c2 = (uint3)v;		c3 = (uint3)v;		}

        uint3x4(const bool3x4& v);
		uint3x4(const uint3x4& v);
        uint3x4(const int3x4& v);
        uint3x4(const float3x4& v);
        uint3x4(const double3x4& v);

		//=========================================================
		// OPERATORS
		//=========================================================
		//=========================================================
		// Assignment operators : T&
		//=========================================================
		// simple assignment	            a = b	
        uint3x4& operator = (const uint3x4& rhs)            { this->c0 = rhs.c0;  this->c1 = rhs.c1;  this->c2 = rhs.c2;  this->c3 = rhs.c3;  return (*this); }
		// addition assignment	            a += b	    
        uint3x4& operator +=(const uint3x4& rhs)            { this->c0 += rhs.c0; this->c1 += rhs.c1; this->c2 += rhs.c2; this->c3 += rhs.c3; return (*this); }
		// subtraction assignment	        a -= b	    
        uint3x4& operator -=(const uint3x4& rhs)            { this->c0 -= rhs.c0; this->c1 -= rhs.c1; this->c2 -= rhs.c2; this->c3 -= rhs.c3; return (*this); }
		// multiplication assignment	    a *= b	    
        uint3x4& operator *=(const uint3x4& rhs)            { this->c0 *= rhs.c0; this->c1 *= rhs.c1; this->c2 *= rhs.c2; this->c3 *= rhs.c3; return (*this); }
		// division assignment	            a /= b	    
        uint3x4& operator /=(const uint3x4& rhs)            { this->c0 /= rhs.c0; this->c1 /= rhs.c1; this->c2 /= rhs.c2; this->c3 /= rhs.c3; return (*this); }
		// modulo assignment	            a %= b	    
        uint3x4& operator %=(const uint3x4& rhs)            { this->c0 %= rhs.c0; this->c1 %= rhs.c1; this->c2 %= rhs.c2; this->c3 %= rhs.c3; return (*this); }

        // bitwise AND assignment	        a &= b	    
        uint3x4& operator &=(const uint3x4& rhs)            { this->c0 &= rhs.c0; this->c1 &= rhs.c1; this->c2 &= rhs.c2; this->c3 &= rhs.c3; return (*this); }
        // bitwise OR assignment	        a |= b	    
        uint3x4& operator |=(const uint3x4& rhs)            { this->c0 |= rhs.c0; this->c1 |= rhs.c1; this->c2 |= rhs.c2; this->c3 |= rhs.c3; return (*this); }
        // bitwise XOR assignment	        a ^= b	
        // bitwise left shift assignment	a <<= b	
        // bitwise right shift assignment   a >>= b
        
        //=========================================================
        // Increment/decrement operators
        //=========================================================
        // pre-increment    : 	++a		T&
        uint3x4& operator ++ ()                             { ++this->c0; ++this->c1; ++this->c2; ++this->c3; return (*this);     }
        // pre - decrement  : 	--a		T&
        uint3x4& operator -- ()                             { --this->c0; --this->c1; --this->c2; --this->c3; return (*this);     }
        // post-increment   : 	a++
        uint3x4  operator ++ (int)                          { auto temp = *this; ++(*this); return (temp); }
        // post-decrement   :	a--
        uint3x4  operator -- (int)                          { auto temp = *this; --(*this); return (temp); }

        //=========================================================
        // Arithmetic operators
        //=========================================================
        // unary plus       :   +a
        uint3x4 operator + () const                         { return uint3x4(+this->c0, +this->c1, +this->c2, +this->c3); }
        // unary minus      :   -a
        uint3x4 operator - () const                         { return uint3x4(-this->c0, -this->c1, -this->c2, -this->c3); }

		// addition         :   a + b
        uint3x4 operator + (const uint3x4& rhs)		const	{ return uint3x4(this->c0 + rhs.c0,		this->c1 + rhs.c1,		this->c2 + rhs.c2,		this->c3 + rhs.c3); }
        uint3x4 operator + (uint rhs)				const	{ return uint3x4(this->c0 + rhs,		this->c1 + rhs,			this->c2 + rhs,			this->c3 + rhs); }
		friend uint3x4 operator + (uint lhs, const uint3x4& rhs) { return uint3x4(lhs + rhs.c0, lhs + rhs.c1, lhs + rhs.c2, lhs + rhs.c3); }

		// subtraction      :   a - b
        uint3x4 operator - (const uint3x4& rhs)		const	{ return uint3x4(this->c0 - rhs.c0,		this->c1 - rhs.c1,		this->c2 - rhs.c2,		this->c3 - rhs.c3); }
        uint3x4 operator - (uint rhs)				const	{ return uint3x4(this->c0 - rhs,		this->c1 - rhs,			this->c2 - rhs,			this->c3 - rhs); }
		friend uint3x4 operator - (uint lhs, const uint3x4& rhs) { return uint3x4(lhs - rhs.c0, lhs - rhs.c1, lhs - rhs.c2, lhs - rhs.c3); }

		// multiplication   :   a * b
        uint3x4 operator * (const uint3x4& rhs)     const   { return uint3x4(this->c0 * rhs.c0,		this->c1 * rhs.c1,		this->c2 * rhs.c2,		this->c3 * rhs.c3); }
        uint3x4 operator * (uint rhs)               const   { return uint3x4(this->c0 * rhs,		this->c1 * rhs,			this->c2 * rhs,			this->c3 * rhs); }
		friend uint3x4 operator * (uint lhs, const uint3x4& rhs) { return uint3x4(lhs * rhs.c0, lhs * rhs.c1, lhs * rhs.c2, lhs * rhs.c3); }

		// division         :   a / b
        uint3x4 operator / (const uint3x4& rhs)		const	{ return uint3x4(this->c0 / rhs.c0,		this->c1 / rhs.c1,		this->c2 / rhs.c2,		this->c3 / rhs.c3); }
        uint3x4 operator / (uint rhs)				const	{ return uint3x4(this->c0 / rhs,		this->c1 / rhs,			this->c2 / rhs,			this->c3 / rhs); }
		friend uint3x4 operator / (uint lhs, const uint3x4& rhs) { return uint3x4(lhs / rhs.c0, lhs / rhs.c1, lhs / rhs.c2, lhs / rhs.c3); }

		// modulo           :   a % b
        uint3x4 operator % (const uint3x4& rhs)		const	{ return uint3x4(this->c0 % rhs.c0,		this->c1 % rhs.c1,		this->c2 % rhs.c2,		this->c3 % rhs.c3); }
        uint3x4 operator % (uint rhs)				const	{ return uint3x4(this->c0 % rhs,		this->c1 % rhs,			this->c2 % rhs,			this->c3 % rhs); }
		friend uint3x4 operator % (uint lhs, const uint3x4& rhs) { return uint3x4(lhs % rhs.c0, lhs % rhs.c1, lhs % rhs.c2, lhs % rhs.c3); }

		// bitwise NOT      :   ~a
		uint3x4 operator ~ () const							{ return uint3x4(~this->c0, ~this->c1, ~this->c2, ~this->c3); }

		// bitwise AND      :   a & b
        uint3x4 operator & (const uint3x4& rhs)		const	{ return uint3x4(this->c0 & rhs.c0,		this->c1 & rhs.c1,		this->c2 & rhs.c2,		this->c3 & rhs.c3); }
        uint3x4 operator & (uint rhs)				const	{ return uint3x4(this->c0 & rhs,		this->c1 & rhs,			this->c2 & rhs,			this->c3 & rhs); }
		friend uint3x4 operator & (uint lhs, const uint3x4& rhs) { return uint3x4(lhs & rhs.c0, lhs & rhs.c1, lhs & rhs.c2, lhs & rhs.c3); }

		// bitwise OR       :   a | b
        uint3x4 operator | (const uint3x4& rhs)		const	{ return uint3x4(this->c0 | rhs.c0,		this->c1 | rhs.c1,		this->c2 | rhs.c2,		this->c3 | rhs.c3); }
        uint3x4 operator | (uint rhs)				const	{ return uint3x4(this->c0 | rhs,		this->c1 | rhs,			this->c2 | rhs,			this->c3 | rhs); }
		friend uint3x4 operator | (uint lhs, const uint3x4& rhs) { return uint3x4(lhs | rhs.c0, lhs | rhs.c1, lhs | rhs.c2, lhs | rhs.c3); }

		// bitwise XOR      :   a ^ b
        uint3x4 operator ^ (const uint3x4& rhs)		const	{ return uint3x4(this->c0 ^ rhs.c0,		this->c1 ^ rhs.c1,		this->c2 ^ rhs.c2,		this->c3 ^ rhs.c3); }
        uint3x4 operator ^ (uint rhs)				const	{ return uint3x4(this->c0 ^ rhs,		this->c1 ^ rhs,			this->c2 ^ rhs,			this->c3 ^ rhs); }
		friend uint3x4 operator ^ (uint lhs, const uint3x4& rhs) { return uint3x4(lhs ^ rhs.c0, lhs ^ rhs.c1, lhs ^ rhs.c2, lhs ^ rhs.c3); }

		// bitwise left shift : a << b
		uint3x4 operator << (int n) const					{ return uint3x4(this->c0 << n, this->c1 << n, this->c2 << n, this->c3 << n); }

		// bitwise right shift: a >> b
        uint3x4 operator >> (int n) const					{ return uint3x4(this->c0 >> n, this->c1 >> n, this->c2 >> n, this->c3 >> n); }

		//=========================================================
		// Logical operators
		//=========================================================
		// negation	        :   not a, !a
		// AND	            :   a and b, a && b
		// inclusive OR	    :   a or b,  a || b

		//=========================================================
		// Comparison operators
		//=========================================================
		// equal to         :   a == 
        bool3x4 operator == (const uint3x4& rhs)	const	{ return bool3x4(this->c0 == rhs.c0,	this->c1 == rhs.c1,		this->c2 == rhs.c2,		this->c3 == rhs.c3); }
        bool3x4 operator == (uint rhs)				const	{ return bool3x4(this->c0 == rhs,		this->c1 == rhs,		this->c2 == rhs,		this->c3 == rhs); }
		friend bool3x4 operator == (uint lhs, const uint3x4& rhs) { return bool3x4(lhs == rhs.c0, lhs == rhs.c1, lhs == rhs.c2, lhs == rhs.c3); }

		// not equal to     :   a != b
        bool3x4 operator != (const uint3x4& rhs)	const	{ return bool3x4(this->c0 != rhs.c0,	this->c1 != rhs.c1,		this->c2 != rhs.c2,		this->c3 != rhs.c3); }
        bool3x4 operator != (uint rhs)				const	{ return bool3x4(this->c0 != rhs,		this->c1 != rhs,		this->c2 != rhs,		this->c3 != rhs); }
		friend bool3x4 operator != (uint lhs, const uint3x4& rhs) { return bool3x4(lhs != rhs.c0, lhs != rhs.c1, lhs != rhs.c2, lhs != rhs.c3); }

		// less than        :   a < b
        bool3x4 operator < (const uint3x4& rhs)		const	{ return bool3x4(this->c0 < rhs.c0,		this->c1 < rhs.c1,		this->c2 < rhs.c2,		this->c3 < rhs.c3); }
        bool3x4 operator < (uint rhs)				const	{ return bool3x4(this->c0 < rhs,		this->c1 < rhs,			this->c2 < rhs,			this->c3 < rhs); }
		friend bool3x4 operator < (uint lhs, const uint3x4& rhs) { return bool3x4(lhs < rhs.c0, lhs < rhs.c1, lhs < rhs.c2, lhs < rhs.c3); }

		// greater than     :   a > b
        bool3x4 operator > (const uint3x4& rhs)		const	{ return bool3x4(this->c0 > rhs.c0,		this->c1 > rhs.c1,		this->c2 > rhs.c2,		this->c3 > rhs.c3); }
        bool3x4 operator > (uint rhs)				const	{ return bool3x4(this->c0 > rhs,		this->c1 > rhs,			this->c2 > rhs,			this->c3 > rhs); }
		friend bool3x4 operator > (uint lhs, const uint3x4& rhs) { return bool3x4(lhs > rhs.c0, lhs > rhs.c1, lhs > rhs.c2, lhs > rhs.c3); }

		// less than or equal to    : a <= b	
		bool3x4 operator <= (const uint3x4& rhs)	const	{ return bool3x4(this->c0 <= rhs.c0,	this->c1 <= rhs.c1,		this->c2 <= rhs.c2,		this->c3 <= rhs.c3); }
        bool3x4 operator <= (uint rhs)				const	{ return bool3x4(this->c0 <= rhs,		this->c1 <= rhs,		this->c2 <= rhs,		this->c3 <= rhs); }
		friend bool3x4 operator <= (uint lhs, const uint3x4& rhs) { return bool3x4(lhs <= rhs.c0, lhs <= rhs.c1, lhs <= rhs.c2, lhs <= rhs.c3); }

		// greater than or equal to : a >= b
        bool3x4 operator >= (const uint3x4& rhs)	const	{ return bool3x4(this->c0 >= rhs.c0,	this->c1 >= rhs.c1,		this->c2 >= rhs.c2,		this->c3 >= rhs.c3); }
        bool3x4 operator >= (uint rhs)				const	{ return bool3x4(this->c0 >= rhs,		this->c1 >= rhs,		this->c2 >= rhs,		this->c3 >= rhs); }
        friend bool3x4 operator >= (uint lhs, const uint3x4& rhs) { return bool3x4(lhs >= rhs.c0, lhs >= rhs.c1, lhs >= rhs.c2, lhs >= rhs.c3); }

        //=========================================================
        // Conversion operators
        //=========================================================


		//=========================================================
		// member access
		//=========================================================
        /// <summary>Returns the uint3 element at a specified index.</summary>
		uint3& operator[] (int index)
		{
#if defined(_DEBUG ) || defined(_DEVELOPMENT)
			if (index >= 4)
				throw std::exception("index must be between[0...3]");
#endif
			return data[index];
		}
		
        //=========================================================
        // METHOD
		//=========================================================
        /// <summary>Returns true if the uint3x4 is equal to a given uint3x4, false otherwise.</summary>
        /// <param name="rhs">Right hand side argument to compare equality with.</param>
        /// <returns>The result of the equality comparison.</returns>
        bool Equals(const uint3x4& rhs) const { return c0.Equals(rhs.c0) && c1.Equals(rhs.c1) && c2.Equals(rhs.c2) && c3.Equals(rhs.c3); }

        /// <summary>Returns true if the uint3x4 is equal to a given uint3x4, false otherwise.</summary>
        /// <param name="o">Right hand side argument to compare equality with.</param>
        /// <returns>The result of the equality comparison.</returns>
        //override bool Equals(object o) { return o is uint3x4 converted && Equals(converted); }

        /// <summary>Returns a hash code for the uint3x4.</summary>
        /// <returns>The computed hash code.</returns>
        int GetHashCode() { return (int)math::hash(*this); }

        /// <summary>Returns a string representation of the uint3x4.</summary>
        /// <returns>String representation of the value.</returns>
        std::string ToString() const
        {
            return std::format("uint3x4({0}, {1}, {2}, {3},  {4}, {5}, {6}, {7},  {8}, {9}, {10}, {11})", c0.x, c1.x, c2.x, c3.x, c0.y, c1.y, c2.y, c3.y, c0.z, c1.z, c2.z, c3.z);
        }
	};
    __declspec(selectany) const uint3x4 uint3x4::zero = uint3x4(0, 0, 0, 0);
#pragma pack(pop)


} // namespace ecs

