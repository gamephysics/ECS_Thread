﻿#pragma once

//=============================================================================
// ECS MATH 
//=============================================================================
namespace ecs
{
	namespace math
	{
		//=========================================================
		// math-double2x4
		//=========================================================
		/// <summary>Return the double4x2 transpose of a double2x4 matrix.</summary>
		/// <param name="v">Value to transpose.</param>
		/// <returns>Transposed value.</returns>
		double4x2 transpose(const double2x4& v);

		/// <summary>Returns a uint hash code of a double2x4 matrix.</summary>
		/// <param name="v">Matrix value to hash.</param>
		/// <returns>uint hash of the argument.</returns>
		uint hash(const double2x4& v);

		/// <summary>
		/// Returns a uint2 vector hash code of a double2x4 matrix.
		/// When multiple elements are to be hashes together, it can more efficient to calculate and combine wide hash
		/// that are only reduced to a narrow uint hash at the very end instead of at every step.
		/// </summary>
		/// <param name="v">Matrix value to hash.</param>
		/// <returns>uint2 hash of the argument.</returns>
		uint2 hashwide(const double2x4& v);
	}

	// https://github.com/Unity-Technologies/Unity.Mathematics/tree/master/src/Unity.Mathematics/double2x4.gen.cs
#pragma pack(push,1)
	struct double2x4
	{
	public:
		static const double2x4		zero;       /// <summary>double2x4 zero value.</summary>
		union {
			struct
			{
				double2 c0;	/// <summary>Column 0 of the matrix.</summary>
				double2 c1;	/// <summary>Column 1 of the matrix.</summary>
				double2 c2;	/// <summary>Column 2 of the matrix.</summary>
				double2 c3;	/// <summary>Column 2 of the matrix.</summary>
			};
			double2		data[4];
		};

	public:
		//=========================================================
		// CONSTRUCTOR
		//=========================================================
		double2x4() : c0(0), c1(0), c2(0), c3(0) {}
		double2x4(const double2& _c0, const double2& _c1, const double2& _c2, const double2& _c3) { c0 = _c0;	c1 = _c1;	c2 = _c2;	c3 = _c3; }
		
        /// <summary>Constructs a double2x4 matrix from 8 double values given in row-major order.</summary>
        /// mXY : The matrix at row X, column Y will be set to this value		
        double2x4(  double m00, double m01, double m02, double m03,
                    double m10, double m11, double m12, double m13)
        {
            c0 = double2(m00, m10);
            c1 = double2(m01, m11);
            c2 = double2(m02, m12);
            c3 = double2(m03, m13);
        }
        
        double2x4(bool v);
		double2x4(uint v)		        {   c0 = v;     c1 = v;     c2 = v;     c3 = v;		}
        double2x4(int v)                {   c0 = v;     c1 = v;     c2 = v;     c3 = v;     }
		double2x4(float v)		        {   c0 = v;     c1 = v;     c2 = v;     c3 = v;		}
        double2x4(double v)             {   c0 = v;     c1 = v;     c2 = v;     c3 = v;     }

        double2x4(const bool2x4& v);
		double2x4(const uint2x4& v);
        double2x4(const int2x4& v);
        double2x4(const float2x4& v);
        double2x4(const double2x4& v);

		//=========================================================
		// OPERATORS
		//=========================================================
		//=========================================================
		// Assignment operators : T&
		//=========================================================
		// simple assignment	            a = b	
		double2x4& operator = (const double2x4& rhs)            { this->c0 =  rhs.c0; this->c1 =  rhs.c1; this->c2  = rhs.c2; this->c3 = rhs.c3; return (*this); }
		// addition assignment	            a += b	
		double2x4& operator +=(const double2x4& rhs)            { this->c0 += rhs.c0; this->c1 += rhs.c1; this->c2 += rhs.c2; this->c3 += rhs.c3; return (*this); }
		// subtraction assignment	        a -= b	
		double2x4& operator -=(const double2x4& rhs)            { this->c0 -= rhs.c0; this->c1 -= rhs.c1; this->c2 -= rhs.c2; this->c3 -= rhs.c3; return (*this); }
		// multiplication assignment	    a *= b	
		double2x4& operator *=(const double2x4& rhs)            { this->c0 *= rhs.c0; this->c1 *= rhs.c1; this->c2 *= rhs.c2; this->c3 *= rhs.c3; return (*this); }
		// division assignment	            a /= b	
		double2x4& operator /=(const double2x4& rhs)            { this->c0 /= rhs.c0; this->c1 /= rhs.c1; this->c2 /= rhs.c2; this->c3 /= rhs.c3; return (*this); }
		// modulo assignment	            a %= b	
		double2x4& operator %=(const double2x4& rhs)            { this->c0 %= rhs.c0; this->c1 %= rhs.c1; this->c2 %= rhs.c2; this->c3 %= rhs.c3; return (*this); }
		
		// bitwise AND assignment	        a &= b	
		// bitwise OR assignment	        a |= b	
		// bitwise XOR assignment	        a ^= b	
		// bitwise left shift assignment	a <<= b	
		// bitwise right shift assignment   a >>= b

		//=========================================================
		// Increment/decrement operators
		//=========================================================
		// pre-increment    : 	++a		T&
		double2x4& operator ++ ()                               { ++this->c0; ++this->c1; ++this->c2; ++this->c3; return (*this);   }
		// pre - decrement  : 	--a		T&
		double2x4& operator -- ()                               { --this->c0; --this->c1; --this->c2; --this->c3; return (*this);   }
		// post-increment   : 	a++
		double2x4  operator ++ (int)                            { auto temp = *this; ++(*this); return (temp); }
		// post-decrement   :	a--
		double2x4  operator -- (int)							{ auto temp = *this; --(*this); return (temp); }

		//=========================================================
		// Arithmetic operators
		//=========================================================
		// unary plus       :   +a
		double2x4 operator + () const                           { return double2x4(+this->c0, +this->c1, +this->c2, +this->c3); }
		// unary minus      :   -a
		double2x4 operator - () const                           { return double2x4(-this->c0, -this->c1, -this->c2, -this->c3); }

		//=========================================================
		// Arithmetic operators
		//=========================================================
		// unary plus       :   +a
		// unary minus      :   -a
		// addition         :   a + b
        double2x4 operator + (const double2x4& rhs)		const	{ return double2x4(this->c0 + rhs.c0,	this->c1 + rhs.c1,	this->c2 + rhs.c2,	this->c3 + rhs.c3); }
        double2x4 operator + (double rhs)				const	{ return double2x4(this->c0 + rhs,		this->c1 + rhs,		this->c2 + rhs,		this->c3 + rhs); }
		friend double2x4 operator + (double lhs, const double2x4& rhs) { return double2x4(lhs + rhs.c0, lhs + rhs.c1, lhs + rhs.c2, lhs + rhs.c3); }

		// subtraction      :   a - b
        double2x4 operator - (const double2x4& rhs)		const	{ return double2x4(this->c0 - rhs.c0,	this->c1 - rhs.c1,	this->c2 - rhs.c2,	this->c3 - rhs.c3); }
        double2x4 operator - (double rhs)				const	{ return double2x4(this->c0 - rhs,		this->c1 - rhs,		this->c2 - rhs,		this->c3 - rhs); }
		friend double2x4 operator - (double lhs, const double2x4& rhs) { return double2x4(lhs - rhs.c0, lhs - rhs.c1, lhs - rhs.c2, lhs - rhs.c3); }

		// multiplication   :   a * b
        double2x4 operator * (const double2x4& rhs)		const	{ return double2x4(this->c0 * rhs.c0,	this->c1 * rhs.c1,	this->c2 * rhs.c2,	this->c3 * rhs.c3); }
        double2x4 operator * (double rhs)				const	{ return double2x4(this->c0 * rhs,		this->c1 * rhs,		this->c2 * rhs,		this->c3 * rhs); }
        friend double2x4 operator * (double lhs, const double2x4& rhs) { return double2x4(lhs * rhs.c0, lhs * rhs.c1, lhs * rhs.c2, lhs * rhs.c3); }

		// division         :   a / b
        double2x4 operator / (const double2x4& rhs)		const	{ return double2x4(this->c0 / rhs.c0,	this->c1 / rhs.c1,	this->c2 / rhs.c2,	this->c3 / rhs.c3); }
        double2x4 operator / (double rhs)				const	{ return double2x4(this->c0 / rhs,		this->c1 / rhs,		this->c2 / rhs,		this->c3 / rhs); }
		friend double2x4 operator / (double lhs, const double2x4& rhs) { return double2x4(lhs / rhs.c0, lhs / rhs.c1, lhs / rhs.c2, lhs / rhs.c3); }

		// modulo           :   a % b
        double2x4 operator % (const double2x4& rhs)		const	{ return double2x4(this->c0 % rhs.c0,	this->c1 % rhs.c1,	this->c2 % rhs.c2,	this->c3 % rhs.c3); }
        double2x4 operator % (double rhs)				const	{ return double2x4(this->c0 % rhs,		this->c1 % rhs,		this->c2 % rhs,		this->c3 % rhs); }
		friend double2x4 operator % (double lhs, const double2x4& rhs) { return double2x4(lhs % rhs.c0, lhs % rhs.c1, lhs % rhs.c2, lhs % rhs.c3); }

		// bitwise NOT      :   ~a
		// bitwise AND      :   a & b
		// bitwise OR       :   a | b
		// bitwise XOR      :   a ^ b
		// bitwise left shift : a << b
		// bitwise right shift: a >> b

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
        bool2x4 operator == (const double2x4& rhs)		const	{ return bool2x4(this->c0 == rhs.c0,	this->c1 == rhs.c1, this->c2 == rhs.c2, this->c3 == rhs.c3); }
        bool2x4 operator == (double rhs)				const	{ return bool2x4(this->c0 == rhs,		this->c1 == rhs,	this->c2 == rhs,	this->c3 == rhs); }
		friend bool2x4 operator == (double lhs, const double2x4& rhs) { return bool2x4(lhs == rhs.c0, lhs == rhs.c1, lhs == rhs.c2, lhs == rhs.c3); }

		// not equal to     :   a != b
		bool2x4 operator != (const double2x4& rhs)		const	{ return bool2x4(this->c0 != rhs.c0,	this->c1 != rhs.c1, this->c2 != rhs.c2, this->c3 != rhs.c3); }
        bool2x4 operator != (double rhs)				const	{ return bool2x4(this->c0 != rhs,		this->c1 != rhs,	this->c2 != rhs,	this->c3 != rhs); }	
		friend bool2x4 operator != (double lhs, const double2x4& rhs) { return bool2x4(lhs != rhs.c0, lhs != rhs.c1, lhs != rhs.c2, lhs != rhs.c3); }

		// less than        :   a < b
        bool2x4 operator < (const double2x4& rhs)		const	{ return bool2x4(this->c0 < rhs.c0,		this->c1 < rhs.c1,	this->c2 < rhs.c2,	this->c3 < rhs.c3); }
        bool2x4 operator < (double rhs)					const	{ return bool2x4(this->c0 < rhs,		this->c1 < rhs,		this->c2 < rhs,		this->c3 < rhs); }
		friend bool2x4 operator < (double lhs, const double2x4& rhs) { return bool2x4(lhs < rhs.c0, lhs < rhs.c1, lhs < rhs.c2, lhs < rhs.c3); }

		// greater than     :   a > b
        bool2x4 operator > (const double2x4& rhs)		const	{ return bool2x4(this->c0 > rhs.c0,		this->c1 > rhs.c1,	this->c2 > rhs.c2,	this->c3 > rhs.c3); }
        bool2x4 operator > (double rhs)					const	{ return bool2x4(this->c0 > rhs,		this->c1 > rhs,		this->c2 > rhs,		this->c3 > rhs); }
		friend bool2x4 operator > (double lhs, const double2x4& rhs) { return bool2x4(lhs > rhs.c0, lhs > rhs.c1, lhs > rhs.c2, lhs > rhs.c3); }

		// less than or equal to    : a <= b	
		bool2x4 operator <= (const double2x4& rhs)		const	{ return bool2x4(this->c0 <= rhs.c0,	this->c1 <= rhs.c1, this->c2 <= rhs.c2, this->c3 <= rhs.c3); }
        bool2x4 operator <= (double rhs)				const	{ return bool2x4(this->c0 <= rhs,		this->c1 <= rhs,	this->c2 <= rhs,	this->c3 <= rhs); }
		friend bool2x4 operator <= (double lhs, const double2x4& rhs) { return bool2x4(lhs <= rhs.c0, lhs <= rhs.c1, lhs <= rhs.c2, lhs <= rhs.c3); }

		// greater than or equal to : a >= b
        bool2x4 operator >= (const double2x4& rhs)		const	{ return bool2x4(this->c0 >= rhs.c0,	this->c1 >= rhs.c1, this->c2 >= rhs.c2, this->c3 >= rhs.c3); }
        bool2x4 operator >= (double rhs)				const	{ return bool2x4(this->c0 >= rhs,		this->c1 >= rhs,	this->c2 >= rhs,	this->c3 >= rhs); }
		friend bool2x4 operator >= (double lhs, const double2x4& rhs) { return bool2x4(lhs >= rhs.c0, lhs >= rhs.c1, lhs >= rhs.c2, lhs >= rhs.c3); }

		//=========================================================
		// Conversion operators
		//=========================================================


		//=========================================================
		// member access
		//=========================================================
		/// <summary>Returns the double2 element at a specified index.</summary>
		double2& operator[] (int index)
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
        /// <summary>Returns true if the double2x4 is equal to a given double2x4, false otherwise.</summary>
        /// <param name="rhs">Right hand side argument to compare equality with.</param>
        /// <returns>The result of the equality comparison.</returns>
        bool Equals(const double2x4& rhs) const { return c0.Equals(rhs.c0) && c1.Equals(rhs.c1) && c2.Equals(rhs.c2) && c3.Equals(rhs.c3); }

        /// <summary>Returns true if the double2x4 is equal to a given double2x4, false otherwise.</summary>
        /// <param name="o">Right hand side argument to compare equality with.</param>
        /// <returns>The result of the equality comparison.</returns>
        //override bool Equals(object o) { return o is double2x4 converted && Equals(converted); }

        /// <summary>Returns a hash code for the double2x4.</summary>
        /// <returns>The computed hash code.</returns>
        int GetHashCode() { return (int)math::hash(*this); }

        /// <summary>Returns a string representation of the double2x4.</summary>
        /// <returns>String representation of the value.</returns>
        std::string ToString() const
        {
            return std::format("double2x4({0}, {1}, {2}, {3},  {4}, {5}, {6}, {7})", c0.x, c1.x, c2.x, c3.x, c0.y, c1.y, c2.y, c3.y);
        }
	};
	__declspec(selectany) const double2x4	double2x4::zero = double2x4(0, 0, 0, 0);
#pragma pack(pop)


} // namespace ecs
