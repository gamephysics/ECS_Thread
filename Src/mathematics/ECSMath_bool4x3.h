﻿#pragma once

//=============================================================================
// ECS MATH 
//=============================================================================
namespace ecs
{
	namespace math
	{
		//=========================================================
		// math-bool4x3
		//=========================================================
		/// <summary>Return the bool3x4 transpose of a bool4x3 matrix.</summary>
		/// <param name="v">Value to transpose.</param>
		/// <returns>Transposed value.</returns>
		bool3x4 transpose(const bool4x3& v);

		/// <summary>Returns a uint hash code of a bool4x3 matrix.</summary>
		/// <param name="v">Matrix value to hash.</param>
		/// <returns>uint hash of the argument.</returns>
		uint hash(const bool4x3& v);

		/// <summary>
		/// Returns a uint4 vector hash code of a bool4x3 matrix.
		/// When multiple elements are to be hashes together, it can more efficient to calculate and combine wide hash
		/// that are only reduced to a narrow uint hash at the very end instead of at every step.
		/// </summary>
		/// <param name="v">Matrix value to hash.</param>
		/// <returns>uint4 hash of the argument.</returns>
		uint4 hashwide(const bool4x3& v);
	}

    // https://github.com/Unity-Technologies/Unity.Mathematics/tree/master/src/Unity.Mathematics/bool4x3.gen.cs
#pragma pack(push,1)
	struct bool4x3
	{
	public:
		union {
			struct
			{
				bool4 c0;	/// <summary>Column 0 of the matrix.</summary>
				bool4 c1;	/// <summary>Column 1 of the matrix.</summary>
				bool4 c2;	/// <summary>Column 2 of the matrix.</summary>
			};
			bool4		data[3];
		};
	public:
		//=========================================================
		// CONSTRUCTOR
		//=========================================================
        bool4x3() : c0(false), c1(false), c2(false) {}
        bool4x3(const bool4& _c0, const bool4& _c1, const bool4& _c2)   { c0 = _c0;	c1 = _c1; c2 = _c2; }

		
        /// <summary>Constructs a bool4x3 matrix from 12 bool values given in row-major order.</summary>
        /// mXY : The matrix at row X, column Y will be set to this value		
        bool4x3(bool m00, bool m01, bool m02,
                bool m10, bool m11, bool m12,
                bool m20, bool m21, bool m22,
                bool m30, bool m31, bool m32)
        {
            c0 = bool4(m00, m10, m20, m30);
            c1 = bool4(m01, m11, m21, m31);
            c2 = bool4(m02, m12, m22, m32);
        }

        /// <summary>Constructs a bool4x3 matrix from a single bool value by assigning it to every component.</summary>
        /// <param name="v">bool to convert to bool4x3</param>
        bool4x3(bool v)                     {   c0 = v;     c1 = v;     c2 = v;     }


        //=========================================================
        // OPERATORS
        //=========================================================
        //=========================================================
        // Assignment operators : T&
        //=========================================================
        // simple assignment	            a = b	
        // addition assignment	            a += b	
        // subtraction assignment	        a -= b	
        // multiplication assignment	    a *= b	
        // division assignment	            a /= b	
        // modulo assignment	            a %= b	

        // bitwise AND assignment	        a &= b	
        // bitwise OR assignment	        a |= b	
        // bitwise XOR assignment	        a ^= b	
        // bitwise left shift assignment	a <<= b	
        // bitwise right shift assignment   a >>= b

        //=========================================================
        // Increment/decrement operators
        //=========================================================
        // pre-increment    : 	++a		T&
        // pre - decrement  : 	--a		T&
        // post-increment   : 	a++
        // post-decrement   :	a--

        //=========================================================
        // Arithmetic operators
        //=========================================================
        // unary plus       :   +a
        // unary minus      :   -a
        // addition         :   a + b
        // subtraction      :   a - b
        // multiplication   :   a * b
        // division         :   a / b
        // modulo           :   a % b

        // bitwise NOT      :   ~a
        // bitwise AND      :   a & b
        bool4x3 operator & (const bool4x3& rhs)     const   { return bool4x3(this->c0 & rhs.c0,     this->c1 & rhs.c1,      this->c2 & rhs.c2); }
        bool4x3 operator & (bool rhs)               const   { return bool4x3(this->c0 & rhs,        this->c1 & rhs,         this->c2 & rhs); }
        friend bool4x3 operator & (bool lhs, const bool4x3& rhs) { return bool4x3(lhs & rhs.c0, lhs & rhs.c1, lhs & rhs.c2); }

        // bitwise OR       :   a | b
        bool4x3 operator | (const bool4x3& rhs)     const   { return bool4x3(this->c0 | rhs.c0,     this->c1 | rhs.c1,      this->c2 | rhs.c2); }
        bool4x3 operator | (bool rhs)               const   { return bool4x3(this->c0 | rhs,        this->c1 | rhs,         this->c2 | rhs); }
        friend bool4x3 operator | (bool lhs, const bool4x3& rhs) { return bool4x3(lhs | rhs.c0, lhs | rhs.c1, lhs | rhs.c2); }

        // bitwise XOR      :   a ^ b
        bool4x3 operator ^ (const bool4x3& rhs)     const   { return bool4x3(this->c0 ^ rhs.c0,     this->c1 ^ rhs.c1,      this->c2 ^ rhs.c2); }
        bool4x3 operator ^ (bool rhs)               const   { return bool4x3(this->c0 ^ rhs,        this->c1 ^ rhs,         this->c2 ^ rhs); }
        friend bool4x3 operator ^ (bool lhs, const bool4x3& rhs) { return bool4x3(lhs ^ rhs.c0, lhs ^ rhs.c1, lhs ^ rhs.c2); }
        
        // bitwise left shift : a << b
        // bitwise right shift: a >> b
        //=========================================================
        // Logical operators
        //=========================================================
        // negation	        :   not a, !a
        bool4x3 operator ! () const                         { return bool4x3(!this->c0, !this->c1, !this->c2); }
        // AND	            :   a and b, a && b
        // inclusive OR	    :   a or b,  a || b

        //=========================================================
        // Comparison operators
        //=========================================================
        // equal to         :   a == 
        bool4x3 operator == (const bool4x3& rhs)    const   { return bool4x3(this->c0 == rhs.c0,    this->c1 == rhs.c1,     this->c2 == rhs.c2); }
        bool4x3 operator == (bool rhs)              const   { return bool4x3(this->c0 == rhs,       this->c1 == rhs,        this->c2 == rhs); }
        friend bool4x3 operator == (bool lhs, const bool4x3& rhs) { return bool4x3(lhs == rhs.c0, lhs == rhs.c1, lhs == rhs.c2); }

        // not equal to     :   a != b
        bool4x3 operator != (const bool4x3& rhs)    const   { return bool4x3(this->c0 != rhs.c0,    this->c1 != rhs.c1,     this->c2 != rhs.c2); }
        bool4x3 operator != (bool rhs)              const   { return bool4x3(this->c0 != rhs,       this->c1 != rhs,        this->c2 != rhs); }
        friend bool4x3 operator != (bool lhs, const bool4x3& rhs) { return bool4x3(lhs != rhs.c0, lhs != rhs.c1, lhs != rhs.c2); }

        // less than        :   a < b
        // greater than     :   a > b
        // less than or equal to    : a <= b	
        // greater than or equal to : a >= b

        //=========================================================
        // Conversion operators
        //=========================================================


        //=========================================================
        // member access
        //=========================================================
        bool4& operator[] (int index)
		{
#if defined(_DEBUG ) || defined(_DEVELOPMENT)
			if (index >= 3)
				throw std::exception("index must be between[0...2]");
#endif
			return data[index];
		}

        //=========================================================
        // METHOD
        //=========================================================
        /// <summary>Returns true if the bool4x3 is equal to a given bool4x3, false otherwise.</summary>
        /// <param name="rhs">Right hand side argument to compare equality with.</param>
        /// <returns>The result of the equality comparison.</returns>
        bool Equals(const bool4x3& rhs) const { return c0.Equals(rhs.c0) && c1.Equals(rhs.c1) && c2.Equals(rhs.c2); }

        /// <summary>Returns true if the bool4x3 is equal to a given bool4x3, false otherwise.</summary>
        /// <param name="o">Right hand side argument to compare equality with.</param>
        /// <returns>The result of the equality comparison.</returns>
        //override bool Equals(object o) { return o is bool4x3 converted && Equals(converted); }

        /// <summary>Returns a hash code for the bool4x3.</summary>
        /// <returns>The computed hash code.</returns>
        int GetHashCode() { return (int)math::hash(*this); }

        /// <summary>Returns a string representation of the bool4x3.</summary>
        /// <returns>String representation of the value.</returns>
        std::string ToString() const
        {
            return std::format("bool4x3({0}, {1}, {2},  {3}, {4}, {5},  {6}, {7}, {8},  {9}, {10}, {11})", c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z, c0.w, c1.w, c2.w);
        }
	};
#pragma pack(pop)


} // namespace ecs