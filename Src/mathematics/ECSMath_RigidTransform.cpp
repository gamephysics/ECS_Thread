#include "ECS.h"

//=============================================================================
// ECS MATH 
//=============================================================================
namespace ecs
{
	namespace math
	{
		//=========================================================
		// math-RigidTransform
		//=========================================================
		/// <summary>Returns the inverse of a RigidTransform.</summary>
		/// <param name="t">The RigidTransform to invert.</param>
		/// <returns>The inverse RigidTransform.</returns>
		RigidTransform inverse(const RigidTransform& t)
		{
			quaternion invRotation = math::inverse(t.rot);
			float3 invTranslation = math::mul(invRotation, -t.pos);
			return RigidTransform(invRotation, invTranslation);
		}

		/// <summary>Returns the result of transforming the RigidTransform b by the RigidTransform a.</summary>
		/// <param name="a">The RigidTransform on the left.</param>
		/// <param name="b">The RigidTransform on the right.</param>
		/// <returns>The RigidTransform of a transforming b.</returns>
		RigidTransform mul(const RigidTransform& a, const RigidTransform& b)
		{
			return RigidTransform(math::mul(a.rot, b.rot), math::mul(a.rot, b.pos) + a.pos);
		}

		/// <summary>Returns the result of transforming a float4 homogeneous coordinate by a RigidTransform.</summary>
		/// <param name="a">The RigidTransform.</param>
		/// <param name="pos">The position to be transformed.</param>
		/// <returns>The transformed position.</returns>
		float4 mul(const RigidTransform& a, const float4& pos)
		{
			return float4(math::mul(a.rot, pos.xyz()) + a.pos * pos.w, pos.w);
		}

		/// <summary>Returns the result of rotating a float3 vector by a RigidTransform.</summary>
		/// <param name="a">The RigidTransform.</param>
		/// <param name="dir">The direction vector to rotate.</param>
		/// <returns>The rotated direction vector.</returns>
		float3 rotate(const RigidTransform& a, const float3& dir)
		{
			return math::mul(a.rot, dir);
		}

		/// <summary>Returns the result of transforming a float3 point by a RigidTransform.</summary>
		/// <param name="a">The RigidTransform.</param>
		/// <param name="pos">The position to transform.</param>
		/// <returns>The transformed position.</returns>
		float3 transform(const RigidTransform& a, const float3& pos)
		{
			return math::mul(a.rot, pos) + a.pos;
		}

		/// <summary>Returns a uint hash code of a RigidTransform.</summary>
		/// <param name="t">The RigidTransform to hash.</param>
		/// <returns>The hash code of the input RigidTransform</returns>
		uint hash(const RigidTransform& t)
		{
			return math::hash(t.rot) + 0xC5C5394Bu * math::hash(t.pos);
		}

		/// <summary>
		/// Returns a uint4 vector hash code of a RigidTransform.
		/// When multiple elements are to be hashes together, it can more efficient to calculate and combine wide hash
		/// that are only reduced to a narrow uint hash at the very end instead of at every step.
		/// </summary>
		/// <param name="t">The RigidTransform to hash.</param>
		/// <returns>The uint4 wide hash code.</returns>
		uint4 hashwide(const RigidTransform& t)
		{
			return math::hashwide(t.rot) + 0xC5C5394Bu * math::hashwide(t.pos).xyzz();
		}
	}

} // namespace ecs
