#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	enum FrontFace
	{
		FRONT_FACE_COUNTER_CLOCKWISE = 0,
		FRONT_FACE_CLOCKWISE = 1,
		FRONT_FACE_MAX_ENUM = 0x7FFFFFFF
	};

	enum CullModeFlag
	{
		CULL_MODE_NONE = 0,
		CULL_MODE_FRONT_BIT = 0x00000001,
		CULL_MODE_BACK_BIT = 0x00000002,
		CULL_MODE_FRONT_AND_BACK = 0x00000003,
		CULL_MODE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
	};

	enum PolygonMode
	{
		POLYGON_MODE_FILL = 0,
		POLYGON_MODE_LINE = 1,
		POLYGON_MODE_POINT = 2,
		POLYGON_MODE_FILL_RECTANGLE_NV = 1000153000,
		POLYGON_MODE_MAX_ENUM = 0x7FFFFFFF
	};

	enum PrimitiveTopology {
		PRIMITIVE_TOPOLOGY_POINT_LIST = 0,
		PRIMITIVE_TOPOLOGY_LINE_LIST = 1,
		PRIMITIVE_TOPOLOGY_LINE_STRIP = 2,
		PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3,
		PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 4,
		PRIMITIVE_TOPOLOGY_TRIANGLE_FAN = 5,
		PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY = 6,
		PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY = 7,
		PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY = 8,
		PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY = 9,
		PRIMITIVE_TOPOLOGY_PATCH_LIST = 10,
		PRIMITIVE_TOPOLOGY_MAX_ENUM = 0x7FFFFFFF
	};

	/// <summary>
	/// <para>Build a rotation 4 * 4 matrix created from an axis vector and an angle</para>
	/// <para>see <a href="https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glRotate.xml">glRotate man page</a></para>
	/// </summary>
	/// <param name="m">Input matrix multiplied by this rotation matrix</param>
	/// <param name="angle">Rotation angle expressed in radians</param>
	/// <param name="v">Rotation axis, recommended to be normalized</param>
	/// <returns>rotation 4 * 4 matrix</returns>
	CHAOS_API Tensor Rotate(const Tensor& m, const float angle, const Tensor& v);

	/// <summary>
	/// Build a right handed look at view matrix
	/// </summary>
	/// <param name="eye">Positon of teh camera</param>
	/// <param name="center">center Positon where the camera is looking at</param>
	/// <param name="up">up Normalized up vector, how the camera is oriented. Typically (0, 0, 1)</param>
	/// <returns>view matrix</returns>
	CHAOS_API Tensor LookAt(const Tensor& eye, const Tensor& center, const Tensor& up);

	/// <summary>
	/// <para>Creates a matrix for a right handed, symetric perspective-view frustum.</para>
	/// <para>The near and far clip planes correspond to z normalized device coordinates of -1 and +1 respectively. (OpenGL clip volume definition)</para>
	/// <para>see <a href="https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml">gluPerspective man page</a></para>
	/// </summary>
	/// <param name="fovy">the field of view angle, in degrees, in the y direction. Expressed in radians.</param>
	/// <param name="aspect">the aspect ratio that determines the field of view in the x direction. The aspect ratio is the ratio of x (width) to y (height).</param>
	/// <param name="z_near">the distance from the viewer to the near clipping plane (always positive).</param>
	/// <param name="z_far">the distance from the viewer to the far clipping plane (always positive).</param>
	/// <returns>a symetric perspective-view frustum matrix</returns>
	CHAOS_API Tensor Perspective(float fovy, float aspect, float z_near, float z_far);
}