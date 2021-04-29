#include "highgui/highgui.hpp"

#include "utils/op.hpp"

namespace chaos
{
	Tensor Rotate(const Tensor& m, const float angle, const Tensor& v)
	{
		const float a = angle;
		const float c = cos(a);
		const float s = sin(a);

		Tensor axis = Operator::L2Norm(v);
		Tensor temp = Operator::Mul(axis, 1 - c); //axis * (1 - c);

		Tensor Rotate = Tensor(Shape(3u, 3u), DataType::D4, Packing::CHW);
		Rotate[0] = c + temp[0] * axis[0];
		Rotate[1] = temp[0] * axis[1] + s * axis[2];
		Rotate[2] = temp[0] * axis[2] - s * axis[1];

		Rotate[3] = temp[1] * axis[0] - s * axis[2];
		Rotate[4] = c + temp[1] * axis[1];
		Rotate[5] = temp[1] * axis[2] + s * axis[0];

		Rotate[6] = temp[2] * axis[0] + s * axis[1];
		Rotate[7] = temp[2] * axis[1] - s * axis[0];
		Rotate[8] = c + temp[2] * axis[2];

		Tensor Result = Tensor::zeros(Shape(4, 4));

		Result[0] = m[0] * Rotate[0] + m[4] * Rotate[1] + m[8] * Rotate[2];
		Result[1] = m[1] * Rotate[0] + m[5] * Rotate[1] + m[9] * Rotate[2];
		Result[2] = m[2] * Rotate[0] + m[6] * Rotate[1] + m[10] * Rotate[2];

		Result[4] = m[0] * Rotate[3] + m[4] * Rotate[4] + m[8] * Rotate[5];
		Result[5] = m[1] * Rotate[3] + m[5] * Rotate[4] + m[9] * Rotate[5];
		Result[6] = m[2] * Rotate[3] + m[6] * Rotate[4] + m[10] * Rotate[5];

		Result[8] = m[0] * Rotate[6] + m[4] * Rotate[7] + m[8] * Rotate[8];
		Result[9] = m[1] * Rotate[6] + m[5] * Rotate[7] + m[9] * Rotate[8];
		Result[10] = m[2] * Rotate[6] + m[6] * Rotate[7] + m[10] * Rotate[8];


		Result[12] = m[12];
		Result[13] = m[13];
		Result[14] = m[14];
		Result[15] = m[15];

		return Result;
	}

	Tensor LookAt(const Tensor& eye, const Tensor& center, const Tensor& up) // look at left hand
	{
		const Tensor f = Operator::L2Norm(Operator::Sub(center, eye));
		const Tensor s = Operator::L2Norm(Operator::Cross(up, f));
		const Tensor u = Operator::Cross(f, s);

		Tensor Result = Tensor::eye(4, 4);
		Result[0] = s[0]; // .x;
		Result[4] = s[1]; // .y;
		Result[8] = s[2]; // .z;
		Result[1] = u[0]; // .x;
		Result[5] = u[1]; // .y;
		Result[9] = u[2]; // .z;
		Result[2] = f[0]; // .x;
		Result[6] = f[1]; // .y;
		Result[10] = f[2]; // .z;
		Result[12] = -Operator::Dot(s, eye);
		Result[13] = -Operator::Dot(u, eye);
		Result[14] = -Operator::Dot(f, eye);
		return Result;
	}

	Tensor Perspective(float fovy, float aspect, float z_near, float z_far) // rh no
	{
		const float tan_half_fovy = std::tan(fovy / 2.f);

		Tensor Result = Tensor::zeros(Shape(4, 4));

		Result[0] = 1.f / (aspect * tan_half_fovy);
		Result[5] = 1.f / (tan_half_fovy);
		Result[10] = -(z_far + z_near) / (z_far - z_near);
		Result[11] = -1.f;
		Result[14] = -(2.f * z_far * z_near) / (z_far - z_near);
		return Result;
	}
	
}