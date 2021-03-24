#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class Decomp : public Layer
	{
	public:
		enum
		{
			LUP,
			CHOLESKY,
			SVD,
			EIG,
			QR,
		};

		using Layer::Layer;
	};
}