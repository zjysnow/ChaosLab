#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class Decomp : public Layer
		{
		public:
			Decomp();
			
		};

		class LUP : public Decomp
		{
		public:

		};

		class SVD : public Decomp
		{
		public:

		};
	}
}