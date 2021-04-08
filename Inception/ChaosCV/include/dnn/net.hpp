#pragma once

#include "core/core.hpp"

#include "dnn/layer.hpp"

namespace chaos
{
	class CHAOS_API Net
	{
	public:

		void CreateEx() const;

		std::vector<Ptr<Layer>> layers;
	};

	class CHAOS_API Ex
	{
	public:
		Ex(const Net* net);

		const Net* net;
	};
}