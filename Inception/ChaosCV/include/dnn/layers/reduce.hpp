#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class CHAOS_API Reduce : public Layer
		{
		public:
			enum OpType
			{
				SUM,
				AVG,
				MAX,
				MIN,
			};

			Reduce();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const;
			void Set(const std::string& pname, const std::any& param);

			int op_type = SUM;
			bool all = false;
			Array<int> vecdim = { 0 };
			float alpha = 1.f;
		};
	}
}