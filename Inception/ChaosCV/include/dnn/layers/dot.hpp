#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Dot : public Layer
		{
		public:
			Dot();
			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt = Option()) const override;
			void Set(const std::string& pname, const std::any& val) final;

			//size_t bottoms_count() const final { return 2; }
			//size_t tops_count() const final { return 1; }

			bool all = false;
			std::vector<uint32> vecdim = { 0 };
		};
	}
}