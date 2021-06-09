#pragma once

#include "dnn/layer.hpp"

namespace chaos::inline dnn
{
	class CHAOS_API Invert : public Layer
	{
	public:
		Invert();
		void CreatePipeline(const Option&) override;
		void DestroyPipeline(const Option&) override;

		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

		Ptr<Layer> svd;
		Ptr<Layer> backsubst;
	};
}