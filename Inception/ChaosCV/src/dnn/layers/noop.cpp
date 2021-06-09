#include "dnn/layers/noop.hpp"

namespace chaos::inline dnn
{
	Noop::Noop() : Layer("Noop")
	{
		support_inplace = true;
		support_vulkan = true;
	}

	void Noop::Forward(const std::vector<Tensor>&, std::vector<Tensor>&, const Option&) const {}
	void Noop::Forward(std::vector<Tensor>&, const Option&) const {}

	void Noop::Forward(const std::vector<VulkanTensor>&, std::vector<VulkanTensor>&, ComputeCommand&, const Option&) const {}
	void Noop::Forward(std::vector<VulkanTensor>&, ComputeCommand&, const Option&) const {}
}