#include "dnn/layers/noop.hpp"

namespace chaos
{
	Noop::Noop() : Layer(L"Noop")
	{
		support_inplace = true;
	}

	void Noop::Forward(std::vector<Tensor>&, const Option&) const {}
}