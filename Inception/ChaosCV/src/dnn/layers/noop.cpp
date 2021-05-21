#include "dnn/layers/noop.hpp"

namespace chaos
{
	namespace dnn
	{
		Noop::Noop() : Layer("Noop") { support_inplace = true; }
		void Noop::Forward(std::vector<Tensor>&, const Option&) const {}
	}
}