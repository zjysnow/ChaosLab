#include "dnn/layers/data.hpp"

namespace chaos
{
	namespace dnn
	{
		Data::Data() : Layer("Data") { support_inplace = true; }

		void Data::Forward(std::vector<Tensor>&, const Option&) const {}
	}
}