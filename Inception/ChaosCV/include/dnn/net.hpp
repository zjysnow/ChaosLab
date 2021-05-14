#pragma once

#include "core/core.hpp"

#include "dnn/layer.hpp"

#include <map>

namespace chaos
{
	class CHAOS_API Node
	{
	public:
		Node();

		std::string producer{};
		std::vector<std::string> consumers{};
	};

	template<class Type>
	inline static std::shared_ptr<Layer> LayerCast(const Type& layer)
	{
		return std::make_shared<Type>(layer);
	}


	class Extractor;
	class CHAOS_API Net
	{
	public:
		virtual ~Net() = default;
		Extractor CreateExtractor() const;
		
		template<class...Layers>
		static Net Sequential(Layers... layer)
		{
			Net net;
			net.layers = { LayerCast(layer)... };
			return net;
		}

	protected:
		

		void ForwardTo(const std::string& name, std::vector<Tensor>& blobs, const Option& opt) const;

		friend class Extractor;
		
		std::vector<Ptr<Layer>> layers;
	};

	class CHAOS_API Extractor
	{
	public:
		virtual ~Extractor();


	protected:
		Extractor(const Net* net);

		friend Extractor Net::CreateExtractor() const;
		const Net* net;

		std::vector<Tensor> blobs;
	};
}