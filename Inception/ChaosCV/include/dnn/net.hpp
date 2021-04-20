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


	class Extractor;
	class CHAOS_API Net
	{
	public:
		virtual ~Net() = default;
		Extractor CreateExtractor() const;
		

	protected:
		void ForwardTo(const std::string& name, std::vector<Tensor>& blobs, const Option& opt) const;

		friend class Extractor;
		std::map<std::string, Ptr<Layer>> layers;
		
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