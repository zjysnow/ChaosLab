#pragma once

#include "core/core.hpp"
#include "dnn/layer.hpp"

#include <map>
#include <functional>

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API LayerRegistry
		{
		public:
			using Creator = std::function<Ptr<Layer>()>;
			using CreatorRegistry = std::map<std::string, Creator>;

			static CreatorRegistry& Registry()
			{
				static CreatorRegistry registry;
				return registry;
			}

			static void AddCreator(const std::string& type, Creator creator)
			{
				auto& registry = Registry();
				CHECK_EQ(0, registry.count(type)) << "Layer type " << type << " already registered";
				registry[type] = creator;
			}

			static Ptr<Layer> CreateLayer(const std::string& type)
			{
				auto& registry = Registry();
				CHECK_EQ(1, registry.count(type)) << "unknown layer type: " << type << "(known types: " << LayerTypeList() << ")";
				return registry[type]();
			}

			static std::vector<std::string> LayerTypeList()
			{
				auto& registry = Registry();
				std::vector<std::string> layer_types;
				for (const auto& [type, creator] : registry)
				{
					layer_types.push_back(type);
				}
				return layer_types;
			}

		private:
			LayerRegistry() = default;
		};

		class LayerRegisterer
		{
		public:
			LayerRegisterer(const std::string& type, const LayerRegistry::Creator& creator)
			{
				LayerRegistry::AddCreator(type, creator);
			}
		};
	}
}

#define REGISTER_LAYER(type)						\
chaos::Ptr<chaos::dnn::Layer> Creator_##type##Layer()	\
{													\
	return std::make_shared<##type>();		\
}													\
static chaos::dnn::LayerRegisterer g_creator_##type(#type, Creator_##type##Layer);