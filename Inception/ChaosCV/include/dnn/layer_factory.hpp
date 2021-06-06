#pragma once

#include "core/core.hpp"
#include "dnn/layer.hpp"

#include <map>

namespace chaos
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

		static Ptr<Layer> CreateLayer(const std::string& type)
		{
			auto& registry = Registry();
			CHECK_EQ(registry.count(type), 1) << "unknown layer type: " << type << " (known types: " << ListLayerType() << ")";;
			return registry[type]();
		}


		static void AddCreator(const std::string& type, const Creator& creator)
		{
			auto& registry = Registry();
			CHECK_EQ(registry.count(type), 0) << "layer type " << type << " already registered.";
			registry[type] = creator;
		}

		static std::vector<std::string> ListLayerType()
		{
			// »ñµÃ×¢²á±í  
			auto& registry = Registry();
			std::vector<std::string> list(registry.size());
			for (size_t i = 0; const auto & [type, creator] : registry)
			{
				list[i++] = type;
			}
			return list;
		}

	private:
		LayerRegistry() = default;
	};

	class CHAOS_API LayerRegisterer
	{
	public:
		LayerRegisterer(const std::string& type, const std::function<Ptr<Layer>()>& creator)
		{
			LayerRegistry::AddCreator(type, creator);
		}
	};
}

#define REGISTER_LAYER(type, layer) \
static chaos::LayerRegisterer g_creator_##layer(type, [](){ return chaos::Ptr<chaos::Layer>(new layer()); });
