#pragma once

#include "core/core.hpp"

#include <vulkan/vulkan.hpp>

namespace chaos
{
	class CHAOS_API VulkanInstance
	{
	public:
		static VulkanInstance& GetInstance();

		operator VkInstance();
		VkInstance instance;

		VkDebugUtilsMessengerEXT debug_messenger;

		int support_VK_EXT_debug_utils = 0;
	protected:
		VulkanInstance();
		~VulkanInstance();

	private:
		static VulkanInstance holder;
	};

	CHAOS_API void CreateGPUInstance();
	CHAOS_API void DestroyGPUInstance();

	class CHAOS_API GPUInfo
	{
	public:
		enum Type
		{
			OTHER,
			DISCRETE,
			INTEGRATED,
			VIRTUAL,
			CPU, // see VK_PHYSICAL_DEVICE_TYPE_CPU
		};

		// vulkan physical device
		VkPhysicalDevice physical_device;

		// memory properties
		VkPhysicalDeviceMemoryProperties physical_device_memory_properties;

		// see GPUInfo::Type
		int type;

		uint32_t graphics_queue_family_index;
		uint32_t transfer_queue_family_index;
		uint32_t compute_queue_family_index;
		uint32_t graphics_queue_count;
		uint32_t transfer_queue_count;
		uint32_t compute_queue_count;
		
	};

	CHAOS_API int GetGPUCount();
	CHAOS_API int GetDefaultGPUIndex();
	CHAOS_API const GPUInfo& GetGPUInfo(int device_index = GetDefaultGPUIndex());

	class CHAOS_API VulkanDevice
	{
	public:
		VulkanDevice(int device_index = GetDefaultGPUIndex());
		~VulkanDevice();

		const GPUInfo& info;

	protected:

	private:
		VkDevice device;
	};
	
	CHAOS_API VulkanDevice* GetGPUDevice(int device_index = GetDefaultGPUIndex());
}