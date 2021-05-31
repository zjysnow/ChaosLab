#pragma once

#include "core/core.hpp"
#include "core/vulkan.hpp"

#include <mutex>

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

	CHAOS_API int GetGPUCount();
	CHAOS_API int GetDefaultGPUIndex();

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

		// see GPUInfo::Type
		int type;
		std::string device_name;

		std::vector<MemoryPropertyFlag> memory_properties;

		uint32 graphics_queue_family_index;
		uint32 transfer_queue_family_index;
		uint32 compute_queue_family_index;
		uint32 graphics_queue_count;
		uint32 transfer_queue_count;
		uint32 compute_queue_count;

	};
	CHAOS_API const GPUInfo& GetGPUInfo(int device_index);

	class CHAOS_API VulkanDevice
	{
	public:
		VulkanDevice(int device_index = GetDefaultGPUIndex());
		~VulkanDevice();

		const GPUInfo& info;

		VkDevice GetDevice() const noexcept { return device; }

		uint32 FindMemoryTypeIndex(uint32 memory_type_bits, int required, int preferred, int preferred_not) const;
		bool IsMemoryMappable(uint32 memory_type_index) const;
		bool IsMemoryCoherent(uint32 memory_type_index) const;

		VkQueue AcquireQueue(uint32 queue_family_index) const;
		void ReclaimQueue(uint32 queue_family_index, VkQueue queue) const;
	private:
		VkDevice device;

		// hardware queue
		mutable std::vector<VkQueue> compute_queues;
		mutable std::vector<VkQueue> graphics_queues;
		mutable std::vector<VkQueue> transfer_queues;
		mutable std::mutex queue_lock;
	};

	CHAOS_API const VulkanDevice* GetGPUDevice(int device_index = GetDefaultGPUIndex());
}