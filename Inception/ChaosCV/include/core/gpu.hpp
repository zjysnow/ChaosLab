#pragma once

#include "core/core.hpp"

#include <vulkan/vulkan.hpp>

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
		int support_VK_KHR_surface = 0;
		int support_VK_KHR_win32_surface = 0;
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

		// hardware limit
		size_t memory_map_alignment;
		size_t buffer_offset_alignment;
		size_t buffer_image_granularity;
		size_t non_coherent_atom_size;
		uint32_t max_image_dimension_1d;
		uint32_t max_image_dimension_2d;
		uint32_t max_image_dimension_3d;

		// here just 4 queue families: compute, graphics, transfer, sparsebinding

		uint32_t graphics_queue_family_index;
		uint32_t transfer_queue_family_index;
		uint32_t compute_queue_family_index;
		uint32_t graphics_queue_count;
		uint32_t transfer_queue_count;
		uint32_t compute_queue_count;

		int support_VK_KHR_swapchain;
		int support_VK_KHR_get_memory_requirements2;
	};

	CHAOS_API int GetGPUCount();
	CHAOS_API int GetDefaultGPUIndex();
	CHAOS_API const GPUInfo& GetGPUInfo(int device_index = GetDefaultGPUIndex());

	class GraphicsPipeline;
	class CHAOS_API VulkanDevice
	{
	public:
		VulkanDevice(int device_index = GetDefaultGPUIndex());
		~VulkanDevice();

		const GPUInfo& info;

		VkDevice GetDevice() const noexcept { return device; }

		//PipelineCache* GetPipelineCache() const noexcept;

		// device extensions
		PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR; // create swap chain
		PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR; // destroy swap chain

		uint32 FindMemoryTypeIndex(uint32 memory_type_bits, const VkFlags& required, const VkFlags& preferred, const VkFlags& preferred_not) const;

		VkQueue AcquireQueue(uint32 queue_family_index) const;
		void ReclaimQueue(uint32 queue_family_index, VkQueue queue) const;
	protected:
		void InitDeviceExtension();

	private:
		VkDevice device;

		// hardware queue
		mutable std::vector<VkQueue> compute_queues;
		mutable std::vector<VkQueue> graphics_queues;
		mutable std::vector<VkQueue> transfer_queues;
		mutable std::mutex queue_lock;
	};
	
	CHAOS_API VulkanDevice* GetGPUDevice(int device_index = GetDefaultGPUIndex());

}