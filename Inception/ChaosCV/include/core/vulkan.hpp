#pragma once

#include "core/def.hpp"
#include "core/types.hpp"

#define DEFINE_VULKAN_HANDLE(object)    \
struct object##_T;                      \
using object=object##_T*;

DEFINE_VULKAN_HANDLE(VkInstance)
DEFINE_VULKAN_HANDLE(VkDevice)
DEFINE_VULKAN_HANDLE(VkPhysicalDevice)

DEFINE_VULKAN_HANDLE(VkQueue)

DEFINE_VULKAN_HANDLE(VkBuffer)
DEFINE_VULKAN_HANDLE(VkDeviceMemory)

DEFINE_VULKAN_HANDLE(VkPipeline)
DEFINE_VULKAN_HANDLE(VkPipelineLayout)
DEFINE_VULKAN_HANDLE(VkDescriptorSetLayout)

DEFINE_VULKAN_HANDLE(VkShaderModule)

DEFINE_VULKAN_HANDLE(VkCommandPool)
DEFINE_VULKAN_HANDLE(VkCommandBuffer)
DEFINE_VULKAN_HANDLE(VkFence)
DEFINE_VULKAN_HANDLE(VkSemaphore)
DEFINE_VULKAN_HANDLE(VkDescriptorPool)

DEFINE_VULKAN_HANDLE(VkDebugUtilsMessengerEXT)

namespace chaos
{
	using MemoryPropertyFlag = Flag;
	using BufferUsageFlag = Flag;
	using ShaderStageFlag = Flag;
	using DescriptorType = Flag;
	using AccessFlag = Flag;
	using PipelineStageFlag = Flag;

	union VulkanSpecializationType
	{
		int i;
		float f;
		uint32 u32;
	};

	union VulkanConstantType
	{
		int i;
		float f;
	};

	class VulkanBufferMemory
	{
	public:
		VkBuffer buffer = nullptr;
		VkDeviceMemory memory = nullptr;
		size_t capacity = 0;
		size_t offset = 0;
		void* mapped_data = nullptr;
		int ref_cnt = 0;
	};
}