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

DEFINE_VULKAN_HANDLE(VkDebugUtilsMessengerEXT)

namespace chaos
{
	using MemoryPropertyFlag = Flag;
	using BufferUsageFlag = Flag;

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
		VkBuffer buffer;
		VkDeviceMemory memory;
		size_t capacity;
		size_t offset;
		void* mapped_data = nullptr;
		int ref_cnt;
	};
}