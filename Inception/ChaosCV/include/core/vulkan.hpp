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
DEFINE_VULKAN_HANDLE(VkPipelineCache)
DEFINE_VULKAN_HANDLE(VkPipelineLayout)

DEFINE_VULKAN_HANDLE(VkDescriptorSet)
DEFINE_VULKAN_HANDLE(VkDescriptorSetLayout)
DEFINE_VULKAN_HANDLE(VkDescriptorUpdateTemplate)

DEFINE_VULKAN_HANDLE(VkShaderModule)

DEFINE_VULKAN_HANDLE(VkCommandPool)
DEFINE_VULKAN_HANDLE(VkCommandBuffer)
DEFINE_VULKAN_HANDLE(VkFence)
DEFINE_VULKAN_HANDLE(VkSemaphore)
DEFINE_VULKAN_HANDLE(VkDescriptorPool)

DEFINE_VULKAN_HANDLE(VkSurfaceKHR)
DEFINE_VULKAN_HANDLE(VkSwapchainKHR)
DEFINE_VULKAN_HANDLE(VkFramebuffer)
DEFINE_VULKAN_HANDLE(VkRenderPass)

DEFINE_VULKAN_HANDLE(VkDebugUtilsMessengerEXT)

struct VkAllocationCallbacks;
struct VkDescriptorUpdateTemplateCreateInfo;


struct VkSurfaceFormatKHR;
struct VkSurfaceCapabilitiesKHR;
struct VkExtent2D;

enum VkResult;
enum VkDescriptorType;
enum VkFormat;
enum VkPolygonMode;
enum VkFrontFace;
enum VkPrimitiveTopology;
enum VkCullModeFlagBits;
enum VkShaderStageFlagBits;
enum VkPresentModeKHR;
enum VkPipelineStageFlagBits;
enum VkAccessFlagBits;

namespace chaos
{
	using MemoryPropertyFlag = Flag;
	using BufferUsageFlag = Flag;
	
	enum FrontFace
	{
		FRONT_FACE_COUNTER_CLOCKWISE = 0,
		FRONT_FACE_CLOCKWISE = 1,
		FRONT_FACE_MAX_ENUM = 0x7FFFFFFF
	};

	enum CullModeFlag
	{
		CULL_MODE_NONE = 0,
		CULL_MODE_FRONT_BIT = 0x00000001,
		CULL_MODE_BACK_BIT = 0x00000002,
		CULL_MODE_FRONT_AND_BACK = 0x00000003,
		CULL_MODE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
	};

	enum PolygonMode
	{
		POLYGON_MODE_FILL = 0,
		POLYGON_MODE_LINE = 1,
		POLYGON_MODE_POINT = 2,
		POLYGON_MODE_FILL_RECTANGLE_NV = 1000153000,
		POLYGON_MODE_MAX_ENUM = 0x7FFFFFFF
	};

	enum PrimitiveTopology {
		PRIMITIVE_TOPOLOGY_POINT_LIST = 0,
		PRIMITIVE_TOPOLOGY_LINE_LIST = 1,
		PRIMITIVE_TOPOLOGY_LINE_STRIP = 2,
		PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3,
		PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 4,
		PRIMITIVE_TOPOLOGY_TRIANGLE_FAN = 5,
		PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY = 6,
		PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY = 7,
		PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY = 8,
		PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY = 9,
		PRIMITIVE_TOPOLOGY_PATCH_LIST = 10,
		PRIMITIVE_TOPOLOGY_MAX_ENUM = 0x7FFFFFFF
	};

	
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
		mutable int access_flag;
		mutable int stage_flag;
		void* mapped_data = nullptr;
		int ref_cnt = 0;
	};
}