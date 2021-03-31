#include "core/core.hpp"
#include "core/gpu.hpp"
#include "core/buffer.hpp"

namespace chaos
{
	VulkanAllocator::VulkanAllocator(const VulkanDevice* vkdev) : vkdev(vkdev) {}

	VulkanBuffer* VulkanAllocator::FastMalloc(size_t capacity)
	{
		VulkanBuffer* data = new VulkanBuffer();
		data->buffer = (VkBuffer)CreateBuffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		VkMemoryRequirements memory_requirements;
		vkGetBufferMemoryRequirements(vkdev->GetDevice(), data->buffer, &memory_requirements);

		if (memory_type_index == (uint32)-1)
		{
			if (vkdev->info.type == GPUInfo::Type::INTEGRATED)
			{
				// integrated gpu, prefer unified memory
				memory_type_index = vkdev->FindMemoryTypeIndex(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
			}
			else
			{
				// discrete gpu, device local
				memory_type_index = vkdev->FindMemoryTypeIndex(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
			}
			mappable = vkdev->IsMemoryMappable(memory_type_index);
			coherent = vkdev->IsMemoryCoherent(memory_type_index);
		}

		data->memory = (VkDeviceMemory)AllocateMemory(memory_requirements.size, memory_type_index);

		// ignore memoryRequirements.alignment as we always bind at zero offset
		vkBindBufferMemory(vkdev->GetDevice(), data->buffer, data->memory, 0);

		data->capacity = capacity;
		data->offset = 0;

		if (mappable)
		{
			VkResult ret = vkMapMemory(vkdev->GetDevice(), data->memory, 0, capacity, 0, &data->mapped_data);
			CHECK_EQ(VK_SUCCESS, ret) << "vkMapMemory failed " << ret;
		}
		return data;
	}
	void VulkanAllocator::FastFree(VulkanBuffer* data)
	{
		if (mappable) vkUnmapMemory(vkdev->GetDevice(), data->memory);
		vkDestroyBuffer(vkdev->GetDevice(), data->buffer, nullptr);
		vkFreeMemory(vkdev->GetDevice(), data->memory, nullptr);
	}

	void* VulkanAllocator::CreateBuffer(size_t size, uint32 usage)
	{
		VkBufferCreateInfo buffer_create_info{};
		buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_create_info.size = size;
		buffer_create_info.usage = usage;
		buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkBuffer buffer = nullptr;
		VkResult ret = vkCreateBuffer(vkdev->GetDevice(), &buffer_create_info, 0, &buffer);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateBuffer failed " << ret;

		return buffer;
	}

	void* VulkanAllocator::AllocateMemory(size_t size, uint32 memory_type_index)
	{
		VkMemoryAllocateInfo memory_allocate_info{};
		memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memory_allocate_info.allocationSize = size;
		memory_allocate_info.memoryTypeIndex = memory_type_index;

		VkDeviceMemory memory = nullptr;
		VkResult ret = vkAllocateMemory(vkdev->GetDevice(), &memory_allocate_info, 0, &memory);
		CHECK_EQ(VK_SUCCESS, ret) << "vkAllocateMemory failed " << ret;

		return memory;
	}



	VulkanStagingAllocator::VulkanStagingAllocator(const VulkanDevice* vkdev) : VulkanAllocator(vkdev)
	{
		mappable = true;
		coherent = true;
	}

	VulkanBuffer* VulkanStagingAllocator::FastMalloc(size_t capacity)
	{
		VulkanBuffer* data = new VulkanBuffer();
		data->buffer = (VkBuffer)CreateBuffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		VkMemoryRequirements memory_requirements;
		vkGetBufferMemoryRequirements(vkdev->GetDevice(), data->buffer, &memory_requirements);

		if (memory_type_index == (uint32)-1)
		{
			memory_type_index = vkdev->FindMemoryTypeIndex(memory_requirements.memoryTypeBits,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}

		data->memory = (VkDeviceMemory)AllocateMemory(memory_requirements.size, memory_type_index);

		// ignore memoryRequirements.alignment as we always bind at zero offset
		vkBindBufferMemory(vkdev->GetDevice(), data->buffer, data->memory, 0);

		data->capacity = capacity;
		data->offset = 0;

		VkResult ret = vkMapMemory(vkdev->GetDevice(), data->memory, 0, capacity, 0, &data->mapped_data);
		CHECK_EQ(VK_SUCCESS, ret) << "vkMapMemory failed " << ret;

		return data;
	}

}
