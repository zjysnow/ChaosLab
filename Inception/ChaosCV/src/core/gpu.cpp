#include "core/gpu.hpp"

#include <vulkan/vulkan.hpp>

#include <mutex>

namespace chaos
{
	static std::mutex g_instance_lock;

	VulkanInstance VulkanInstance::holder;
	VulkanInstance::VulkanInstance()
	{
		instance = nullptr;
	}
	VulkanInstance::~VulkanInstance() {}
	VulkanInstance& VulkanInstance::GetInstance()
	{
		return holder;
	}
	VulkanInstance::operator VkInstance()
	{
		return instance;
	}
	static VulkanInstance& g_instance = VulkanInstance::GetInstance();

	static constexpr int MAX_GPU_COUNT = 8; // 8 is enough for most systems
	static uint32 g_gpu_count = 0;
	static int g_default_gpu_index = -1;
	static GPUInfo g_gpu_infos[MAX_GPU_COUNT];
	static VulkanDevice* g_devices[MAX_GPU_COUNT] = { nullptr };

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
			LOG(ERROR) << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* create_info, const VkAllocationCallbacks* allocator, VkDebugUtilsMessengerEXT* debug_messenger)
	{
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
		if (func != nullptr)
		{
			return func(instance, create_info, allocator, debug_messenger);
		}
		else
		{
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debug_messenger, const VkAllocationCallbacks* allocator)
	{
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
		if (func != nullptr)
		{
			func(instance, debug_messenger, allocator);
		}
	}

	static uint32_t FindDeviceComputeQueue(const std::vector<VkQueueFamilyProperties>& queue_family_properties)
	{
		// first try, compute only queue
		for (uint32_t i = 0; i < queue_family_properties.size(); i++)
		{
			const VkQueueFamilyProperties& queue_family_property = queue_family_properties[i];

			if ((queue_family_property.queueFlags & VK_QUEUE_COMPUTE_BIT)
				and not(queue_family_property.queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				return i;
			}
		}
		// second try, any queue with compute and graphics
		for (uint32_t i = 0; i < queue_family_properties.size(); i++)
		{
			const VkQueueFamilyProperties& queue_family_property = queue_family_properties[i];

			if ((queue_family_property.queueFlags & VK_QUEUE_COMPUTE_BIT)
				and (queue_family_property.queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				return i;
			}
		}
		// third try, any queue with compute
		for (uint32_t i = 0; i < queue_family_properties.size(); i++)
		{
			const VkQueueFamilyProperties& queue_family_property = queue_family_properties[i];

			if (queue_family_property.queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				return i;
			}
		}
		return -1;
	}

	static uint32 FindDeviceGraphicsQueue(const std::vector<VkQueueFamilyProperties>& queue_family_properties)
	{
		// first try, graphics only queue
		for (uint32 i = 0; i < queue_family_properties.size(); i++)
		{
			const VkQueueFamilyProperties& queue_family_property = queue_family_properties[i];

			if ((queue_family_property.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				&& !(queue_family_property.queueFlags & VK_QUEUE_COMPUTE_BIT))
			{
				return i;
			}
		}
		// second try, any queue with graphics and compute
		for (uint32 i = 0; i < queue_family_properties.size(); i++)
		{
			const VkQueueFamilyProperties& queue_family_property = queue_family_properties[i];

			if ((queue_family_property.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				&& (queue_family_property.queueFlags & VK_QUEUE_COMPUTE_BIT))
			{
				return i;
			}
		}
		// third try, any queue with graphics
		for (uint32 i = 0; i < queue_family_properties.size(); i++)
		{
			const VkQueueFamilyProperties& queue_family_property = queue_family_properties[i];

			if (queue_family_property.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				return i;
			}
		}
		return -1;
	}

	static uint32 FindDeviceTransferQueue(const std::vector<VkQueueFamilyProperties>& queue_family_properties)
	{
		// first try, transfer only queue
		for (uint32 i = 0; i < queue_family_properties.size(); i++)
		{
			const VkQueueFamilyProperties& queue_family_property = queue_family_properties[i];

			if ((queue_family_property.queueFlags & VK_QUEUE_TRANSFER_BIT)
				&& !(queue_family_property.queueFlags & VK_QUEUE_COMPUTE_BIT)
				&& !(queue_family_property.queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				return i;
			}
		}
		// second try, any queue with transfer
		for (uint32 i = 0; i < queue_family_properties.size(); i++)
		{
			const VkQueueFamilyProperties& queue_family_property = queue_family_properties[i];

			if (queue_family_property.queueFlags & VK_QUEUE_TRANSFER_BIT)
			{
				return i;
			}
		}
		// third try, use compute queue
		uint32 compute_queue_index = FindDeviceComputeQueue(queue_family_properties);
		if (compute_queue_index != (uint32_t)-1)
		{
			return compute_queue_index;
		}
		// fourth try, use graphics queue
		uint32_t graphics_queue_index = FindDeviceGraphicsQueue(queue_family_properties);
		if (graphics_queue_index != (uint32_t)-1)
		{
			return graphics_queue_index;
		}
		return -1;
	}

	static int FindDefaultVulkanDeviceIndex()
	{
		// first try, discrete gpu
		for (uint32_t i = 0; i < g_gpu_count; i++)
		{
			if (g_gpu_infos[i].type == GPUInfo::DISCRETE)
				return i;
		}
		// second try, integrated gpu
		for (uint32_t i = 0; i < g_gpu_count; i++)
		{
			if (g_gpu_infos[i].type == GPUInfo::INTEGRATED)
				return i;
		}
		// third try, any probed device
		if (g_gpu_count > 0)
			return 0;

		LOG(FATAL) << "no vulkan device";
		return -1;
	}

	void CreateGPUInstance()
	{
		std::lock_guard lock(g_instance_lock);

		if ((VkInstance)g_instance != nullptr) return;

		VkResult ret;

		std::vector<const char*> enabled_layers;
		uint32 instance_layer_property_count;
		ret = vkEnumerateInstanceLayerProperties(&instance_layer_property_count, nullptr);
		CHECK_EQ(VK_SUCCESS, ret) << "vkEnumerateInstanceLayerProperties failed " << ret;
		std::vector<VkLayerProperties> instance_layer_properties(instance_layer_property_count);
		ret = vkEnumerateInstanceLayerProperties(&instance_layer_property_count, instance_layer_properties.data());
		CHECK_EQ(VK_SUCCESS, ret) << "vkEnumerateInstanceLayerProperties failed " << ret;
		for (const auto& lp : instance_layer_properties)
		{
			if (0 == std::strcmp(lp.layerName, "VK_LAYER_KHRONOS_validation"))
			{
				enabled_layers.push_back("VK_LAYER_KHRONOS_validation");
			}
		}

		std::vector<const char*> enabled_extensions;
		uint32 instance_extension_property_count;
		ret = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_property_count, nullptr);
		CHECK_EQ(VK_SUCCESS, ret) << "vkEnumerateInstanceExtensionProperties failed " << ret;
		std::vector<VkExtensionProperties> instance_extension_properties(instance_extension_property_count);
		ret = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_property_count, instance_extension_properties.data());
		CHECK_EQ(VK_SUCCESS, ret) << "vkEnumerateInstanceExtensionProperties failed " << ret;
		for (const auto& exp : instance_extension_properties)
		{
			if (0 == std::strcmp(exp.extensionName, "VK_EXT_debug_utils"))
			{
				g_instance.support_VK_EXT_debug_utils = exp.specVersion;
			}
		}
		if (g_instance.support_VK_EXT_debug_utils) enabled_extensions.push_back("VK_EXT_debug_utils");

		uint32 instance_api_version = VK_MAKE_VERSION(MAJOR, MINOR, PATCH);

		VkApplicationInfo application_info{};
		application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		application_info.pApplicationName = APPLICATION_NAME;
		application_info.applicationVersion = NULL;
		application_info.pEngineName = ENGINE_NAME;
		application_info.engineVersion = ENGINE_VERSION;
		application_info.apiVersion = instance_api_version;

		VkInstanceCreateInfo instance_create_info{};
		instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instance_create_info.pApplicationInfo = &application_info;
		instance_create_info.enabledLayerCount = (uint32)enabled_layers.size();
		instance_create_info.ppEnabledLayerNames = enabled_layers.data();
		instance_create_info.enabledExtensionCount = (uint32)enabled_extensions.size();
		instance_create_info.ppEnabledExtensionNames = enabled_extensions.data();

		VkInstance instance = 0;
		ret = vkCreateInstance(&instance_create_info, 0, &instance);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateInstance failed " << ret;
		g_instance.instance = instance;

		if (g_instance.support_VK_EXT_debug_utils)
		{
			VkDebugUtilsMessengerCreateInfoEXT debug_messenger_create_info{};
			debug_messenger_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			debug_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			debug_messenger_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			debug_messenger_create_info.pfnUserCallback = debugCallback;

			ret = CreateDebugUtilsMessengerEXT(instance, &debug_messenger_create_info, nullptr, &g_instance.debug_messenger);
			CHECK_EQ(VK_SUCCESS, ret) << "CreateDebugUtilsMessengerEXT failed " << ret;
		}

		ret = vkEnumeratePhysicalDevices((VkInstance)g_instance, &g_gpu_count, 0);
		CHECK_EQ(VK_SUCCESS, ret) << "vkEnumeratePhysicalDevices " << ret;
		if (g_gpu_count > MAX_GPU_COUNT) g_gpu_count = MAX_GPU_COUNT;
		std::vector<VkPhysicalDevice> physical_devices(g_gpu_count);
		ret = vkEnumeratePhysicalDevices(g_instance, &g_gpu_count, physical_devices.data());
		CHECK_EQ(VK_SUCCESS, ret) << "vkEnumeratePhysicalDevices " << ret;
		for (uint32_t i = 0; i < g_gpu_count; i++)
		{
			const VkPhysicalDevice& physical_device = physical_devices[i];
			GPUInfo& gpu_info = g_gpu_infos[i];

			gpu_info.physical_device = physical_device;

			VkPhysicalDeviceProperties physical_device_properties;
			vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

			switch (physical_device_properties.deviceType)
			{
			case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
				gpu_info.type = GPUInfo::DISCRETE;
				break;
			case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
				gpu_info.type = GPUInfo::INTEGRATED;
				break;
			case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
				gpu_info.type = GPUInfo::VIRTUAL;
				break;
			case VK_PHYSICAL_DEVICE_TYPE_CPU:
				gpu_info.type = GPUInfo::CPU;
				break;
			default:
				gpu_info.type = GPUInfo::OTHER;
				break;
			}

			gpu_info.device_name = physical_device_properties.deviceName;

			VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
			vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_device_memory_properties);
			for (uint32 i = 0; i < physical_device_memory_properties.memoryTypeCount; i++)
			{
				gpu_info.memory_properties.push_back(physical_device_memory_properties.memoryTypes[i].propertyFlags);
			}

			// find compute queue
			uint32_t queue_family_properties_count;
			vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_properties_count, 0);
			std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_properties_count);
			vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_properties_count, queue_family_properties.data());
			gpu_info.graphics_queue_family_index = FindDeviceGraphicsQueue(queue_family_properties);
			gpu_info.transfer_queue_family_index = FindDeviceTransferQueue(queue_family_properties);
			gpu_info.compute_queue_family_index = FindDeviceComputeQueue(queue_family_properties);
			gpu_info.graphics_queue_count = queue_family_properties[gpu_info.graphics_queue_family_index].queueCount;
			gpu_info.transfer_queue_count = queue_family_properties[gpu_info.transfer_queue_family_index].queueCount;
			gpu_info.compute_queue_count = queue_family_properties[gpu_info.compute_queue_family_index].queueCount;

			// get device extension
			uint32_t device_extension_property_count = 0;
			ret = vkEnumerateDeviceExtensionProperties(physical_device, NULL, &device_extension_property_count, NULL);
			CHECK_EQ(VK_SUCCESS, ret) << "vkEnumerateDeviceExtensionProperties failed " << ret;
			std::vector<VkExtensionProperties> device_extension_properties(device_extension_property_count);
			ret = vkEnumerateDeviceExtensionProperties(physical_device, NULL, &device_extension_property_count, device_extension_properties.data());
			CHECK_EQ(VK_SUCCESS, ret) << "vkEnumerateDeviceExtensionProperties failed " << ret;
		}

		g_default_gpu_index = FindDefaultVulkanDeviceIndex();
	}

	void DestroyGPUInstance()
	{
		std::lock_guard lock(g_instance_lock);

		if ((VkInstance)g_instance == nullptr) return;

		for (size_t i = 0; i < MAX_GPU_COUNT; i++)
		{

		}

		if (g_instance.support_VK_EXT_debug_utils)
		{
			DestroyDebugUtilsMessengerEXT(g_instance, g_instance.debug_messenger, nullptr);
			g_instance.debug_messenger = nullptr;
		}
		vkDestroyInstance(g_instance, nullptr);
		g_instance.instance = nullptr;
	}

	static bool IsGPUInstanceReady()
	{
		std::lock_guard lock(g_instance_lock);
		return (VkInstance)g_instance != 0;
	}

	static void TryCreateGPUInstance()
	{
		if (not IsGPUInstanceReady())
			CreateGPUInstance();
	}

	int GetGPUCount()
	{
		TryCreateGPUInstance();
		return g_gpu_count;
	}
	int GetDefaultGPUIndex()
	{
		TryCreateGPUInstance();
		return g_default_gpu_index;

	}
	const GPUInfo& GetGPUInfo(int device_index)
	{
		TryCreateGPUInstance();
		return g_gpu_infos[device_index];
	}


	VulkanDevice::VulkanDevice(int device_index) : info(g_gpu_infos[device_index])
	{
		std::vector<const char*> enabled_extensions;

		VkPhysicalDeviceFeatures device_features{};
		device_features.fillModeNonSolid = true;

		std::vector<float> compute_queue_priorities(info.compute_queue_count, 1.f);
		std::vector<float> graphics_queue_priorities(info.graphics_queue_count, 1.f);
		std::vector<float> transfer_queue_priorities(info.transfer_queue_count, 1.f);

		VkDeviceQueueCreateInfo compute_queue_create_info{};
		compute_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		compute_queue_create_info.queueFamilyIndex = info.compute_queue_family_index;
		compute_queue_create_info.queueCount = info.compute_queue_count;
		compute_queue_create_info.pQueuePriorities = compute_queue_priorities.data();

		VkDeviceQueueCreateInfo graphics_queue_create_info{};
		graphics_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		graphics_queue_create_info.queueFamilyIndex = info.graphics_queue_family_index;
		graphics_queue_create_info.queueCount = info.graphics_queue_count;
		graphics_queue_create_info.pQueuePriorities = graphics_queue_priorities.data();

		VkDeviceQueueCreateInfo transfer_queue_create_info{};
		transfer_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		transfer_queue_create_info.queueFamilyIndex = info.transfer_queue_family_index;
		transfer_queue_create_info.queueCount = info.transfer_queue_count;
		transfer_queue_create_info.pQueuePriorities = transfer_queue_priorities.data();

		std::vector<VkDeviceQueueCreateInfo> device_queue_create_infos;
		auto have = [=](uint32 queue_family_index) {
			return 1 == std::count_if(device_queue_create_infos.begin(), device_queue_create_infos.end(), 
				[=](const VkDeviceQueueCreateInfo& info) { return info.queueFamilyIndex == queue_family_index; });
		};
		if (not have(info.compute_queue_family_index))
		{
			device_queue_create_infos.push_back(compute_queue_create_info);
		}
		if (not have(info.graphics_queue_family_index))
		{
			device_queue_create_infos.push_back(graphics_queue_create_info);
		}
		if (not have(info.transfer_queue_family_index))
		{
			device_queue_create_infos.push_back(transfer_queue_create_info);
		}

		VkDeviceCreateInfo device_create_info{};
		device_create_info.pQueueCreateInfos = device_queue_create_infos.data();
		device_create_info.queueCreateInfoCount = (uint32)device_queue_create_infos.size();
		device_create_info.enabledLayerCount = 0;
		device_create_info.enabledExtensionCount = (uint32)enabled_extensions.size();
		device_create_info.ppEnabledExtensionNames = enabled_extensions.data();
		device_create_info.pEnabledFeatures = &device_features;

		VkResult ret = vkCreateDevice(info.physical_device, &device_create_info, 0, &device);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateDevice failed " << ret;

		// init device extension

		// create queues
		compute_queues.resize(info.compute_queue_count);
		for (uint32_t i = 0; i < info.compute_queue_count; i++)
		{
			vkGetDeviceQueue(device, info.compute_queue_family_index, i, &compute_queues[i]);
		}
		if (info.compute_queue_family_index != info.graphics_queue_family_index)
		{
			graphics_queues.resize(info.graphics_queue_count);
			for (uint32_t i = 0; i < info.graphics_queue_count; i++)
			{
				vkGetDeviceQueue(device, info.graphics_queue_family_index, i, &graphics_queues[i]);
			}
		}
		if (info.compute_queue_family_index != info.transfer_queue_family_index && info.graphics_queue_family_index != info.transfer_queue_family_index)
		{
			transfer_queues.resize(info.transfer_queue_count);
			for (uint32_t i = 0; i < info.transfer_queue_count; i++)
			{
				vkGetDeviceQueue(device, info.transfer_queue_family_index, i, &transfer_queues[i]);
			}
		}
		
	}
	VulkanDevice::~VulkanDevice()
	{
		vkDestroyDevice(device, nullptr);
	}

	uint32 VulkanDevice::FindMemoryTypeIndex(uint32 memory_type_bits, int required, int preferred, int preferred_not) const
	{
		uint32 memory_type_count = (uint32)info.memory_properties.size();
		// first try, find required and with preferred and without preferred_not
		for (uint32 i = 0; i < memory_type_count; i++)
		{
			bool is_required = (1 << i) & memory_type_bits;
			if (is_required)
			{
				//const VkMemoryType& memoryType = info.physical_device_memory_properties.memoryTypes[i];
				const auto& memory_property = info.memory_properties[i];
				if ((memory_property & required) == required
					and (preferred && (memory_property & preferred))
					and (preferred_not && not (memory_property & preferred_not)))
				{
					return i;
				}
			}
		}
		// second try, find required and with preferred
		for (uint32 i = 0; i < memory_type_count; i++)
		{
			bool is_required = (1 << i) & memory_type_bits;
			if (is_required)
			{
				//const VkMemoryType& memoryType = info.physical_device_memory_properties.memoryTypes[i];
				const auto& memory_property = info.memory_properties[i];
				if ((memory_property & required) == required
					and (preferred and (memory_property & preferred)))
				{
					return i;
				}
			}
		}
		// third try, find required and without preferred_not
		for (uint32 i = 0; i < memory_type_count; i++)
		{
			bool is_required = (1 << i) & memory_type_bits;
			if (is_required)
			{
				//const VkMemoryType& memoryType = info.physical_device_memory_properties.memoryTypes[i];
				const auto& memory_property = info.memory_properties[i];
				if ((memory_property & required) == required
					&& (preferred_not && !(memory_property & preferred_not)))
				{
					return i;
				}
			}
		}
		// fourth try, find any required
		for (uint32 i = 0; i < memory_type_count; i++)
		{
			bool is_required = (1 << i) & memory_type_bits;
			if (is_required)
			{
				//const VkMemoryType& memoryType = info.physical_device_memory_properties.memoryTypes[i];
				const auto& memory_property = info.memory_properties[i];
				if ((memory_property & required) == required)
				{
					return i;
				}
			}
		}
		LOG(FATAL) << std::format("no such memory type {} {} {} {}", memory_type_bits, required, preferred, preferred_not);
		return -1;
	}
	bool VulkanDevice::IsMemoryMappable(uint32 memory_type_index) const
	{
		return info.memory_properties[memory_type_index] & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	}
	bool VulkanDevice::IsMemoryCoherent(uint32 memory_type_index) const
	{
		return info.memory_properties[memory_type_index] & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	}
}