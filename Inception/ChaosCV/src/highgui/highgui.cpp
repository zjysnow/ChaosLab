#include "highgui/highgui.hpp"

#include <Windows.h>
#include <vulkan/vulkan_win32.h>

namespace chaos
{
	static VulkanInstance& g_instance = VulkanInstance::GetInstance();

	//class VulkanWindowImpl : public VulkanWindow
	//{
	//public:
	//	VulkanWindowImpl(const VulkanDevice* vkdev) : vkdev(vkdev)
	//	{

	//	}

	//	~VulkanWindowImpl()
	//	{
	//		
	//		
	//	}

	//	void Show() final
	//	{
	//		//while (not glfwWindowShouldClose(window))
	//		//{
	//		//	glfwPollEvents();
	//		//}
	//		//vkDeviceWaitIdle(vkdev->GetDevice());
	//	}

	//	void CreateSurface()
	//	{
	//		//CHECK_EQ(VK_SUCCESS, glfwCreateWindowSurface(g_instance, window, nullptr, &surface)) << "failed to create window";
	//	}

	//	void Create(const std::wstring& name)
	//	{
	//		
	//	}

	//	void Cleanup()
	//	{
	//		vkDestroySurfaceKHR(g_instance, surface, nullptr);
	//	}

	//	const VulkanDevice* vkdev = nullptr;
	//	VkSurfaceKHR surface = nullptr;
	//};

	//Ptr<VulkanWindow> VulkanWindow::Create(const std::wstring& name, int device_id, uint32 width, uint32 height)
	//{
	//	return Ptr<VulkanWindow>(new VulkanWindowImpl(GetGPUDevice(device_id)));
	//}
}