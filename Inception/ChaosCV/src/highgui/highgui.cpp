#include "core/gpu.hpp"
#include "highgui/highgui.hpp"

#include <GLFW/glfw3.h>

namespace chaos
{
	static VulkanInstance& g_insgance = VulkanInstance::GetInstance();

	class VulkanWindowImpl : public VulkanWindow
	{
	public:
		VulkanWindowImpl(const VulkanDevice* vkdev) : vkdev(vkdev)
		{
			glfwInit();
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		}

		~VulkanWindowImpl()
		{
			glfwDestroyWindow(window);
			glfwTerminate();
		}

		void Show() final
		{
			while (not glfwWindowShouldClose(window))
			{
				glfwPollEvents();
			}
			vkDeviceWaitIdle(vkdev->GetDevice());
		}

		GLFWwindow* window = nullptr;
		const VulkanDevice* vkdev = nullptr;
		VkSurfaceKHR surface = nullptr;
	};

	Ptr<VulkanWindow> VulkanWindow::Create(const std::wstring& name, int device_id, uint32 width, uint32 height)
	{
		return Ptr<VulkanWindow>(new VulkanWindowImpl(GetGPUDevice(device_id)));
	}
}