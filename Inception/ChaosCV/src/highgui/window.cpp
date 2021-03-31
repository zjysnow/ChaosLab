#include "core/command.hpp"
#include "core/pipeline.hpp"
#include "highgui/window.hpp"
#include "highgui/painter.hpp"


#include <Windows.h>
#include <vulkan/vulkan_win32.h>

#define WIN_CLASS_NAME L"ChaosWindow"

namespace chaos
{
	VulkanInstance& g_instance = VulkanInstance::GetInstance();

	LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	class VulkanWindowImpl : public VulkanWindow
	{
	public:
		VulkanWindowImpl(const std::wstring& name, uint32 width, uint32 height, const VulkanDevice* vkdev) : vkdev(vkdev)
		{
			auto h_instance = GetModuleHandle(NULL);
			WNDCLASS window_class{};
			window_class.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
			window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
			window_class.hInstance = h_instance;
			window_class.lpfnWndProc = WndProc;
			window_class.lpszClassName = WIN_CLASS_NAME; // name.data();
			window_class.style = CS_HREDRAW | CS_VREDRAW;

			CHECK(RegisterClass(&window_class)) << "regist window failed.";

			window_handle = CreateWindowEx(WS_EX_APPWINDOW,
				WIN_CLASS_NAME, // window class //name.data(),
				name.data(), /// window text
				WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_TILED | WS_CAPTION | WS_SYSMENU, // window style
				CW_USEDEFAULT, CW_USEDEFAULT, // position
				width, height, // size
				NULL, // No parent window
				NULL, // No window menu
				h_instance,
				NULL);

			VkWin32SurfaceCreateInfoKHR surface_create_info{};
			surface_create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
			surface_create_info.hinstance = (HINSTANCE)h_instance;
			surface_create_info.hwnd = (HWND)window_handle;
			VkResult ret = vkCreateWin32SurfaceKHR(g_instance, &surface_create_info, 0, &surface);
			CHECK_EQ(VK_SUCCESS, ret) << "vkCreateWin32SurfaceKHR failed " << ret;

			// to check the gpu if support present
			VkBool32 support_present = false;
			// try graphics queue family first
			vkGetPhysicalDeviceSurfaceSupportKHR(vkdev->info.physical_device, vkdev->info.graphics_queue_family_index, surface, &support_present);
			present_queue_family_index = vkdev->info.graphics_queue_family_index;
			if (not support_present)
			{
				// transfer dose not support present, so just check compute queue family
				vkGetPhysicalDeviceSurfaceSupportKHR(vkdev->info.physical_device, vkdev->info.compute_queue_family_index, surface, &support_present);
				CHECK(support_present) << "physical device do not support present";
				present_queue_family_index = vkdev->info.compute_queue_family_index;
			}

			
			VkSurfaceCapabilitiesKHR present_capabilities;
			ret = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkdev->info.physical_device, surface, &present_capabilities); // should use vkdev->
			CHECK_EQ(VK_SUCCESS, ret) << "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed " << ret;

			current_transform = present_capabilities.currentTransform;
			image_count = present_capabilities.minImageCount + 1;
			if (present_capabilities.maxImageCount > 0 && image_count > present_capabilities.maxImageCount)
			{
				image_count = present_capabilities.maxImageCount;
			}

			extent = present_capabilities.currentExtent;
			if (extent.width == UINT32_MAX)
			{
				extent.width = (std::max)(present_capabilities.minImageExtent.width, (std::min)(present_capabilities.maxImageExtent.width, width));
				extent.height = (std::max)(present_capabilities.minImageExtent.height, (std::min)(present_capabilities.maxImageExtent.height, height));
			}

			uint32_t format_count;
			ret = vkGetPhysicalDeviceSurfaceFormatsKHR(vkdev->info.physical_device, surface, &format_count, nullptr);
			CHECK_EQ(VK_SUCCESS, ret) << "vkGetPhysicalDeviceSurfaceFormatsKHR failed " << ret;
			std::vector<VkSurfaceFormatKHR> present_formats(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(vkdev->info.physical_device, surface, &format_count, present_formats.data());
			CHECK_EQ(VK_SUCCESS, ret) << "vkGetPhysicalDeviceSurfaceFormatsKHR failed " << ret;

			surface_format = present_formats[0];
			for (const auto& format : present_formats)
			{
				if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				{
					surface_format = format;
					break;
				}
			}

			uint32_t present_mode_count;
			ret = vkGetPhysicalDeviceSurfacePresentModesKHR(vkdev->info.physical_device, surface, &present_mode_count, nullptr);
			CHECK_EQ(VK_SUCCESS, ret) << "vkGetPhysicalDeviceSurfacePresentModesKHR failed " << ret;
			std::vector<VkPresentModeKHR> present_modes(present_mode_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(vkdev->info.physical_device, surface, &present_mode_count, present_modes.data());
			CHECK_EQ(VK_SUCCESS, ret) << "vkGetPhysicalDeviceSurfacePresentModesKHR failed " << ret;

			present_mode = VK_PRESENT_MODE_FIFO_KHR;
			for (const auto& mode : present_modes)
			{
				if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
				{
					present_mode = mode;
					break;
				}
			}

			CreateSwapChain(); // and vkimage and image view
		}

		~VulkanWindowImpl()
		{
			CleanupSwapChain();

			vkDestroySurfaceKHR(g_instance, surface, nullptr);
		}

		void GetFrameBufferSize(uint32& width, uint32& height)
		{
			RECT area;
			GetClientRect(window_handle, &area);
			width = area.right;
			height = area.bottom;
		}

		void Show() final
		{
			ShowWindow(window_handle, SW_SHOWNA);

			while (not should_close)
			{
				PollEvents();
				painter->command->Present(swap_chain, present_queue_family_index);
			}
			vkDeviceWaitIdle(vkdev->GetDevice());
		}

		void PollEvents()
		{
			MSG msg;
			while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{
				if (msg.message == WM_QUIT)
				{
					should_close = true;
				}
				else
				{
					TranslateMessage(&msg);
					DispatchMessage(&msg);
				}
			}
		}
		void CreateSwapChain()
		{
			VkResult ret;

			VkSwapchainCreateInfoKHR swap_chain_create_info{};
			swap_chain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			swap_chain_create_info.surface = surface;

			swap_chain_create_info.minImageCount = image_count;
			swap_chain_create_info.imageFormat = surface_format.format;
			swap_chain_create_info.imageColorSpace = surface_format.colorSpace;
			swap_chain_create_info.imageExtent = extent;
			swap_chain_create_info.imageArrayLayers = 1;
			swap_chain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

			std::vector<uint32_t> queue_families = { vkdev->info.graphics_queue_family_index };
			if (present_queue_family_index != vkdev->info.graphics_queue_family_index)
			{
				queue_families.push_back(present_queue_family_index);
				swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				swap_chain_create_info.queueFamilyIndexCount = 2;
			}
			else
			{
				swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
				swap_chain_create_info.queueFamilyIndexCount = 1;
			}
			swap_chain_create_info.pQueueFamilyIndices = queue_families.data();


			swap_chain_create_info.preTransform = current_transform; //present_capabilities.currentTransform;
			swap_chain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			swap_chain_create_info.presentMode = present_mode;
			swap_chain_create_info.clipped = VK_TRUE;

			swap_chain_create_info.oldSwapchain = VK_NULL_HANDLE;

			ret = vkCreateSwapchainKHR(vkdev->GetDevice(), &swap_chain_create_info, nullptr, &swap_chain);
			CHECK_EQ(VK_SUCCESS, ret) << "vkCreateSwapchainKHR failed " << ret;

			ret = vkGetSwapchainImagesKHR(vkdev->GetDevice(), swap_chain, &image_count, nullptr);
			CHECK_EQ(VK_SUCCESS, ret) << "vkGetSwapchainImagesKHR failed " << ret;


			// create images and image views

			images.resize(image_count);
			ret = vkGetSwapchainImagesKHR(vkdev->GetDevice(), swap_chain, &image_count, images.data());
			CHECK_EQ(VK_SUCCESS, ret) << "vkGetSwapchainImagesKHR failed " << ret;

			image_views.resize(image_count);
			for (size_t i = 0; i < image_count; i++)
			{
				VkImageViewCreateInfo create_info{};
				create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				create_info.image = images[i];
				create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
				create_info.format = surface_format.format;
				create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
				create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
				create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
				create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
				create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				create_info.subresourceRange.baseMipLevel = 0;
				create_info.subresourceRange.levelCount = 1;
				create_info.subresourceRange.baseArrayLayer = 0;
				create_info.subresourceRange.layerCount = 1;

				VkResult ret = vkCreateImageView(vkdev->GetDevice(), &create_info, nullptr, &image_views[i]);
				CHECK_EQ(VK_SUCCESS, ret) << "vkCreateImageView failed " << ret;
			}
		}

		void CleanupSwapChain()
		{
			for (auto buffer : frame_buffers)
			{
				vkDestroyFramebuffer(vkdev->GetDevice(), buffer, nullptr);
			}

			for (auto& view : image_views)
			{
				vkDestroyImageView(vkdev->GetDevice(), view, nullptr);
			}
			vkDestroySwapchainKHR(vkdev->GetDevice(), swap_chain, nullptr);
		}

		void CreatePipeline() final
		{
			painter->width = extent.width;
			painter->height = extent.height;

			painter->CreatePipeline(surface_format.format);

			CreateFrameBuffer();
		}


		void CreateFrameBuffer()
		{
			frame_buffers.resize(image_count);

			for (size_t i = 0; i < image_count; i++)
			{
				VkImageView attachments[] = { image_views[i] };

				VkFramebufferCreateInfo framebuffer_info{};
				framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebuffer_info.renderPass = painter->pipeline->render_pass;
				framebuffer_info.attachmentCount = 1;
				framebuffer_info.pAttachments = attachments;
				framebuffer_info.width = extent.width;
				framebuffer_info.height = extent.height;
				framebuffer_info.layers = 1;

				VkResult ret = vkCreateFramebuffer(vkdev->GetDevice(), &framebuffer_info, nullptr, &frame_buffers[i]);
				CHECK_EQ(VK_SUCCESS, ret) << "vkCreateFramebuffer failed " << ret;
			}
			
			painter->buffers_count = image_count;
			painter->frame_buffers = frame_buffers.data();
		}


		HWND window_handle;
		const VulkanDevice* vkdev;

		bool should_close = false;

		VkSurfaceKHR surface;
		uint32 present_queue_family_index;
		uint32 image_count;
		VkSurfaceFormatKHR surface_format;
		VkPresentModeKHR present_mode;
		VkSurfaceTransformFlagBitsKHR current_transform;
		VkExtent2D extent;

		VkSwapchainKHR swap_chain;
		std::vector<VkImage> images;
		std::vector<VkImageView> image_views;
		std::vector<VkFramebuffer> frame_buffers;
	};


	Ptr<VulkanWindow> VulkanWindow::Create(const std::wstring& name, uint32 width, uint32 height, int device_id)
	{
		return Ptr<VulkanWindow>(new VulkanWindowImpl(name, width, height, GetGPUDevice(device_id)));
	}


	LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		switch (uMsg)
		{
		case WM_CREATE:
			break;
		case WM_PAINT:
		{
			PAINTSTRUCT ps;
			HDC hdc = BeginPaint(hWnd, &ps);
			EndPaint(hWnd, &ps);
			break;
		}
		case WM_CLOSE:
			DestroyWindow(hWnd);
			break;
		case WM_DESTROY:
			PostQuitMessage(0);
			break;
		default:
			break;
		}
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
}