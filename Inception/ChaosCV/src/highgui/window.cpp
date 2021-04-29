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

			present_queue_family_index = vkdev->FindPresentQueueFamilyIndex(surface);

			vkdev->GetSurfaceCapabilities(surface, surface_capabilities);

			image_count = surface_capabilities.minImageCount + 1;
			if (surface_capabilities.maxImageCount > 0 && image_count > surface_capabilities.maxImageCount)
			{
				image_count = surface_capabilities.maxImageCount;
			}

			extent = surface_capabilities.currentExtent;
			if (extent.width == UINT32_MAX)
			{
				extent.width = (std::max)(surface_capabilities.minImageExtent.width, (std::min)(surface_capabilities.maxImageExtent.width, width));
				extent.height = (std::max)(surface_capabilities.minImageExtent.height, (std::min)(surface_capabilities.maxImageExtent.height, height));
			}

			vkdev->GetSurfaceFormat(surface, present_format);

			present_mode = vkdev->GetSurfacePresentMode(surface);

			CreateSwapChainImages();
		}

		~VulkanWindowImpl()
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
		void CreateSwapChainImages()
		{
			VkResult ret;

			VkSwapchainCreateInfoKHR swap_chain_create_info{};
			swap_chain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			swap_chain_create_info.surface = surface;

			swap_chain_create_info.minImageCount = image_count;
			swap_chain_create_info.imageFormat = present_format.format;
			swap_chain_create_info.imageColorSpace = present_format.colorSpace;
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


			swap_chain_create_info.preTransform = surface_capabilities.currentTransform;
			swap_chain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			swap_chain_create_info.presentMode = present_mode;
			swap_chain_create_info.clipped = VK_TRUE;

			swap_chain_create_info.oldSwapchain = VK_NULL_HANDLE;

			//vkdev->CreateSwapChain(&swap_chain_create_info, &swap_chain);
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
				create_info.format = present_format.format;
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

		void* CreateFrameBuffer(const VkRenderPass& render_pass)
		{
			frame_buffers.resize(image_count);

			for (size_t i = 0; i < image_count; i++)
			{
				VkImageView attachments[] = { image_views[i] };

				VkFramebufferCreateInfo framebuffer_info{};
				framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebuffer_info.renderPass = render_pass; //painter->pipeline->render_pass;
				framebuffer_info.attachmentCount = 1;
				framebuffer_info.pAttachments = attachments;
				framebuffer_info.width = extent.width;
				framebuffer_info.height = extent.height;
				framebuffer_info.layers = 1;

				VkResult ret = vkCreateFramebuffer(vkdev->GetDevice(), &framebuffer_info, nullptr, &frame_buffers[i]);
				CHECK_EQ(VK_SUCCESS, ret) << "vkCreateFramebuffer failed " << ret;
			}

			return frame_buffers.data();
		}

		uint32 height() const noexcept { return extent.height; }
		uint32 width() const noexcept { return extent.width; }
		int image_format() const noexcept { return present_format.format; }

		HWND window_handle;
		const VulkanDevice* vkdev;

		bool should_close = false;

		VkSurfaceKHR surface;
		uint32 present_queue_family_index;
		
		VkSurfaceCapabilitiesKHR surface_capabilities;
		VkExtent2D extent;

		VkPresentModeKHR present_mode;
		VkSurfaceFormatKHR present_format;

		std::vector<VkImageView> image_views;
		std::vector<VkFramebuffer> frame_buffers;

		VkSwapchainKHR swap_chain;
		std::vector<VkImage> images;
	};


	Ptr<VulkanWindow> VulkanWindow::Create(const std::wstring& name, uint32 width, uint32 height, const VulkanDevice* vkdev)
	{
		return Ptr<VulkanWindow>(new VulkanWindowImpl(name, width, height, vkdev));
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