#pragma once

#include "core/core.hpp"
#include "core/gpu.hpp"
#include "core/command.hpp"

#include "dnn/layer.hpp"


#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

static constexpr float eps = 1E-5;
// can create data by Matlab or randn

namespace chaos
{

	class GPU
	{
	public:
		GPU() : vkdev(GetGPUDevice()), cmd(vkdev) 
		{
			opt.use_vulkan_compute = true;
			opt.blob_vkallocator = new VulkanLocalAllocator(vkdev);
			opt.staging_vkallocator = new VulkanStagingAllocator(vkdev);
		}
		~GPU() 
		{ 
			cmd.Release();
			delete opt.staging_vkallocator;
			delete opt.blob_vkallocator;
			DestroyGPUInstance(); 
		}

		const VulkanDevice* vkdev;
		Option opt;
		ComputeCommand cmd;
	};

	static GPU g_gpu;
}