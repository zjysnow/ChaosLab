#pragma once
#include "core/core.hpp"

namespace chaos
{
	class Net;
	class CHAOS_API Blob
	{
	public:
		Blob() = default;
		Blob(const std::string& name) : name(name) {}

		Blob& operator=(const Blob& blob);
		

		std::string name;
		// layer index which produce this blob as output
		int producer = -1;
		// layer index which need this blob as input
		std::vector<int> consumers;

		Net* net = nullptr;
	};
}