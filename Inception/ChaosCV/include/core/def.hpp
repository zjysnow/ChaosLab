#pragma once

#define APPLICATION_NAME "ChaosCV"
#define ENGINE_NAME "ChaosEngine"

#define MAJOR 1
#define MINOR 0
#define PATCH 0
#define ENGINE_VERSION 20210321

#ifdef _WIN32
#ifdef CHAOS_EXPORT
#define CHAOS_API __declspec(dllexport)
#else
#define CHAOS_API __declspec(dllimport)
#endif
#else
#define CHAOS_API
#endif

namespace chaos
{
	constexpr static float CV_PI = 3.1415926535897932384626433832795f;
}