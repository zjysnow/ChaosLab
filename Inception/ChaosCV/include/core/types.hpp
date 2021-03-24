#pragma once

#include "core/def.hpp"

#include <numeric>

namespace chaos
{
	using uchar = unsigned char;
	using int8 = __int8;
	using uint8 = unsigned __int8;
	using int16 = __int16;
	using uint16 = unsigned __int16;
	using int32 = __int32;
	using uint32 = unsigned __int32;
	using int64 = __int64;
	using uint64 = unsigned __int64;

	enum class DeviceType
	{
		CPU,
		GPU,
	};

	enum class LogSeverity
	{
		INFO,
		WARNING,
		ERROR,
		FATAL,
	};

	enum class DataType
	{
		D1 = 1, // int8 or uint8
		D2 = 2, // float16, int16 or uint16
		D4 = 4, // float, int32 or uint32
		D8 = 8, // double, int64 or uint64
	};

	enum class Packing
	{
		CHW = 1, //scalar
		C2HW2 = 2, // complex
		C3HW3 = 3, // 3-channel image
		C4HW4 = 4, // 4-channel image, sse or neon
		C8HW8 = 8, // avx or fp16
	};

	template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
	static inline Type operator*(const Type& lhs, const DataType& rhs)
	{
		return lhs * static_cast<Type>(rhs);
	}

	template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
	static inline Type operator*(const Type& lhs, const Packing& rhs)
	{
		return lhs * static_cast<Type>(rhs);
	}

	class CHAOS_API Steps
	{
	public:
		friend class Shape;

		Steps() = default;
		~Steps();

		template<class Type, std::enable_if_t<std::is_convertible_v<Type, uint32>, bool> = true>
		Steps(const std::initializer_list<Type>& list) : Steps(list.size())
		{
			for (size_t i = 0; const auto & val : list)
			{
				data[i++] = static_cast<uint32>(val);
			}
		}
		
		Steps(const Steps& steps);
		Steps& operator=(const Steps& steps);

		uint32& operator[](size_t idx) noexcept { return data[idx]; }
		const uint32& operator[](size_t idx) const noexcept { return data[idx]; }

		CHAOS_API friend bool operator==(const Steps& lhs, const Steps& rhs);
		uint32* data = nullptr;
		size_t size = 0;

	private:
		Steps(size_t size);
	};

	class CHAOS_API Shape
	{
	public:
		Shape() = default;
		~Shape();

		Shape(uint32 d0);
		Shape(uint32 d0, uint32 d1);
		Shape(uint32 d0, uint32 d1, uint32 d2);

		template<class Type, std::enable_if_t<std::is_convertible_v<Type, uint32>, bool> = true>
		Shape(const std::initializer_list<Type>& list) : Shape(list.size()) 
		{
			for (size_t i = 0; const auto & val : list)
			{
				data[i++] = static_cast<uint32>(val);
			}
		}

		Shape(const Shape& shape);
		Shape& operator=(const Shape& shape);

		Steps steps() const;

		uint32& operator[](size_t idx) noexcept { return data[idx]; }
		const uint32& operator[](size_t idx) const noexcept { return data[idx]; }

		size_t total() const;

		CHAOS_API friend bool operator==(const Shape& lhs, const Shape& rhs);
		CHAOS_API friend std::wostream& operator<<(std::wostream& stream, const Shape& shape);

		size_t dims = 0;
		uint32* data = nullptr;

	private:
		Shape(size_t dims);
	};

}