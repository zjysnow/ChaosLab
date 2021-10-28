#include "core/array.hpp"

namespace chaos
{
	Shape::Shape() : Array<int>(0) {}
	Shape::Shape(int d0) : Array<int>(1) { data_[0] = d0; }
	Shape::Shape(int d0, int d1) : Array<int>(2) { data_[0] = d0; data_[1] = d1; }
	Shape::Shape(int d0, int d1, int d2) : Array<int>(3) { data_[0] = d0; data_[1] = d1; data_[2] = d2; }
}