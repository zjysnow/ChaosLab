#include "dnn/layers/permute.hpp"

namespace chaos
{
    //template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
    //void PermuteImpl(size_t count, const Type* src, const uint32* permute_order, const Shape& src_shapes, const Steps& src_steps,
    //    const Shape& dst_shapes, const Steps& dst_steps, size_t num_axes, Type* dst)
    //{
    //    for (size_t i = 0; i < count; i++)
    //    {
    //        size_t src_idx = 0;
    //        size_t dst_idx = 0;
    //        size_t idx = i;
    //        for (int64 j = num_axes - 1; j >= 0; j--)
    //        {
    //            size_t order = permute_order[j];
    //            size_t k = idx % dst_shapes[j];
    //            dst_idx += k * dst_steps[j];
    //            src_idx += k * src_steps[order];
    //            idx /= dst_shapes[j];
    //        }
    //        dst[dst_idx] = src[src_idx];
    //    }
    //}

	Permute::Permute() : Layer("Permute") 
    {
    }

    void Permute::Set(const std::string& name, const std::any& val)
    {
        if (name == "orders")
        {
            orders = std::any_cast<std::vector<uint32>>(val);
        }
    }

	void Permute::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
        size_t num_axes = bottom_blob.shape.dims;
        CHECK_EQ(orders.size(), num_axes) << Format("num axes expect %d, but got %d", orders.size(), num_axes);

        bool need_permute = false;
        for (size_t i = 0; i < num_axes; i++)
        {
            if (i != orders[i])
            {
                need_permute = true;
                break;
            }
        }
        Shape shape = bottom_blob.shape;
        if (not need_permute)
        {
            //top_blob = bottom_blob.Clone(opt.blob_allocator);
            if (top_blob.empty()) top_blob.Create(shape, shape.steps(), bottom_blob.dtype, bottom_blob.packing, opt.blob_allocator);

            bottom_blob.CopyTo(top_blob);
        }
        else
        {
            for (size_t i = 0; i < num_axes; i++) shape[i] = bottom_blob.shape[orders[i]];
            if (top_blob.empty()) top_blob.Create(shape, shape.steps(), bottom_blob.dtype, bottom_blob.packing, opt.blob_allocator);

            CHECK_EQ(shape, top_blob.shape) << "expect " << shape << " but got " << top_blob.shape;

            size_t dims = shape.dims;

            const float* src = (const float*)bottom_blob.data;
            float* dst = (float*)top_blob.data;

            for (size_t i = 0; i < shape.total(); i++)
            {
                size_t src_idx = 0;
                size_t dst_idx = 0;
                size_t idx = i;
                for (size_t d = 0; d < dims; d++)
                {
                    uint32 order = orders[dims - d - 1];
                    size_t k = idx % shape[dims - d - 1];
                    dst_idx += k * top_blob.steps[dims - d - 1];
                    src_idx += k * bottom_blob.steps[order];
                    idx /= shape[dims - d - 1];
                }
                dst[dst_idx] = src[src_idx];
            }
        }
	}
}