#include "core/io.hpp"
#include "ml/pca.hpp"
#include "utils/op.hpp"

namespace chaos
{
	PCA::PCA() {}

	// when cols(A)>rows(A)
	// B = A'A; B*x=b*x; C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> c = b, x=A'*y

	PCA& PCA::operator()(const Tensor& A, int flags)
	{
		CHECK_EQ(2, A.shape.size());

		Array<int> vecdim;
		int m, n;
		if (flags & ROW_MAJOR)
		{
			n = A.shape[1];
			m = A.shape[0];
			vecdim = { 1 };
		}
		else // COL_MAJOR
		{
			n = A.shape[0]; // dims of feature
			m = A.shape[1]; // num of feature
			vecdim = { 0 };
		}

		Tensor mu = mean(A, false, vecdim);
		Tensor x = A - mu;
		// col major
		Tensor R = transpose(x) * x; // mxm

		auto [W, U, Vt] = svd(R);

		std::cout << W << std::endl;
		std::cout << U << std::endl << Vt;

		Tensor lambda = Tensor(Shape(m,m), Depth::D4, Packing::CHW);
		memset(lambda.data, 0, m * m * sizeof(float));
		for (int i = 0; i < W.shape[0]; i++)
		{
			lambda[i+i*m] = 1.f / sqrt(W[i]);
		}
		
		P = x * U * lambda;

		return *this;
	}
}