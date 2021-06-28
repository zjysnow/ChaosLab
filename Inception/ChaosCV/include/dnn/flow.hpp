#pragma once
#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API Flow
	{
	public:
		struct Node
		{
			Node() = default;
			Node(const std::string& name);

			bool Isolate() const;

			std::string name;
			// layer index which produce this blob as output
			int producer = -1;
			// layer index which need this blob as input
			std::vector<int> consumers;
		};

		Flow() = default;

		Node& AddNode(const std::string& name);

		Flow& operator=(int idx);

		const Node& operator[](int idx) const;

		int Find(const std::string& name) const;

		size_t size() const { return nodes.size(); }
		friend Flow::Node* begin(Flow& flow);
		friend Flow::Node* end(Flow& flow);
	private:
		std::vector<Node> nodes;
	};

	
	//template <class Type>
	//Type* begin(Array<Type>& arr)
	//{
	//	return &arr[0];
	//}

	//template <class Type>
	//const Type* begin(const Array<Type>& arr)
	//{
	//	return &arr[0];
	//}

}