#include "dnn/flow.hpp"

namespace chaos
{
	Flow::Node* begin(Flow& flow) { return &flow.nodes[0]; }
	Flow::Node* end(Flow& flow) { return &flow.nodes[0] + flow.nodes.size(); }

	Flow::Node::Node(const std::string& name) : name(name) {}
	bool Flow::Node::Isolate() const
	{
		if (producer == -1 && consumers.empty()) return true;
		return false;
	}

	Flow::Node& Flow::AddNode(const std::string& name)
	{
		int idx = 0;
		for(auto& node : nodes)
		{
			if (name == node.name) return node;
			idx++;
		}
		nodes.push_back(name);
		return nodes[idx];
	}

	Flow& Flow::operator=(int idx)
	{
		for (auto& node : nodes)
		{
			if (node.Isolate()) node.producer = idx;
		}
		return *this;
	}

	const Flow::Node& Flow::operator[](int idx) const
	{
		return nodes[idx];
	}

	int Flow::Find(const std::string& name) const
	{
		for (int idx = 0; const auto& node : nodes)
		{
			if (name == node.name) return idx;
			idx++;
		}
		LOG(FATAL) << "can not find blob " << name;
		return -1;
	}
}