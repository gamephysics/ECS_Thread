// ECS.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "ECS.h"

int main()
{
	entt::registry		m_registry;		// ecs library
	thread::thread_pool	m_Pool(8, 0);	// thread 8 count, cpu affinity from 0 ~ 7

	// 1. entity �� �����.
	auto id = m_registry.create();

	// 2. entity �� �ʿ��� structure ���� �߰��Ѵ�.
	m_registry.emplace<ecs::Position>(id, 0.f, 0.f);
	m_registry.emplace<ecs::Sprite>(id);

	// 3. structure ���� ��ȸ�ϸ鼭 �����Ѵ�.
	if (auto view = m_registry.view<ecs::Position, ecs::Sprite>(); view.size_hint() > 0)
	{
		auto handle = view.handle();

		m_Pool.parallel_for(handle->begin(), handle->end(), [&view](const auto start, const auto end, auto block) -> auto
		{
			for (auto e = start; e != end; ++e)
			{
				auto entityID = *e;
				if (view.contains(entityID))
				{
					auto& p = view.get<ecs::Position>(entityID);
					p.m_x = 100;
					p.m_y = 100;
				}
			}
		});
	}
}
