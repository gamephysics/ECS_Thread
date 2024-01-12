#include "ECS.h"

namespace thread
{
	//============================================================================================================================================================
	// 
	//============================================================================================================================================================
#define MS_VC_EXCEPTION 0x406D1388
#pragma pack(push,8)
	struct THREADNAME_INFO
	{
		DWORD	dwType;		// Must be 0x1000.
		LPCSTR	szName;		// Pointer to name (in user addr space).
		DWORD	dwThreadID; // NThread ID (-1=caller thread).
		DWORD	dwFlags;	// Reserved for future use, must be zero.
	};
#pragma pack(pop)

	//--------------------------------------------------------------------
	// Code : ThreadName
	// Desc : Thread 이름 설정 (VS Debug 용)
	//--------------------------------------------------------------------
	void ThreadName(std::thread::native_handle_type thread_handle, const char* threadName)
	{
		THREADNAME_INFO info;
		info.dwType = 0x1000;
		info.szName = threadName;
		info.dwThreadID = ::GetThreadId(thread_handle);
		info.dwFlags = 0;

		__try
		{
			RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
		}
		__except (EXCEPTION_EXECUTE_HANDLER)
		{
		}
	}

	//--------------------------------------------------------------------
	// Code : SetProcessorCPU
	// Desc : CPU Affinity
	//--------------------------------------------------------------------
	void SetProcessorCPU(std::thread::native_handle_type thread_handle, int32_t cpuNumber)
	{
		// core number starts from 0
		auto result = SetThreadAffinityMask(thread_handle, (static_cast<DWORD_PTR>(1) << cpuNumber));
		assert(result != 0);
	};
}