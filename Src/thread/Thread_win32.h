#pragma once


namespace thread
{
	// Thread 이름 설정 (VS Debug 용)
	void	ThreadName(std::thread::native_handle_type  thread_handle, const char* threadName);

	// CPU Affinity
	void    SetProcessorCPU(std::thread::native_handle_type thread_handle, int32_t cpuNumber);
}
