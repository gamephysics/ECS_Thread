#pragma once

// ============================================================================================= //
// @file BS_thread_pool.hpp
// @author Barak Shoshany (baraksh@gmail.com) (http://baraksh.com)
// @version 3.5.0
// @date 2023-05-25
// @copyright Copyright (c) 2023 Barak Shoshany. Licensed under the MIT license. If you found this project useful, please consider starring it on GitHub! If you use this library in software of any kind, please provide a link to the GitHub repository https://github.com/bshoshany/thread-pool in the source code and documentation. If you use this library in published research, please cite it as follows: Barak Shoshany, "A C++17 Thread Pool for High-Performance Scientific Computing", doi:10.5281/zenodo.4742687, arXiv:2105.00613 (May 2021)
// 
// @brief BS::thread_pool: a fast, lightweight, and easy-to-use C++17 thread pool library. This header file contains the entire library, including the main BS::thread_pool class and the helper classes BS::multi_future, BS::blocks, BS:synced_stream, and BS::timer.
// ============================================================================================= // 

//#define BS_THREAD_POOL_VERSION "v3.5.0 (2023-05-25)"
//
//#include <chrono>             // std::chrono
//#include <condition_variable> // std::condition_variable
//#include <exception>          // std::current_exception
//#include <functional>         // std::bind, std::function, std::invoke
//#include <future>             // std::future, std::promise
//#include <iostream>           // std::cout, std::endl, std::flush, std::ostream
//#include <memory>             // std::make_shared, std::make_unique, std::shared_ptr, std::unique_ptr
//#include <mutex>              // std::mutex, std::scoped_lock, std::unique_lock
//#include <queue>              // std::queue
//#include <thread>             // std::thread
//#include <type_traits>        // std::common_type_t, std::conditional_t, std::decay_t, std::invoke_result_t, std::is_void_v
//#include <utility>            // std::forward, std::move, std::swap
//#include <vector>             // std::vector

namespace thread
{
    // A convenient shorthand for the type of std::thread::hardware_concurrency(). Should evaluate to unsigned int.
    using concurrency_t = std::invoke_result_t<decltype(std::thread::hardware_concurrency)>;

    //--------------------------------------------------------------------
    // class: blocks
    // Desc	: A helper class to divide a range into blocks. Used by parallelize_loop() and push_loop().
    // T1   : range 의 first index type
    // T2   : range 의 end   index type
    //--------------------------------------------------------------------
    template <typename T1, typename T2, typename T = std::common_type_t<T1, T2>>
    class [[nodiscard]] blocks
    {
    private:
        size_t  block_size  = 0;    // The size of each block (except possibly the last block).
        T       first_index = 0;    // i 번째 Block 의 first index
        T       end_index   = 0;    // i 번째 Block 의 end   index
        size_t  num_blocks  = 0;    // 처리해야할 Block 개수
        size_t  total_size  = 0;    // range 범위의 처리 개수

    public:
        //--------------------------------------------------------------------
        // Construct a blocks object with the given specifications.
        //--------------------------------------------------------------------
        // range        :  [first ~ end) 
        // num_blocks_  : 몇개의 block 으로 나눠 연산을 수행할것인가?
        //--------------------------------------------------------------------
        blocks(const T1 first_index_, const T2 end_index_, const size_t num_blocks_) : first_index(static_cast<T>(first_index_)), end_index(static_cast<T>(end_index_)), num_blocks(num_blocks_)
        {
            if (end_index < first_index)   std::swap(end_index, first_index);

			total_size = static_cast<size_t>(end_index_ - first_index);
			block_size = static_cast<size_t>(total_size / num_blocks);

			if (block_size == 0)
			{
				block_size = 1;
				num_blocks = (total_size > 1) ? total_size : 1;
			}
        }
        [[nodiscard]] T      start(const size_t i)  const { return static_cast<T>(first_index + i * block_size) ; }                                             // i 번째 Block 의 first index
        [[nodiscard]] T      end(const size_t i)    const { return (i == num_blocks - 1) ? end_index : (static_cast<T>(first_index + (i + 1) * block_size)); }  // i 번째 Block 의 end   index
        [[nodiscard]] size_t get_num_blocks()       const { return num_blocks; }                // 처리해야할 Block 개수 : [first ~ end) 까지 num_blocks 개수로 나누었다.      
        [[nodiscard]] size_t get_total_size()       const { return total_size; }                // range 범위의 처리 개수
    };


	//--------------------------------------------------------------------
	// class: multi_future
	// Desc	: A helper class to facilitate waiting for and/or getting the results of multiple futures at once.
	//--------------------------------------------------------------------
    template <typename T>
    class [[nodiscard]] multi_future
    {
    private:
        std::vector<std::future<T>> futures;        // A vector to store the futures.

    public:
		//--------------------------------------------------------------------
	    // Construct a multi_future object with the given number of futures.
	    //--------------------------------------------------------------------
        multi_future(const size_t num_futures_ = 0) : futures(num_futures_) {}  // num_futures_ : The desired number of futures to store.

        // Get the results from all the futures stored in this multi_future object, rethrowing any stored exceptions.
        // If the futures return void, this function returns void as well. Otherwise, it returns a vector containing the results.
        [[nodiscard]] std::conditional_t<std::is_void_v<T>, void, std::vector<T>> get()
        {
            if constexpr (std::is_void_v<T>)
            {
                for (size_t i = 0; i < futures.size(); ++i)
                    futures[i].get();
                return;
            }
            else
            {
                std::vector<T> results(futures.size());
                for (size_t i = 0; i < futures.size(); ++i)
                    results[i] = futures[i].get();
                return results;
            }
        }

        [[nodiscard]] std::future<T>& operator[](const size_t i)    {   return futures[i];      }                       // i 번째 future
        void push_back(std::future<T> future)                       {   futures.push_back(std::move(future));   }       // append future
        [[nodiscard]] size_t size() const                           {   return futures.size();  }                       // The number of futures

		//--------------------------------------------------------------------
        // Wait for all the futures stored in this multi_future object.
        //--------------------------------------------------------------------
        void wait() const
        {
            for (size_t i = 0; i < futures.size(); ++i)
                futures[i].wait();
        }
    };


	//--------------------------------------------------------------------
	// class: thread_pool
	// Desc	: A fast, lightweight, and easy-to-use C++17 thread pool class.
	//--------------------------------------------------------------------
    class [[nodiscard]] thread_pool
    {
    private:
        bool paused          = false;                       // A flag indicating whether the workers should pause. When set to true, the workers temporarily stop retrieving new tasks out of the queue, although any tasks already executed will keep running until they are finished. When set to false again, the workers resume retrieving tasks.
        bool waiting         = false;                       // wait_for_tasks() is active and expects to be notified whenever a task is done.
        bool workers_running = false;                       // A flag indicating to the workers to keep running.When set to false, the workers terminate permanently.

        std::condition_variable task_available_cv = {};     // A condition variable used to notify worker() that a new task has become available.
        std::condition_variable tasks_done_cv     = {};     // A condition variable used to notify wait_for_tasks() that a tasks is done.
        std::queue<std::function<void()>> tasks   = {};     // A queue of tasks to be executed by the threads.
        size_t                  tasks_running     = 0;      // A counter for the total number of currently running tasks.
        mutable std::mutex      tasks_mutex       = {};     // A mutex to synchronize access to the task queue by different threads.
        
        concurrency_t                  thread_count= 0;     // The number of threads in the pool.
        std::unique_ptr<std::thread[]> threads = nullptr;   // A smart pointer to manage the memory allocated for the threads.

        //--------------------------------------------------------------------
        int                                             cpu_number = -1;    // cpu affinity from number for threads 
        std::unordered_map<std::thread::id, int>        threads_id;
        std::string                                     name    = "the";
        //--------------------------------------------------------------------

	public:
		typedef std::shared_ptr<thread_pool>	SharedPtr;
		typedef std::weak_ptr<thread_pool>	    WeakPtr;

    public:
        // just create thread_pool as member in class :  must call reset();
        explicit thread_pool(const std::string& prefix = "the", int cpu = -1) : name(prefix), cpu_number(cpu) {}

		//--------------------------------------------------------------------
	    // Construct a new thread pool.
	    //--------------------------------------------------------------------
	    // thread_count_ : 0 => Maximize Thread Count
        //                 N => The number of threads to use. 
        //      The default value is the total number of hardware threads available, as reported by the implementation. 
        //      This is usually determined by the number of cores in the CPU. 
        //      If a core is hyperthreaded, it will count as two threads.
	    //--------------------------------------------------------------------
        explicit thread_pool(const concurrency_t thread_count_, int cpu = -1) : thread_count(determine_thread_count(thread_count_)), threads(std::make_unique<std::thread[]>(determine_thread_count(thread_count_))), cpu_number(cpu) 
        {
            create_threads();
        }
        // 모든 task 가 끝날때까지 대기하고나서 모든 Thread 를 제거한다. pool 이 대기중이였다면 남아있는 task 는 진행되지 않는다.
        ~thread_pool()
        {
            destroy();
        }


        // pool 의 thread 개수를 재설정한다.
        // 1) 현재 진행중인 task 는 완료가 될때까지 기다린다.
        // 2) 모든 thread pool 을 소멸시키고고 thread_count_ 개수로 생성한다 
        // 3) 대기중인 task 들이 다시 새로운 thread 에서 수행을 시작합니다.
        // 4) reset 전에 pool 이 대기중이었다면, 새로 생성한 pool 에서도 대기중으로 남아있습니다.
        // # thread_count_ 는 CPU core 개수이지만 core 가 hyperthread 를 지원하면 2개로 계산한다.
        void create(const concurrency_t thread_count_ = 0, int cpu = -1)
		{
			std::unique_lock tasks_lock(tasks_mutex);
			const bool was_paused = paused;
			paused = true;
			tasks_lock.unlock();
			wait_for_tasks();
			destroy_threads();
			thread_count = determine_thread_count(thread_count_);
			threads = std::make_unique<std::thread[]>(thread_count);
			paused = was_paused;
            cpu_number = cpu;
			create_threads();
		}
        
        void destroy()
        {
			wait_for_tasks();
			destroy_threads();
        }
        
    public:
        [[nodiscard]] size_t get_tasks_queued() const               {   const std::scoped_lock tasks_lock(tasks_mutex); return tasks.size();    }   // queue  에 대기하고있는 task 숫자
        [[nodiscard]] size_t get_tasks_running() const              {   const std::scoped_lock tasks_lock(tasks_mutex); return tasks_running;   }   // thread 에서 수행하고있는 task 숫자 
		[[nodiscard]] size_t get_tasks_total() const                {   const std::scoped_lock tasks_lock(tasks_mutex);	return tasks_running + tasks.size();    }   // 완료되지 않은 task 숫자 : get_tasks_total() == get_tasks_queued() + get_tasks_running().
        [[nodiscard]] concurrency_t get_thread_count() const        {   return thread_count;    }   // 생성된 Thread 개수
        ///=================================================================================
        [[nodiscard]] bool   is_paused() const  {   const std::scoped_lock tasks_lock(tasks_mutex); return paused;                          }   // thread pool 이 멈춰있는지 
        void pause()                            {   const std::scoped_lock tasks_lock(tasks_mutex); paused = true;                          }   // 현재 진행중인 task 는 끝날때까지 진행하고, 대기한다. 
		void unpause()                          {   const std::scoped_lock tasks_lock(tasks_mutex); paused = false;        	                }   // 새로운 task 를 받아 진행한다.
        void purge()                            {   const std::scoped_lock tasks_lock(tasks_mutex); while (!tasks.empty())  tasks.pop();    }   // queue 에서 대기중인 모든 task 들을 제거합니다. 현재 실행 중인 작업은 영향을 받지 않지만 대기열에서 아직 대기 중인 작업은 모두 삭제되고 스레드에 의해 실행되지 않습니다. 제거된 작업을 복원할 수 있는 방법은 없습니다.
        ///=================================================================================
        [[nodiscard]] int get_thread_index(std::thread::id id_)const{   auto iter = threads_id.find(id_);    if (iter != threads_id.end()) return iter->second; else return -1; }
        [[nodiscard]] int get_thread_index() const                  {   return get_thread_index(std::this_thread::get_id()); }

		///=================================================================================
        // 사용1) multi_future 를 받아 wait()
        ///=================================================================================
        // num_blocks 개수로 나눠 queue 에 등록해 Parallelize 시킨다. multi_future 를 받아 wait() 한다.
        template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
        [[nodiscard]] multi_future<R> parallelize_loop(const T1 first_index, const T2 index_after_last, F&& loop, const size_t num_blocks = 0)
        {
            assert(get_thread_count() > 0);

            blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
            if (blks.get_total_size() > 0)
            {
                multi_future<R> mf(blks.get_num_blocks());
                for (size_t i = 0; i < blks.get_num_blocks(); ++i)
                    mf[i] = submit(std::forward<F>(loop), blks.start(i), blks.end(i));
                return mf;
            }
            else
            {
                return multi_future<R>();
            }
        }

        // num_blocks 개수로 나눠 queue 에 등록해 Parallelize 시킨다. multi_future 를 받아 wait() 한다.
        template <typename F, typename T, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
        [[nodiscard]] multi_future<R> parallelize_loop(const T index_after_last, F&& loop, const size_t num_blocks = 0)
        {
            assert(get_thread_count() > 0);

            return parallelize_loop(0, index_after_last, std::forward<F>(loop), num_blocks);
        }

		///=================================================================================
		// 사용1) function 과 argument 들을 task queue 에 제출한다.
		// function 이 return value 가 있으면, returned value 를 future 를 통해 반환받도록 한다.
		// function 이 return value 가 없으면, std::future<void> 를 통해 task가 끝날때 까지 wait 할 수 있다.
        ///=================================================================================
        template <typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>
        [[nodiscard]] std::future<R> submit(F&& task, A&&... args)
        {
            assert(get_thread_count() > 0);

            std::shared_ptr<std::promise<R>> task_promise = std::make_shared<std::promise<R>>();
            push_task(
                [task_function = std::bind(std::forward<F>(task), std::forward<A>(args)...), task_promise]
                {
                    try
                    {
                        if constexpr (std::is_void_v<R>)
                        {
                            std::invoke(task_function);
                            task_promise->set_value();
                        }
                        else
                        {
                            task_promise->set_value(std::invoke(task_function));
                        }
                    }
                    catch (...)
                    {
                        try
                        {
                            task_promise->set_exception(std::current_exception());
                        }
                        catch (...)
                        {
                        }
                    }
                });
            return task_promise->get_future();
        }
        ///=================================================================================
        // 사용2) wait_for_tasks() 로 wait 한다.        
        ///=================================================================================
        // num_blocks 개수로 나눠 queue 에 등록해 Parallelize 시킨다. 
        template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>>
        void parallel_for(const T1 first_index, const T2 index_after_last, F&& loop, const size_t num_blocks = 0)
        {
            assert(get_thread_count() > 0);

            blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
            if (blks.get_total_size() > 0)
            {
                for (size_t i = 0; i < blks.get_num_blocks(); ++i)
                    push_task(std::forward<F>(loop), blks.start(i), blks.end(i), i);
            }

            // easy call
            wait_for_tasks();
        }

        // num_blocks 개수로 나눠 queue 에 등록해 Parallelize 시킨다.wait_for_tasks() 로 wait 한다.
        template <typename F, typename T>
        void parallel_for(const T index_after_last, F&& loop, const size_t num_blocks = 0)
        {
            assert(get_thread_count() > 0);

            parallel_for(0, index_after_last, std::forward<F>(loop), num_blocks);
        }

        // num_blocks 개수로 나눠 queue 에 등록해 Parallelize 시킨다.wait_for_tasks() 로 wait 한다.
		template <typename F, typename... A>
		void push_task(F&& task, A&&... args)
		{
            assert(get_thread_count() > 0);
			{
				const std::scoped_lock tasks_lock(tasks_mutex);
				tasks.push(std::bind(std::forward<F>(task), std::forward<A>(args)...)); // cppcheck-suppress ignoredReturnValue
			}
			task_available_cv.notify_one();
		}

        ///=================================================================================
        // 사용2) task 들이 모두 완료되기를 기다린다.
        ///=================================================================================        
        // pool 이 대기 상태에 있으면, 이 함수는 현재 수행중인 task 에 대해서만 대기를 수행한다. (그렇지 않으면 영원이 대기하게 된다.)
        // Note: 특정 task 만 대기하려고 하면 submit() 함수를 사용하던지, 생성된 future 의 wait() 함수를 호출대 대기해라
    private:
        ///=================================================================================
        // Wait for tasks to be completed. Normally, this function waits for all tasks, 
        // both those that are currently running in the threads and those that are still waiting in the queue. 
        // However, if the pool is paused, this function only waits for the currently running tasks (otherwise it would wait forever). 
        // Note: To wait for just one specific task, use submit() instead, and call the wait() member function of the generated future.
        ///=================================================================================
        void wait_for_tasks()
        {
            std::unique_lock tasks_lock(tasks_mutex);
            waiting = true;
            tasks_done_cv.wait(tasks_lock, [this] { return !tasks_running && (paused || tasks.empty()); });
            waiting = false;
        }

        ///=================================================================================
        // Wait for tasks to be completed, but stop waiting after the specified duration has passed.
        ///=================================================================================
        // R            : An arithmetic type representing the number of ticks to wait.
        // P            : An std::ratio representing the length of each tick in seconds.
        // duration     : The time duration to wait.
        // @return      : true if all tasks finished running, false if the duration expired but some tasks are still running.
        ///=================================================================================
        template <typename R, typename P>
        bool wait_for_tasks_duration(const std::chrono::duration<R, P>& duration)
        {
            std::unique_lock tasks_lock(tasks_mutex);
            waiting = true;
            const bool status = tasks_done_cv.wait_for(tasks_lock, duration, [this] { return !tasks_running && (paused || tasks.empty()); });
            waiting = false;
            return status;
        }

        ///=================================================================================
        // @brief Wait for tasks to be completed, but stop waiting after the specified time point has been reached.
        ///=================================================================================
        // C            : The type of the clock used to measure time.
        // D            : An std::chrono::duration type used to indicate the time point.
        // timeout_time : The time point at which to stop waiting.
        // @return      : true if all tasks finished running, false if the time point was reached but some tasks are still running.
        ///=================================================================================
        template <typename C, typename D>
        bool wait_for_tasks_until(const std::chrono::time_point<C, D>& timeout_time)
        {
            std::unique_lock tasks_lock(tasks_mutex);
            waiting = true;
            const bool status = tasks_done_cv.wait_until(tasks_lock, timeout_time, [this] { return !tasks_running && (paused || tasks.empty()); });
            waiting = false;
            return status;
        }


    public:
        ///=================================================================================
        // 사용3) for_loop  thread 없이 즉시 for-loop
        ///=================================================================================
        template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>>
        void for_loop(const T1 first_index, const T2 index_after_last, F&& loop, const size_t num_blocks = 0)
        {
            loop_task(std::forward<F>(loop), first_index, index_after_last, num_blocks);
        }
		template <typename F, typename T>
        void for_loop(const T index_after_last, F&& loop, const size_t num_blocks = 0)
        {
            for_loop(0, index_after_last, std::forward<F>(loop), num_blocks);
        }

		template <typename F, typename... A>
        void loop_task(F&& task, A&&... args)
        {
            std::function<void()> task_function = std::bind(std::forward<F>(task), std::forward<A>(args)...);
            {
                std::invoke(task_function);
            }
        }

        ///=================================================================================
        /// thread
        ///=================================================================================
 public:
		void set_cpuaffinity(int cpu)
		{
			cpu_number = cpu;
			// set cpu affinity 설정
			if (cpu_number >= 0)
			{
				for (concurrency_t i = 0; i < thread_count; ++i)
				{
					thread::SetProcessorCPU(threads[i].native_handle(), cpu_number + i);
				}
			}
		}
    private:
        // pool 에 thread 를 생성하고 worker 를 등록합니다.
        void create_threads()
        {
			{
				const std::scoped_lock tasks_lock(tasks_mutex);
				workers_running = true;
			}
			for (concurrency_t i = 0; i < thread_count; ++i)
			{
				threads[i] = std::thread(&thread_pool::worker, this);
				// thread id to thread index
				threads_id.emplace(threads[i].get_id(), i);
				// set thread name 설정 : std::format : CPP20 부터 지원
				thread::ThreadName(threads[i].native_handle(), std::format("{}_thread_pool_{}", name.c_str(), i).c_str());

				// set cpu affinity 설정
				if (cpu_number >= 0)
					thread::SetProcessorCPU(threads[i].native_handle(), cpu_number + i);
			}
        }

        // pool 에 있는 thread 들을 소멸시킵니다.
        void destroy_threads()
        {
			{
				const std::scoped_lock tasks_lock(tasks_mutex);
				workers_running = false;
			}
			task_available_cv.notify_all();
			for (concurrency_t i = 0; i < thread_count; ++i)
			{
				threads[i].join();
			}
			thread_count = 0;
			threads_id.clear();
        }


        // pool 의 thread 개수를 결정한다. (constructor or reset() 에 전달할 용도로)
		[[nodiscard]] concurrency_t determine_thread_count(const concurrency_t thread_count_) const
		{
			if (thread_count_ > 0)				                return thread_count_;
			else if (std::thread::hardware_concurrency() > 0)   return std::thread::hardware_concurrency();
    		else					                            return 1;
		}

        // pool 의 thread 에서 작동되어지는 worker 함수
        // push_task() 에 의해 task 가 존재할때까지 대기 하다가 queue 에서 Task를 받아 수행한다.
        // task 가 완료되면, worker는 wait_for_tasks() 에게 알립니다.
		void worker()
		{
			std::function<void()> task;
			while (true)
			{
				std::unique_lock tasks_lock(tasks_mutex);
				task_available_cv.wait(tasks_lock, [this] { return !tasks.empty() || !workers_running; });
				if (!workers_running)
					break;
				if (paused)
					continue;
				task = std::move(tasks.front());
				tasks.pop();
				++tasks_running;
				tasks_lock.unlock();
				task();
				tasks_lock.lock();
				--tasks_running;
				if (waiting && !tasks_running && (paused || tasks.empty()))
					tasks_done_cv.notify_all();
			}
		}
    };



	//--------------------------------------------------------------------
    // class: synced_stream
    // Desc	: synchronize printing to an output stream by different threads.
    //--------------------------------------------------------------------
    class [[nodiscard]] synced_stream
    {
    private:
        std::ostream& out_stream;                   // The output stream to print to.
        mutable std::mutex stream_mutex = {};       // A mutex to synchronize printing.

    public:
		// A stream manipulator to pass to a synced_stream (an explicit cast of std::endl). 
		// Prints a newline character to the stream, and then flushes it. 
		// Should only be used if flushing is desired, otherwise '\n' should be used instead.
		inline static std::ostream& (&endl)(std::ostream&) = static_cast<std::ostream & (&)(std::ostream&)>(std::endl);

		// A stream manipulator to pass to a synced_stream (an explicit cast of std::flush). Used to flush the stream.
		inline static std::ostream& (&flush)(std::ostream&) = static_cast<std::ostream & (&)(std::ostream&)>(std::flush);

    public:
	    // Construct a new synced stream.
        synced_stream(std::ostream& out_stream_ = std::cout) : out_stream(out_stream_) {}   // out_stream_: The output stream to print to.The default value is std::cout.

        // Ensures that no other threads print to this stream simultaneously, 
        // as long as they all exclusively use the same synced_stream object to print.
        // print items
        template <typename... T>
        void print(T&&... items)
        {
            const std::scoped_lock lock(stream_mutex);
            (out_stream << ... << std::forward<T>(items));
        }
        // print items & newline character
        template <typename... T>
        void println(T&&... items)
        {
            print(std::forward<T>(items)..., '\n');
        }
    };


	//--------------------------------------------------------------------
	// class: timer
	// Desc	: A helper class to measure execution time for benchmarking purposes.
	//--------------------------------------------------------------------
    class [[nodiscard]] timer
    {
    private:
        std::chrono::time_point<std::chrono::steady_clock>  start_time  = std::chrono::steady_clock::now();         // The time point when measuring started.
        std::chrono::duration<double>                       elapsed_time= std::chrono::duration<double>::zero();    // The duration that has elapsed between start() and stop().
    public:

        void start()        {   start_time      = std::chrono::steady_clock::now();                }       // Start(or restart) measuring time.
        void stop()         {   elapsed_time    = std::chrono::steady_clock::now() - start_time;   }       // Stop measuring time and store the elapsed time since start().
        
        // The number of milliseconds. : Get the number of milliseconds that have elapsed between start() and stop(). 
        [[nodiscard]] std::chrono::milliseconds::rep ms() const     {   return (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time)).count();   }
        [[nodiscard]] float sec() const                             {   return (ms() / 1000.f);    }
    };

}