#pragma once

#define NOMINMAX

//=========================================================
// STANDARD
//=========================================================
#include <chrono>             // std::chrono
#include <condition_variable> // std::condition_variable
#include <exception>          // std::current_exception
#include <functional>         // std::bind, std::function, std::invoke
#include <future>             // std::future, std::promise
#include <iostream>           // std::cout, std::endl, std::flush, std::ostream
#include <memory>             // std::make_shared, std::make_unique, std::shared_ptr, std::unique_ptr
#include <mutex>              // std::mutex, std::scoped_lock, std::unique_lock
#include <queue>              // std::queue
#include <thread>             // std::thread
#include <type_traits>        // std::common_type_t, std::conditional_t, std::decay_t, std::invoke_result_t, std::is_void_v
#include <utility>            // std::forward, std::move, std::swap
#include <vector>             // std::vector


#include <windows.h>          
#include <psapi.h>
#include <pdh.h>
#include <dbghelp.h>
#include <crtdbg.h>
#include <iphlpapi.h>



//=========================================================
// ENTT (Entity Component System)
// https://github.com/skypjack/entt
//=========================================================
#include "entt/entt.hpp"

//======= CORE ======= 
#include "CoreTypeDef.h"

//======= MATHEMATICS ======= 
#include "mathematics/ECSMath.h"

//======= THREAD ======= 
#include "thread/Thread_win32.h"
#include "thread/Thread_Pool.h"

//======= Source ======= 
#include "Components.h"


