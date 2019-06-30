#pragma once

#include <cstdint>
#include <cassert>
#include <vector>
#include <thread>
#include <atomic>
#include <immintrin.h>

//#define GLM_FORCE_MESSAGES
//#define GLM_FORCE_ALIGNED_GENTYPES //TODO: Force aligned vector types!
#define GLM_FORCE_INLINE
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>
#include <glm/mat4x4.hpp>

#include "Utils.h"