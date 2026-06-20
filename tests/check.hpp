#pragma once

// A tiny zero-dependency check harness. CHECK records a failure and keeps
// going; main() returns the failure count, which CTest reads as pass/fail.

#include <cstdio>

namespace test {
inline int& failures() {
    static int count = 0;
    return count;
}
}  // namespace test

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
            ++::test::failures();                                         \
        }                                                                 \
    } while (0)
