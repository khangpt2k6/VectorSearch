#include "vectorsearch/distance.hpp"

#include <cmath>
#include <initializer_list>

#include "check.hpp"

using namespace vectorsearch;

int main() {
    // squared_l2
    {
        float a[] = {0.0f, 0.0f};
        float b[] = {3.0f, 4.0f};
        CHECK(squared_l2(a, b, 2) == 25.0f);
        CHECK(squared_l2(a, a, 2) == 0.0f);
    }

    // dot
    {
        float a[] = {1.0f, 2.0f, 3.0f};
        float b[] = {4.0f, 5.0f, 6.0f};
        CHECK(dot(a, b, 3) == 32.0f);
    }

    // cosine: identical ~0, orthogonal ~1, zero vector == 1
    {
        float v[] = {1.0f, 2.0f, 3.0f};
        CHECK(std::fabs(cosine_distance(v, v, 3)) < 1e-6f);

        float x[] = {1.0f, 0.0f};
        float y[] = {0.0f, 1.0f};
        CHECK(std::fabs(cosine_distance(x, y, 2) - 1.0f) < 1e-6f);

        float zero[] = {0.0f, 0.0f};
        float ones[] = {1.0f, 1.0f};
        CHECK(cosine_distance(zero, ones, 2) == 1.0f);
    }

    // smaller-is-closer for direction-based metrics
    {
        float q[] = {1.0f, 0.0f, 0.0f};
        float near[] = {0.9f, 0.1f, 0.0f};
        float far[] = {-1.0f, 0.0f, 0.0f};
        for (Metric m : {Metric::L2, Metric::Cosine}) {
            CHECK(distance(m, q, near, 3) < distance(m, q, far, 3));
        }
    }

    if (test::failures() == 0) {
        std::printf("test_distance: all checks passed\n");
    }
    return test::failures() == 0 ? 0 : 1;
}
