//
// Created by guiwenhou on 2020/9/10.
//

#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <iostream>

class ApplyFoo {

private:
    float *const my_a;
public:
    void operator()(const tbb::blocked_range <size_t> &r) const {
        float *a = my_a;
        for (int i = r.begin(); i != r.end(); ++i) {
            printf("gevar = %f\n", a[i]);
        }
    }

    ApplyFoo(float a[]) :
            my_a(a) {
    }
};

void parallelApplyFoo(float a[], size_t n) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), ApplyFoo(a), tbb::auto_partitioner());
}

int main() {
    //tbb::task_scheduler_init init;
    float a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    parallelApplyFoo(a, 10);
    return 0;
}