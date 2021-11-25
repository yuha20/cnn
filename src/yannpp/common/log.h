#ifndef LOG_H
#define LOG_H
#define VERBOSE 1
#define PARALLEL_FOR 0
#define PARALLEL_REDUCE 0

namespace yannpp {
    template<typename T>
    class array3d_t;

    void log(const char *fmt, ...);
    void logD(const char *fmt, ...);
    void log(array3d_t<float> const &arr);
}

#endif // LOG_H
