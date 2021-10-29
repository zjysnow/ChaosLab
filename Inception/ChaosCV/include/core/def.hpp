#pragma once

#define APPLICATION_NAME "ChaosCV"
#define ENGINE_NAME "ChaosEngine"

#define MAJOR 1
#define MINOR 0
#define PATCH 0
#define ENGINE_VERSION 20210321

#ifdef _WIN32
#ifdef CHAOS_EXPORT
#define CHAOS_API __declspec(dllexport)
#else
#define CHAOS_API __declspec(dllimport)
#endif
#else
#define CHAOS_API
#endif

#define CHAOS_PREDICT_BRANCH_NOT_TAKEN(x) x

#define LOG(severity) chaos::LogMessage(__FILE__, __LINE__, chaos::##severity).stream()

#define LOG_IF(severity, condition) \
  !(condition) ? (void) 0 : chaos::LogMessageVoidify() & LOG(severity)

// CHECK dies with a fatal error if condition is not true.
#define CHECK(condition) LOG_IF(FATAL, CHAOS_PREDICT_BRANCH_NOT_TAKEN(!(condition))) << "Check failed: " #condition ". "

#define CHECK_EQ(val1, val2) CHECK(val1 == val2)
#define CHECK_NE(val1, val2) CHECK(val1 != val2)
#define CHECK_LE(val1, val2) CHECK(val1 <= val2)
#define CHECK_LT(val1, val2) CHECK(val1 <  val2)
#define CHECK_GE(val1, val2) CHECK(val1 >= val2)
#define CHECK_GT(val1, val2) CHECK(val1 >  val2)

#define CHECK_NEAR(val1, val2, margin)  \
  do {                                  \
    CHECK_LE((val1), (val2)+(margin));  \
    CHECK_GE((val1), (val2)-(margin));  \
  } while (false);