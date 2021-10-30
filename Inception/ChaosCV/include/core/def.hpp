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

#define LOG(severity) chaos::LogMessage(__FILE__, __LINE__, severity).stream()

#define LOG_IF(severity, condition) \
  !(condition) ? (void) 0 : chaos::LogMessageVoidify() & LOG(severity)

#ifdef NDEBUG
#define CHECK(condition) LOG_IF(ERROR, CHAOS_PREDICT_BRANCH_NOT_TAKEN(!(condition))) << "Check failed: " #condition ". "
#else
// CHECK dies with a fatal error if condition is not true.
#define CHECK(condition) LOG_IF(FATAL, CHAOS_PREDICT_BRANCH_NOT_TAKEN(!(condition))) << "Check failed: " #condition ". "
#endif

#define CHECK_EQ(val1, val2) CHECK(val1 == val2)
#define CHECK_NE(val1, val2) CHECK(val1 != val2)
#define CHECK_LE(val1, val2) CHECK(val1 <= val2)
#define CHECK_LT(val1, val2) CHECK(val1 <  val2)
#define CHECK_GE(val1, val2) CHECK(val1 >= val2)
#define CHECK_GT(val1, val2) CHECK(val1 >  val2)

//#define CHECK_NEAR(val1, val2, margin)  \
//  do {                                  \
//    CHECK_LE((val1), (val2)+(margin));  \
//    CHECK_GE((val1), (val2)-(margin));  \
//  } while (false)

// debug-logging macros
#if defined(NDEBUG) and not defined(CHECK_ALWAYS_ON)
#define DLOG(severity) \
  true ? (void) 0 : chaos::LogMessageVoidify() & LOG(severity)

#define DLOG_IF(severity, condition) \
  (true || !(condition)) ? (void) 0 : chaos::LogMessageVoidify() & LOG(severity);

#define DCHECK(condition) while(false) CHECK(condition)

#define DCHECK_EQ(val1, val2) while(false) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) while(false) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) while(false) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) while(false) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) while(false) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) while(false) CHECK_GT(val1, val2)
#else
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, condition) LOG_IF(severity, condition)

#define DCHECK(condition) CHECK(condition)

#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)
#endif
