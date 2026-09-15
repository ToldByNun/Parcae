# Parcae compiler warning helpers
#
# Usage from the root CMakeLists.txt:
#   include(cmake/CompilerWarnings.cmake)
#   parcae_set_project_warnings(<target>)

function(parcae_set_project_warnings target_name)
  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR "parcae_set_project_warnings: target '${target_name}' does not exist")
  endif()

  if(MSVC)
    target_compile_options("${target_name}" PRIVATE
      /W4
      /permissive-
      /w14242 # conversion: possible loss of data
      /w14254 # operator: conversion from T1 to T2
      /w14263 # member function does not override any base class virtual member
      /w14265 # class has virtual functions, but destructor is not virtual
      /w14287 # unsigned/negative constant mismatch
      /we4289 # loop control variable declared in the for-loop is used outside
      /w14296 # expression is always false/true
      /w14311 # pointer truncation from T1 to T2
      /w14545 # expression before comma evaluates to a function lacking side effects
      /w14546 # function call before comma missing argument list
      /w14547 # operator before comma has no effect
      /w14549 # operator before comma has no effect; did you intend 'operator'?
      /w14555 # expression has no effect
      /w14619 # pragma warning: there is no warning number N
      /w14640 # thread-unsafe static member initialization
      /w14826 # assignment by itself has no effect
      /w14905 # wide string literal cast to LPSTR
      /w14906 # string literal cast to LPWSTR
      /w14928 # illegal copy-initialization
    )
  else()
    target_compile_options("${target_name}" PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wsign-conversion
      -Wshadow
      -Wnon-virtual-dtor
      -Wold-style-cast
      -Wcast-align
      -Wunused
      -Woverloaded-virtual
      -Wnull-dereference
      -Wdouble-promotion
      -Wformat=2
    )
  endif()
endfunction()
