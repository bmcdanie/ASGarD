#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <type_traits>

// value return version
template<typename F, typename... Args>
auto timer(F &&f, std::string identifier, Args &&... args) ->
    typename std::enable_if<
        !std::is_same<decltype(f(std::forward<Args>(args)...)), void>::value,
        decltype(f(std::forward<Args>(args)...))>::type
{
  auto const beg = std::chrono::high_resolution_clock::now();
  auto const ret = std::forward<F>(f)(std::forward<Args>(args)...);
  auto const end = std::chrono::high_resolution_clock::now();
  auto const dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
  std::cout << identifier << " :" << dur << ": ms\n";
  return ret;
}

// void return version
template<typename F, typename... Args>
auto timer(F &&f, std::string identifier, Args &&... args) ->
    typename std::enable_if<
        std::is_same<decltype(f(std::forward<Args>(args)...)), void>::value,
        void>::type
{
  auto const beg = std::chrono::high_resolution_clock::now();
  std::forward<F>(f)(std::forward<Args>(args)...);
  auto const end = std::chrono::high_resolution_clock::now();
  auto const dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
  std::cout << identifier << " :" << dur << ": ms\n";
}
