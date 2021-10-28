![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)
[![windows](https://github.com/zjysnow/ChaosLab/actions/workflows/cmake.yml/badge.svg)](https://github.com/zjysnow/ChaosLab/actions/workflows/cmake.yml)
[![Build Status](https://dev.azure.com/zjysnow/ChaosLab/_apis/build/status/ChaosLab?branchName=main)](https://dev.azure.com/zjysnow/ChaosLab/_build/latest?definitionId=32&branchName=main)
[![Language grade: C/C++](https://img.shields.io/lgtm/grade/cpp/g/zjysnow/ChaosLab.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/zjysnow/ChaosLab/context:cpp)

![QQ](https://img.shields.io/badge/QQ-980428900-grenn?logo=tencentqq)

# ChaosLab
ChaosCV and Pattern Recognition

## ChaosCV
Shared Items Project
1. core
2. dnn
3. highgui

## Build
提供两种方式编译项目

### Visual Studio
直接使用Visual Studio 2019打开SLN文件编译即可，如果需要使用Sandbox项目，则需要自行创建；需要配置好WSL

### CMake
参考[[shufaCV](https://github.com/scarsty/shufaCV)]项目

```cmake
mkdir build
cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
```