[![Action Status]](https://github.com/zjysnow/ChaosLab/actions/workflows/cmake.yml/badge.svg)](https://github.com/zjysnow/ChaosLab/actions/workflows/cmake.yml)
[![Build Status](https://dev.azure.com/zjysnow/ChaosLab/_apis/build/status/ChaosLab?branchName=main)](https://dev.azure.com/zjysnow/ChaosLab/_build/latest?definitionId=32&branchName=main)

![QQ](https://img.shields.io/badge/QQ-980428900-grenn?logo=tencentqq)

# ChaosLab
ChaosCV and Pattern Recognition

## ChaosCV
Shared Items Project
1. core
2. dnn
3. highgui

## Build
提供两种方式编译项目，使用Visual Studio能够保证完全的功能编译，CMake还在学习中，由于目前VS对C++20支持的最完整，因此暂时不提供兼容其他平台的版本

### Visual Studio
直接使用Visual Studio 2019打开SLN文件编译即可，如果需要使用Sandbox项目，则需要自行创建；需要配置好WSL

### CMake
参考[[shufaCV](https://github.com/scarsty/shufaCV)]项目

```cmake
mkdir build
cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
```