# Yolact模型在C++端部署

最近在做动态SLAM，需要将实例分割模型部署在C++端，总结一下整体的流程。

## Yolact+opencv+DNN+onnx C++部署

深度学习的模型部署有很多方式，不过实现起来都不太简单。一开始尝试了一下将yolact的pth模型转化为onnx模型，再通过OPENCV中的DNN库进行读取运行。不过运行速度很慢一帧图片处理时间在2.7秒左右。参考的是https://github.com/hpc203/yolact-opencv-dnn-cpp-python。按理说在C++中运行速度应该要比纯python快上很多，没细看代码也不太清楚怎么回事。这个要注意一下opencv的版本应该在4.5.1以上，不然读取onnx会报错。原作者只给出了C++程序，ubuntu环境下使用cmake进行编译需要CMakeLists文件，下面给出我的cmake文件，可以支持切换opencv3和opencv4。相关地址要改成自己的绝对地址。

```
cmake_minimum_required(VERSION 2.8)
project(test)
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()
# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
set(CMAKE_PREFIX_PATH "/home/run/opencv4")
#set(OpenCV_DIR /home/run/opencv4)
#message("PROJECT_SOURCE_DIR: " ${OpenCV_DIR})
find_package(OpenCV 4.5.1 QUIET)
#find_package(OpenCV 3.4.10 QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
add_executable(test main.cpp)
target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        )
```

问题就是运行速度太慢了，参考网上的解决办法，我的想法是使用onnx runtime或者TensorRT这种改写程序加速一下，不过这些都没接触过，有点怕浪费时间，以后再尝试把。

## 参考DynaSLAM，使用C++调用python动态库的方式进行部署

整体流程就是安装Yolact所需环境—>编写C++调用python测试程序以及配置CMake环境—>编写yolact_interface.py程序—>yolact_interface python测试—>c++测试

### Yolact安装

***

本来打算使用Yolact++，编译不成功，后续考虑换成Yolact edge实时性更好检测精度稍低

#### 环境配置

CUDA下载：

https://developer.nvidia.com/cuda-toolkit-archive

安装脚本之后配置环境变量：

  sudo  gedit ~/.bashrc打开文件，在文件结尾添加如下语句：

```
export PATH=$PATH:/usr/local/cuda-11.3.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3.1/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-11.3.1/lib64
```

终端运行source ~/.bashrc 

CUDNN安装:

https://blog.csdn.net/weixin_44009878/article/details/113926893

CUDA卸载：

```
cd /usr/local/cuda-11.4/bin
sudo ./cuda-uninstaller
```

安装Anaconda：

https://blog.csdn.net/KIK9973/article/details/118772450

#### Yolact配置

这部分按照github安装即可需要注意torch安装（巨坑！！！）conda install 的torch基本都是cpu版本（命令正确也会默认安装cpu版本），需要用pip安装才是正常gpu版本。出现报错要注意一下torch版本是否为gpu然后再去看CUDA版本的对应关系。

下载网站：https://pytorch.org/get-started/previous-versions/

```
conda env create -f environment.yml
conda activate yolact-env
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install cython
pip install opencv-python pillow pycocotools matplotlib 
```

#### yolact测试

国内服务器下载模型文件https://www.cnblogs.com/isLinXu/p/15309056.html

在yolact目录新建weights文件夹，将下载的模型文件放到weights文件夹中

激活虚拟环境：

conda activate yolact-env

进入yolact的目录下

cd yolact-master

新建input和output文件夹，将测试图片放在input中，测试结果生成在output中

运行测试：

python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.15 --top_k=15 --images=input:output 

模型预测时可指定的参数

* –trained_model：选择模型文件
* –top_k：保存置信率最高的前15个目标
* –config：选择config配置文件
* –image：对单张图像进行预测，路径为单张图像
* –images：对多张图像/图像路径文件夹 进行预测
* –video：指定视频进行预测
* –score_threshold ：剔除掉低于0.15置信度的目标
* –dataset：数据集路径

### c++部署

***

> 参考资料:
>
> C++调用 Python解释器动态链接库的方式获取语义信息
>
> https://zhuanlan.zhihu.com/p/450318119
>
> c++调用anaconda中python cmake编写：
>
> https://blog.csdn.net/qq_38766208/article/details/124156182
>
> https://blog.csdn.net/HelloJinYe/article/details/107162527

整体框架参考https://github.com/DreamWaterFound/Prerequisites-of-On-line-Semantic-VSLAM

#### c++调用python遇见的一些报错：

* python.h没有那个文件或目录

解决办法：sudo apt-get install python3.7-dev或者检查CmakeList编译文件

* opencv 多版本安装：

https://blog.csdn.net/wyyang2/article/details/103989455

* TIFF未引用在cmake时加参数：cmake -D BUILD_TIFF=ON

完整Cmake命令：cmake -D CMAKE_INSTALL_PREFIX=/home/run/opencv4 -D CMAKE_BUILD_TYPE="Rlease" -D OPENCV_GENERATE_PKGCONFIG=ON -DBUILD_TIFF=ON ..

* opencv4和3分别使用 cmakehttps://blog.csdn.net/reasonyuanrobot/article/details/121034930

* pytorch报错：legacy constructor expects device type: cpu but device type

  添加代码：torch.set_default_tensor_type('torch.cuda.FloatTensor')

* cv2.imwrite(name,img)返回值为False。

  路径错误或路径中包含有中文字符，路径为你终端当前路径

整体项目文件：

