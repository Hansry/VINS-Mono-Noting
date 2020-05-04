# VINS-Mono-Learning

 VINS-Mono代码注释 by Hansry

---

## VINS介绍：

VINS是一种具有鲁棒性和通用性的单目视觉惯性状态估计器。  
该算法主要有以下几个模块：  
 1. 预处理  
&emsp; &emsp;1）图像特征光流跟踪  
&emsp; &emsp;2）IMU数据预积分  
 2. 初始化  
&emsp; &emsp;1）纯视觉Sfm  
&emsp; &emsp;2）Sfm与IMU预积分的松耦合  
 3. 基于滑动窗口的非线性优化  
 4. 回环检测与重定位  
 5. 四自由度位姿图优化  

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104194533165.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxODM5MjIy,size_16,color_FFFFFF,t_70)

---

## 代码的文件目录

1、ar_demo：一个ar应用demo  
2、benchmark\_publisher：接收并发布数据集的基准值  
3、camera\_model  
&emsp; &emsp;calib：相机参数标定  
&emsp; &emsp;camera\_models：各种相机模型类  
&emsp; &emsp;chessboard：检测棋盘格  
&emsp; &emsp;gpl  
&emsp; &emsp;sparse\_graph  
&emsp; &emsp;intrinsic\_calib.cc：相机标定模块main函数  
4、config：系统配置文件存放处  
5、feature\_trackers：  
&emsp; &emsp;feature\_tracker\_node.cpp	ROS 节点函数，回调函数  
&emsp; &emsp;feature\_tracker.cpp	图像特征光流跟踪
6、pose\_graph：  
&emsp; &emsp;keyframe.cpp	关键帧选取、描述子计算与匹配   
&emsp; &emsp;pose\_graph.cpp 位姿图的建立与图优化  
&emsp; &emsp;pose\_graph\_node.cpp	ROS 节点函数，回调函数，主线程  
7、support\_files：帮助文档、Bow字典、Brief模板文件  
8、vins\_estimator   
&emsp; &emsp;factor：实现IMU、camera等残差模型  
&emsp; &emsp;initial：系统初始化，外参标定，SFM  
&emsp; &emsp;utility：相机可视化，四元数等数据转换  
&emsp;&emsp; estimator.cpp：紧耦合的VIO状态估计器实现  
&emsp;&emsp; estimator\_node.cpp：ROS 节点函数，回调函数， 主线程  
&emsp; &emsp;feature\_manager.cpp：特征点管理，三角化，关键帧等  
&emsp; &emsp;parameters.cpp：读取参数  

--------------------- 

## 参考资料：  
主要参考了崔华坤的《VINS论文推导及代码解析》  
1、IMU预积分：  
&emsp; &emsp;[VINS-Mono之IMU预积分，预积分误差、协方差及误差对状态量雅克比矩阵的递推方程的推导](https://blog.csdn.net/Hansry/article/details/104203448)  
&emsp; &emsp;[VINS-Mono理论学习——IMU预积分 Pre-integration (Jacobian 协方差)](https://blog.csdn.net/qq_41839222/article/details/86290941)  
2、视觉IMU联合初始化：  
&emsp; &emsp;[VINS-Mono之外参标定和视觉IMU联合初始化](https://blog.csdn.net/Hansry/article/details/104365306)  
&emsp; &emsp;[VINS-Mono理论学习——视觉惯性联合初始化与外参标定](https://blog.csdn.net/qq_41839222/article/details/89106128)  
3、后端非线性优化：  
&emsp; &emsp;[VINS-Mono之后端非线性优化](https://blog.csdn.net/Hansry/article/details/104234046)  
&emsp; &emsp;[VINS-Mono理论学习——后端非线性优化](https://blog.csdn.net/qq_41839222/article/details/93593844)  
4、边缘化：  
&emsp; &emsp;[VSLAM之边缘化 Marginalization 和 FEJ (First Estimated Jocobian)](https://blog.csdn.net/Hansry/article/details/104412753)  
&emsp; &emsp;[VINS-Mono关键知识点总结——边缘化marginalization理论和代码详解](https://blog.csdn.net/weixin_44580210/article/details/95748091)  
5、回环检测与重定位：  
&emsp; &emsp;[vins-mono代码阅读之4自由度位姿图优化](https://zhuanlan.zhihu.com/p/90495876)  
&emsp; &emsp;[VINS-Mono代码解读——回环检测与重定位 pose graph loop closing](https://blog.csdn.net/qq_41839222/article/details/87878550)  

参考VINS代码注释：

https://github.com/ManiiXu/VINS-Mono-Learning by [ManiiXu](https://github.com/ManiiXu)

----
