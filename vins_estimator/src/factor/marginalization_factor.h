#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;//parameter_blocks存放marg相关变量的数据
    std::vector<int> drop_set;//待边缘化的优化变量id

    double **raw_jacobians; //Jacobian
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;//残差 IMU:15X1 视觉2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size指表示位姿的时候有7个维度，平移3+旋转四元素4
    std::unordered_map<long, int> parameter_block_idx; //local size指位姿有6个自由度，优化的时候应该是按照李代数进行优化的
};

///@brief 主要是完成边缘化操作
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    //添加残差块相关信息（优化变量、待边缘化的变量）
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    //计算每个残差对应的Jacobian,并更新parameter_block_data
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;//所有观测项
    int m, n;//m为要边缘化的变量个数乘以变量的长度（localSize） ，n为要保留下来的变量个数乘以变量的长度
    std::unordered_map<long, int> parameter_block_size; //<优化变量内存地址,global size>,global size为各个优化变量的长度（Pose为７个维度）（hansry）
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //<待边缘化的优化变量内存地址,在parameter_block_size中的id> (local size)
    std::unordered_map<long, double *> parameter_block_data;//<优化变量内存地址,double指针类型数据>

    std::vector<int> keep_block_size; //按顺序存放上面的parameter_block_size中被保留的优化变量的长度（global size）
    std::vector<int> keep_block_idx;  //按顺序存放上面的parameter_block_idx中被保留的优化变量的id　（local size）
    std::vector<double *> keep_block_data;//按顺序存放上面的parameter_block_data中被保留的优化变量的double指针类型数据

    Eigen::MatrixXd linearized_jacobians;//边缘化得到的雅克比矩阵
    Eigen::VectorXd linearized_residuals;//边缘化得到的残差向量
    const double eps = 1e-8;
};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
