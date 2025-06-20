#ifndef MPC_H
#define MPC_H

#include <Eigen/Dense>  // 包含 Eigen 库，用于矩阵运算
#include <qpOASES.hpp>

// OpenCV 相关的头文件，虽然本MPC逻辑中未使用，但保留以符合用户提供的上下文
#include <opencv2/opencv.hpp>

class MPC
{
public:
    /**
     * @brief MPC 控制器的构造函数。
     * @param N 预测时域 (Prediction Horizon)。
     * @param Ac 连续时间状态空间矩阵 A。
     * @param Bc 连续时间状态空间矩阵 B。
     * @param C 输出矩阵 C。
     * @param Q 输出误差权重矩阵 (ny x ny)。
     * @param R 控制输入权重矩阵 (nu x nu)。
     * @param dt 采样时间 (Sampling Time)。
     */
    MPC(int N,
        const Eigen::MatrixXd& Ac,
        const Eigen::MatrixXd& Bc,
        const Eigen::MatrixXd& C,
        const Eigen::MatrixXd& Q,
        const Eigen::MatrixXd& R,
        double dt);

    /**
     * @brief 求解 MPC 问题，预测并返回最优控制输入序列。
     * 此方法将构建二次规划 (QP) 问题并尝试求解。
     * @param current_x 当前状态向量 (nx x 1)。
     * @param ref_horizon 预测时域内的参考轨迹 (N x ny)。
     * @return 最优控制输入序列 (N x nu)。在实际中通常只取第一个控制量。
     */
    Eigen::MatrixXd solve(const Eigen::VectorXd& current_x,
                          const Eigen::MatrixXd& ref_horizon);

    /**
     * @brief 模拟MPC控制下的系统预测轨迹和成本。
     * 这是一个辅助函数，用于验证给定控制输入序列的效果，不进行优化。
     * @param current_x 初始状态。
     * @param u_horizon 要应用的控制输入序列 (N x nu)。
     * @param ref_horizon 参考轨迹 (N x ny)。
     * @return 包含预测输出轨迹和总成本的pair。
     */
    std::pair<Eigen::MatrixXd, double> simulate_prediction(
        const Eigen::VectorXd& current_x,
        const Eigen::MatrixXd& u_horizon,
        const Eigen::MatrixXd& ref_horizon) const;

private:
    int N_;      // 预测时域
    int nx_;     // 状态变量数量
    int nu_;     // 输入变量数量
    int ny_;     // 输出变量数量
    double dt_;  // 采样时间

    Eigen::MatrixXd Ad_; // 离散时间状态转移矩阵
    Eigen::MatrixXd Bd_; // 离散时间控制输入矩阵
    Eigen::MatrixXd C_;  // 输出矩阵
    Eigen::MatrixXd Q_;  // 输出误差权重矩阵
    Eigen::MatrixXd R_;  // 控制输入权重矩阵

    /**
     * @brief 将连续时间状态空间模型离散化为离散时间模型。
     * 使用零阶保持 (Zero-Order Hold, ZOH) 方法。
     * @param Ac_cont 连续时间 A 矩阵。
     * @param Bc_cont 连续时间 B 矩阵。
     * @param dt 采样时间。
     * @param Ad_disc 离散化后的 A 矩阵引用。
     * @param Bd_disc 离散化后的 B 矩阵引用。
     */
    void discretize_model(const Eigen::MatrixXd& Ac_cont,
                          const Eigen::MatrixXd& Bc_cont,
                          double dt,
                          Eigen::MatrixXd& Ad_disc,
                          Eigen::MatrixXd& Bd_disc);

    /**
     * @brief 构建MPC问题的H和f矩阵，用于QP求解器。
     * MPC目标函数: J = sum_{i=0}^{N-1} (y_i - r_i)^T Q (y_i - r_i) + u_i^T R u_i
     * 转化为QP形式: min 0.5 * U_horizon^T * H * U_horizon + f^T * U_horizon
     * @param current_x 当前状态。
     * @param ref_horizon 预测时域内的参考轨迹。
     * @param H_qp QP问题中的 H 矩阵引用。
     * @param f_qp QP问题中的 f 向量引用。
     */
    void build_mpc_qp_matrices(const Eigen::VectorXd& current_x,
                               const Eigen::MatrixXd& ref_horizon,
                               Eigen::MatrixXd& H_qp,
                               Eigen::VectorXd& f_qp) const;

    /**
     * @brief 这是一个QP求解器的占位符函数。
     * 在实际应用中，您会在这里集成一个外部的QP求解库，如 qpOASES 或 OSQP。
     * @param H_qp QP问题中的 H 矩阵。
     * @param f_qp QP问题中的 f 向量。
     * @param nu_horizon 控制输入序列的总维度 (N * nu)。
     * @return 优化后的控制输入序列 (N*nu x 1) 向量。
     */
    Eigen::VectorXd solve_qp_placeholder(const Eigen::MatrixXd& H_qp,
                                         const Eigen::VectorXd& f_qp,
                                         int nu_horizon);
};

#endif // MPC_H
