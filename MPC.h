#ifndef MPC_H
#define MPC_H

#include <Eigen/Dense>    // 包含 Eigen 库，用于矩阵运算
#include <qpOASES.hpp>    // 包含 qpOASES 库，用于二次规划求解
#include <utility>        // 用于 std::pair

// OpenCV 相关的头文件，尽管 MPC 逻辑中未使用，但保留以符合用户提供的上下文。
// #include <opencv2/opencv.hpp> // 如果项目其他部分需要，请取消注释

/**
 * @brief 辅助函数：使用 Pade 近似计算矩阵指数。
 * @param M 要计算指数的方阵。
 * @param order Pade 近似的阶数 (例如，6 表示 (6,6) 阶近似)。
 * @return 矩阵 M 的指数，即 exp(M)。
 */
Eigen::MatrixXd matrix_exponential_pade(const Eigen::MatrixXd& M, int order);


class MPC
{
public:
    /**
     * @brief MPC 控制器的构造函数。
     * @param N_p 预测时域 (Np)。
     * @param N_c 控制时域 (Nc)。
     * @param Ac 连续时间状态空间矩阵 A。
     * @param Bc 连续时间状态空间矩阵 B。
     * @param C 输出矩阵 C。
     * @param Q 输出误差权重矩阵 (ny x ny)。必须是正半定矩阵。
     * @param R 控制输入权重矩阵 (nu x nu)。必须是正定矩阵。
     * @param dt 采样时间。
     */
    MPC(int N_p,
        int N_c, // 控制时域 Nc
        const Eigen::MatrixXd& Ac,
        const Eigen::MatrixXd& Bc,
        const Eigen::MatrixXd& C,
        const Eigen::MatrixXd& Q,
        const Eigen::MatrixXd& R,
        double dt,
        bool incremental = false); // 是否使用增量控制优化

    /**
     * @brief solve 方法，根据incremental_，求解 MPC 问题，预测并返回最优控制输入序列。
     * @param ref_horizon 预测时域内的参考轨迹 (Np x ny)。
     * @param current_x 当前状态向量 (nx x 1)。
     * @param u_last 上一时刻的控制输入 (nu x 1)。如果为空，则使用内部存储的值。
     * @return
     */
    Eigen::VectorXd solve(const Eigen::MatrixXd& ref_horizon,
                          const Eigen::VectorXd& current_x,
                          const Eigen::VectorXd& u_last = Eigen::VectorXd());
    /**
     * @brief 求解 MPC 问题，预测并返回最优控制输入序列。
     * 此方法将构建二次规划 (QP) 问题并尝试求解。
     * @param current_x 当前状态向量 (nx x 1)。
     * @param ref_horizon 预测时域内的参考轨迹 (Np x ny)。
     * @return 最优控制输入序列 (Nc x nu)。在实际中通常只取第一个控制量。
     */
    Eigen::VectorXd solve_direct(const Eigen::MatrixXd& ref_horizon,
                          const Eigen::VectorXd& current_x);

    //
    /**
     * @brief 求解增量控制优化问题，返回增量控制输入序列。
     * @param current_x 当前状态向量 (nx x 1)。
     * @param ref_horizon 预测时域内的参考轨迹 (Np x ny)。
     * @param u_last 上一时刻的控制输入 (nu x 1)。如果为空，则使用内部存储的值。
     * @return 最优增量控制输入序列 (Nc x nu)。在实际中通常只取第一个增量控制量。
     */
    Eigen::VectorXd solve_incremental(const Eigen::MatrixXd& ref_horizon,
                                      const Eigen::VectorXd& current_x,
                                      const Eigen::VectorXd& u_last = Eigen::VectorXd());

    /**
     * @brief 模拟 MPC 控制下的系统预测轨迹和成本。
     * 这是一个辅助函数，用于验证给定控制输入序列的效果，不进行优化。
     * @param current_x 初始状态。
     * @param u_horizon 要应用的控制输入序列 (Nc x nu)。
     * @param ref_horizon 参考轨迹 (Np x ny)。
     * @return 包含预测输出轨迹 (MatrixXd) 和总成本 (double) 的 pair。
     */
    std::pair<Eigen::MatrixXd, double> simulate_prediction(
        const Eigen::VectorXd& current_x,
        const Eigen::MatrixXd& u_horizon,
        const Eigen::MatrixXd& ref_horizon) const;

    // 添加公共 getter 方法以访问私有成员 Ad_ 和 Bd_
    const Eigen::MatrixXd& getAd() const { return Ad_; }
    const Eigen::MatrixXd& getBd() const { return Bd_; }


public:
    /**
     * @brief 利用 Phi_ 和 G_ 计算预测输出序列。
     * @param x_current 当前状态 (nx x 1)。
     * @param u_horizon 控制输入序列 (Nc*nu x 1)。
     * @return 预测输出序列 (Np*ny x 1)。
     */
    Eigen::VectorXd predict_y_horizon(const Eigen::VectorXd& x_current,
                                      const Eigen::VectorXd& u_horizon,
                                      const Eigen::VectorXd& u_last = Eigen::VectorXd()) const;

    // /**
    //  * @brief 设置上一时刻的控制输入，用于增量控制优化。
    //  * @param u_last 上一时刻的控制输入 (nu x 1)。
    //  */
    // void setLastControlInput(const Eigen::VectorXd& u_last);
    /**
     * @brief 获取上一时刻的控制输入。
     * @return 上一时刻的控制输入 (nu x 1)。
     */
    const Eigen::VectorXd& getLastControlInput() const { return u_last_; }



private:
    int N_p_;     // 预测时域
    int N_c_;     // 控制时域
    int nx_;      // 状态变量数量
    int nu_;      // 输入变量数量
    int ny_;      // 输出变量数量
    double dt_;  // 采样时间

    Eigen::MatrixXd Ad_; // 离散时间状态转移矩阵
    Eigen::MatrixXd Bd_; // 离散时间控制输入矩阵
    Eigen::MatrixXd C_;  // 输出矩阵
    Eigen::MatrixXd Q_;  // 输出误差权重矩阵
    Eigen::MatrixXd R_;  // 控制输入权重矩阵

    // 用于提高效率的预计算 MPC 矩阵
    Eigen::MatrixXd Phi_;      // Phi 矩阵，用于预测 Y_horizon = Phi * x_k + G * U_horizon
    Eigen::MatrixXd G_;        // G 矩阵，用于预测
    Eigen::MatrixXd Q_bar_;    // 块对角扩展输出权重矩阵
    Eigen::MatrixXd R_bar_;    // 块对角扩展输入权重矩阵
    // 定义直接求uk的 Hessian 矩阵
    Eigen::MatrixXd H_qp_;     // QP 问题的预计算 Hessian 矩阵


    //用于增量控制优化
    bool incremental_; // 是否使用增量控制优化
    Eigen::MatrixXd G_incremental_; // 用于增量控制优化的 G 矩阵
    Eigen::VectorXd Phi_incremental_; // 用于增量控制优化的 Phi 向量
    Eigen::MatrixXd S_;        // S 矩阵，用于上一时刻控制输入的影响

    Eigen::MatrixXd H_qp_delta_u_; // 用于增量控制优化的 Hessian 矩阵

    Eigen::VectorXd u_last_;  // 上一时刻的控制输入

    /**
     * @brief 将连续时间状态空间模型离散化为离散时间模型。
     * 使用零阶保持 (ZOH) 方法，通过计算矩阵指数实现。
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
     * @brief 预计算 MPC 的常量矩阵 (Phi, G, Q_bar, R_bar, H_qp)。
     * 此方法在构造函数中调用一次，以避免在 solve 方法中重复计算，
     * 如果系统模型或权重不发生变化。
     */
    void precompute_mpc_matrices();

    /**
     * @brief 构建 MPC 问题的 H 和 f 矩阵，用于 QP 求解器。
     * MPC 目标函数: J = sum_{i=0}^{Np-1} (y_i - r_i)^T Q (y_i - r_i) + u_i^T R u_i
     * 转换为 QP 形式: min 0.5 * U_horizon^T * H * U_horizon + f^T * U_horizon
     * @param current_x 当前状态。
     * @param ref_horizon 预测时域内的参考轨迹。
     * @param H_qp QP 问题中的 H 矩阵引用 (将被赋值为 H_qp_)。
     * @param f_qp QP 问题中的 f 向量引用。
     */
    void build_mpc_qp_matrices(const Eigen::VectorXd& current_x,
                               const Eigen::MatrixXd& ref_horizon,
                               Eigen::MatrixXd& H_qp,
                               Eigen::VectorXd& f_qp) const;

    // 定义求解delta uk的H 和 f 矩阵，用于 QP 求解器。
    /**
     * @brief 构建增量控制优化的 QP 问题矩阵。
     * 增量控制优化目标函数: J = sum_{i=0}^{Nc-1} (delta_u_i)^T R (delta_u_i)
     * 转换为 QP 形式: min 0.5 * delta_U_horizon^T * H_delta_u * delta_U_horizon + f_delta_u^T * delta_U_horizon
     * @param current_x 当前状态。
     * @param ref_horizon 预测时域内的参考轨迹。
     * @param u_last 上一时刻的控制输入。
     * @param H_qp_delta_u QP 问题中的 H 矩阵引用 (将被赋值为 H_qp_delta_u_)。
     * @param f_qp_delta_u QP 问题中的 f 向量引用。
     */
    void build_incremental_qp_matrices(const Eigen::VectorXd& current_x,
                                       const Eigen::MatrixXd& ref_horizon,
                                       const Eigen::VectorXd& u_last,
                                       Eigen::MatrixXd& H_qp_delta_u,
                                       Eigen::VectorXd& f_qp_delta_u) const;

    /**
     * @brief 使用 qpOASES 求解二次规划 (QP) 问题。
     * 此函数集成了 qpOASES 库以找到最优控制序列。
     * @param H_qp QP 问题中的 H 矩阵。
     * @param f_qp QP 问题中的 f 向量。
     * @param nu_horizon 控制输入序列的总维度 (Nc * nu)。
     * @return 优化后的控制输入序列 (Nc*nu x 1) 向量。
     */
    Eigen::VectorXd solve_qp(const Eigen::MatrixXd& H_qp,
                             const Eigen::VectorXd& f_qp,
                             int nu_horizon);
};

#endif // MPC_H
