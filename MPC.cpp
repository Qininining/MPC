#include "MPC.h"
#include <iostream>
#include <stdexcept> // 必需，用于 std::runtime_error 和 std::invalid_argument
#include <vector>    // 必需，用于 std::vector
#include <cmath>     // 必需，用于 std::fabs（Pade 近似）和 std::log2
#include <limits>    // 必需，用于 std::numeric_limits


// 假设 qpOASES 的头文件已在 MPC.h 中正确包含
#include <qpOASES.hpp>

/**
 * @brief 辅助函数：使用 Pade 近似计算矩阵指数。
 * @param M 要计算指数的方阵。
 * @param order Pade 近似的阶数 (例如，6 表示 (6,6) 阶近似)。
 * @return 矩阵 M 的指数，即 exp(M)。
 */
Eigen::MatrixXd matrix_exponential_pade(const Eigen::MatrixXd& M, int order) {
    if (M.rows() != M.cols()) {
        throw std::invalid_argument("matrix_exponential_pade: 输入矩阵必须是方阵。");
    }
    if (order < 0) {
        throw std::invalid_argument("matrix_exponential_pade: Pade 近似阶数必须是非负数。");
    }

    int n = static_cast<int>(M.rows());
    if (n == 0) {
        return Eigen::MatrixXd::Identity(0, 0);
    }

    // 缩放：减小矩阵范数以提高 Pade 近似精度
    double norm = M.norm();
    int s = 0;
    if (norm > 0.5) { // 如果范数很大，则缩小 M
        s = static_cast<int>(std::floor(std::log2(norm / 0.5))) + 1;
        // 限制 s 以防止过大的缩放因子
        if (s > 10) s = 10; // 避免对于非常大的矩阵或过多平方操作可能导致的溢出
    }
    Eigen::MatrixXd A = M / static_cast<double>(1 << s); // A = M / 2^s

    // 计算 A 的幂次，直到 6 阶，用于 (6,6) Pade 近似
    Eigen::MatrixXd A_sq = A * A;
    Eigen::MatrixXd A_cub = A_sq * A;
    Eigen::MatrixXd A_4 = A_sq * A_sq; // A^4 = (A^2)^2
    Eigen::MatrixXd A_5 = A_4 * A;
    Eigen::MatrixXd A_6 = A_5 * A;

    // 使用标准的 (6,6) 阶 Pade 近似系数
    double c0 = 1.0;
    double c1 = 1.0 / 2.0;
    double c2 = 1.0 / 9.0;
    double c3 = 1.0 / 72.0;
    double c4 = 1.0 / 1008.0;
    double c5 = 1.0 / 30240.0;
    double c6 = 1.0 / 1209600.0;

    Eigen::MatrixXd N_Pade = c0 * Eigen::MatrixXd::Identity(n, n)
                             + c1 * A
                             + c2 * A_sq
                             + c3 * A_cub
                             + c4 * A_4
                             + c5 * A_5
                             + c6 * A_6;

    Eigen::MatrixXd D_Pade = c0 * Eigen::MatrixXd::Identity(n, n)
                             - c1 * A
                             + c2 * A_sq
                             - c3 * A_cub
                             + c4 * A_4
                             - c5 * A_5
                             + c6 * A_6;

    Eigen::MatrixXd exp_M_scaled;
    // 使用小 epsilon 值比较行列式，检查逆矩阵的数值稳定性。
    if (std::fabs(D_Pade.determinant()) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("matrix_exponential_pade: 分母矩阵奇异（或接近奇异），无法计算逆。");
    }
    exp_M_scaled = D_Pade.inverse() * N_Pade;

    // 反缩放：将结果平方 's' 次
    Eigen::MatrixXd result = exp_M_scaled;
    for (int i = 0; i < s; ++i) {
        result = result * result;
    }
    return result;
}


// MPC 控制器的构造函数
MPC::MPC(int N_p,
         int N_c, // 控制时域 Nc
         const Eigen::MatrixXd& Ac,
         const Eigen::MatrixXd& Bc,
         const Eigen::MatrixXd& C,
         const Eigen::MatrixXd& Q,
         const Eigen::MatrixXd& R,
         double dt,
         bool incremental) // 是否使用增量控制优化
    : N_p_(N_p), N_c_(N_c), dt_(dt), C_(C), Q_(Q), R_(R), incremental_(incremental) // 初始化
{
    nx_ = static_cast<int>(Ac.rows()); // rows 是行数
    nu_ = static_cast<int>(Bc.cols()); // cols 是列数
    ny_ = static_cast<int>(C.rows());

    try {
        if (Ac.cols() != nx_ || Bc.rows() != nx_ || C.cols() != nx_ ||
            Q.rows() != ny_ || Q.cols() != ny_ ||
            R.rows() != nu_ || R.cols() != nu_) {
            throw std::invalid_argument("输入矩阵维度不一致。");
        }
        if (N_p_ <= 0 || N_c_ <= 0 || N_c_ > N_p_ || dt_ <= std::numeric_limits<double>::epsilon()) { // 检查 N_c_ 和 dt 的 epsilon
            throw std::invalid_argument("预测时域 Np 必须为正，控制时域 Nc 必须为正且不大于 Np，采样时间 dt 必须为正。");
        }

        discretize_model(Ac, Bc, dt_, Ad_, Bd_);

        precompute_mpc_matrices();

        // 打印初始化信息
        std::cout << "MPC controller initialized:\n";
        std::cout << "  Prediction horizon (Np): " << N_p_ << "\n";
        std::cout << "  Control horizon (Nc): " << N_c_ << "\n"; // 打印 Nc
        std::cout << "  Sampling time (dt): " << dt_ << "\n";
        std::cout << "  State dimension (nx): " << nx_ << "\n";
        std::cout << "  Input dimension (nu): " << nu_ << "\n";
        std::cout << "  Output dimension (ny): " << ny_ << "\n";
        std::cout << "  Ad:\n" << Ad_ << "\n";
        std::cout << "  Bd:\n" << Bd_ << "\n";
        std::cout << "  C:\n" << C_ << "\n";
        std::cout << "  Q:\n" << Q_ << "\n";
        std::cout << "  R:\n" << R_ << "\n";
        std::cout << "  Precomputed Phi:\n" << Phi_ << "\n"; // Psi 重命名为 Phi
        std::cout << "  Precomputed G:\n" << G_ << "\n";   // Gamma 重命名为 G
        std::cout << "  Precomputed H_qp:\n" << H_qp_ << "\n";

        // 如果启用了增量控制优化，打印相关矩阵
        if (incremental_) {
            std::cout << "  Incremental control enabled.\n";
            std::cout << "  G_incremental_:\n" << G_incremental_ << "\n";
            std::cout << "  S_:\n" << S_ << "\n"; // 打印 S 矩阵
            std::cout << "  H_qp_delta_u_:\n" << H_qp_delta_u_ << "\n"; // 打印增量控制 Hessian 矩阵
        } else {
            std::cout << "  Incremental control disabled.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[MPC::MPC] 初始化错误: " << e.what() << std::endl;
        throw; // 重新抛出异常，指示构造失败
    }
}

/**
 * @brief 将连续时间状态空间模型离散化为离散时间模型。
 * 使用零阶保持 (ZOH) 方法，通过计算矩阵指数实现。
 * @param Ac_cont 连续时间 A 矩阵。
 * @param Bc_cont 连续时间 B 矩阵。
 * @param dt 采样时间。
 * @param Ad_disc 离散化后的 A 矩阵引用。
 * @param Bd_disc 离散化后的 B 矩阵引用。
 */
void MPC::discretize_model(const Eigen::MatrixXd& Ac_cont,
                           const Eigen::MatrixXd& Bc_cont,
                           double dt,
                           Eigen::MatrixXd& Ad_disc,
                           Eigen::MatrixXd& Bd_disc)
{
    // 验证维度
    if (Ac_cont.rows() != nx_ || Ac_cont.cols() != nx_ ||
        Bc_cont.rows() != nx_ || Bc_cont.cols() != nu_) {
        throw std::invalid_argument("discretize_model: 输入连续时间矩阵的维度与内部模型维度不匹配。");
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nx_ + nu_, nx_ + nu_);
    M.block(0, 0, nx_, nx_) = Ac_cont;
    M.block(0, nx_, nx_, nu_) = Bc_cont;

    Eigen::MatrixXd expMdt = matrix_exponential_pade(M * dt, 6); // 明确传入默认的阶数 6

    Ad_disc = expMdt.block(0, 0, nx_, nx_);
    Bd_disc = expMdt.block(0, nx_, nx_, nu_);
}

/**
 * @brief 预计算 MPC 的常量矩阵 (Phi, G, Q_bar, R_bar, H_qp)。
 * 此方法在构造函数中调用一次，以避免在 solve 方法中重复计算，
 * 如果系统模型或权重不发生变化。
 */
void MPC::precompute_mpc_matrices() {

    Phi_ = Eigen::MatrixXd::Zero(N_p_ * ny_, nx_);   // Phi 矩阵，维度 Np*p x n
    G_ = Eigen::MatrixXd::Zero(N_p_ * ny_, N_c_ * nu_); // G 矩阵，维度 Np*p x Nc*m

    // 修改: 使用显式乘法预先计算 Ad_ 的幂，以避免使用 .pow() 并提高效率。
    std::vector<Eigen::MatrixXd> Ad_powers(N_p_ + 1);
    Ad_powers[0] = Eigen::MatrixXd::Identity(nx_, nx_); // Ad^0 = I
    for (int k = 1; k <= N_p_; ++k) {
        Ad_powers[k] = Ad_powers[k - 1] * Ad_;
    }

    // 计算 Phi 矩阵和 G 矩阵
    // Phi_：代表初始状态 x(k) 对未来预测输出 Y_p 的影响
    // G_：代表未来控制输入序列 U_c 对未来预测输出 Y_p 的影响

    for (int i = 0; i < N_p_; ++i) { // 遍历预测时域的输出步 (0 到 N_p_-1)，对应 y(k+1|k) 到 y(k+N_p|k)

        // 计算 Phi 矩阵的第 i 行块 (对应 y(k+i+1|k) 中 x(k) 的系数)
        // y(k+i+1|k) 的状态项系数为 C * A^(i+1)
        // 修改: 使用预先计算的幂
        Phi_.block(i * ny_, 0, ny_, nx_) = C_ * Ad_powers[i + 1];

        // 计算 G 矩阵的第 i 行块 (对应 y(k+i+1|k) 中 u 序列的系数)
        for (int j = 0; j < N_c_; ++j) { // 遍历控制时域的输入步 (0 到 N_c_-1)，对应 u(k) 到 u(k+N_c-1)
            if (i >= j) { // 如果当前输出步受当前输入步影响 (未来输入不会影响过去输出)
                // 计算 C * A^(i-j) * B
                // 这是 u(k+j) 对 x(k+i+1|k) 的直接影响，再乘以 C 得到对 y(k+i+1|k) 的影响。
                // 修改: 使用预先计算的幂
                G_.block(i * ny_, j * nu_, ny_, nu_) = C_ * Ad_powers[i - j] * Bd_;
            }
        }

        // 处理控制输入在控制时域 N_c_ 之后保持不变的情况
        // 对于预测时域中超出控制时域的部分 (i >= N_c_)，
        // 即 y(k+i+1|k) 当 i >= N_c_ 时，受 u(k+N_c_-1) 的影响是累积的。
        // 这部分累积效应需要 *添加* 到 G 矩阵的最后一列（j = N_c - 1）的对应块中。
        if (i >= N_c_) {
            Eigen::MatrixXd sum_CA_powers_B = Eigen::MatrixXd::Zero(ny_, nu_);
            // 累加由 u(k+Nc), u(k+Nc+1), ..., u(k+i) 引起的效应，这些输入都等于 u(k+Nc-1)
            // 对应于 C*A^(i-Nc)*B, C*A^(i-Nc-1)*B, ..., C*A^0*B 的和
            for (int s_idx = 0; s_idx <= (i - N_c_); ++s_idx) {
                // 修改: 使用预先计算的幂
                sum_CA_powers_B += C_ * Ad_powers[s_idx] * Bd_;
            }
            // 修正: 从 '=' 改为 '+='。效应是累积的，必须添加到上面循环中已计算的 u(k+Nc-1) 的项上。
            G_.block(i * ny_, (N_c_ - 1) * nu_, ny_, nu_) += sum_CA_powers_B;
        }
    }


    // 构建块对角权重矩阵 Q_bar 和 R_bar
    Q_bar_ = Eigen::MatrixXd::Zero(N_p_ * ny_, N_p_ * ny_); // 输出权重矩阵，维度 Np*p x Np*p
    R_bar_ = Eigen::MatrixXd::Zero(N_c_ * nu_, N_c_ * nu_); // 输入权重矩阵，维度 Nc*m x Nc*m

    for (int i = 0; i < N_p_; ++i) { // 遍历预测时域，填充 Q_bar_
        Q_bar_.block(i * ny_, i * ny_, ny_, ny_) = Q_;
    }

    for (int i = 0; i < N_c_; ++i) { // 遍历控制时域，填充 R_bar_
        R_bar_.block(i * nu_, i * nu_, nu_, nu_) = R_;
    }

    G_incremental_ = Eigen::MatrixXd::Zero(N_p_ * ny_, N_c_ * nu_);
    S_ = Eigen::MatrixXd::Zero(N_p_ * ny_, 1);
    H_qp_ = Eigen::MatrixXd::Zero(N_c_ * nu_, N_c_ * nu_);
    H_qp_delta_u_ = Eigen::MatrixXd::Zero(N_c_ * nu_, N_c_ * nu_);

    u_last_ = Eigen::VectorXd::Zero(nu_);

    if(incremental_) {
        // 增量控制优化
        // 构建下三角矩阵 Id，下三角全为1
        Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(N_c_ * nu_, N_c_ * nu_);
        for (int i = 0; i < N_c_ * nu_; ++i) {
            for (int j = 0; j < N_c_ * nu_; ++j) {
                if (i >= j) { // 下三角部分为1
                    Id(i, j) = 1.0;
                } else {
                    Id(i, j) = 0.0; // 上三角部分为0
                }
            }
        }
        // 增量控制优化的 G 矩阵
        G_incremental_ = G_ * Id; // 将 G 矩阵与下三角矩阵相乘

        Eigen::MatrixXd I_list = Eigen::MatrixXd::Identity(N_c_ * nu_, 1);
        // S 矩阵，表示上一时刻控制输入对预测输出的影响
        S_ = G_ * I_list; // S 矩阵，表示上一时刻控制输入对预测输出的影响

        // 构建 QP 问题的 Hessian 矩阵 H_qp_delta_u
        // H_qp_delta_u = G_incremental_^T * Q_bar_ * G_incremental_ + R_bar_
        H_qp_delta_u_ = G_incremental_.transpose() * Q_bar_ * G_incremental_ + R_bar_; // H_qp = G_incremental^T * Q_bar * G_incremental + R_bar

        // 验证 H_qp 的最终维度
        if (H_qp_delta_u_.rows() != N_c_ * nu_ || H_qp_delta_u_.cols() != N_c_ * nu_) {
            throw std::runtime_error("预计算的 H_qp 矩阵维度不正确。");
        }

    } else {
        // 构建 QP 问题的 Hessian 矩阵 H_qp
        // H_qp = G_^T * Q_bar_ * G_ + R_bar_
        H_qp_ = G_.transpose() * Q_bar_ * G_ + R_bar_; // H_qp = G^T * Q_bar * G + R_bar
            // 注意：QP 目标函数是 0.5 * X^T * H * X + g^T * X。
            // 而一些推导中的 J 包含 1/2，导致 H = 2 * (...)。
            // qpOASES 使用 0.5 * x'Hx + g'x 的形式，因此 H_qp_ 不应包含因子 2。

        // 验证 H_qp 的最终维度
        if (H_qp_.rows() != N_c_ * nu_ || H_qp_.cols() != N_c_ * nu_) {
            throw std::runtime_error("预计算的 H_qp 矩阵维度不正确。");
        }
    }
}

/**
     * @brief solve 方法，根据incremental_，求解 MPC 问题，预测并返回最优控制输入序列。
     * @param ref_horizon 预测时域内的参考轨迹 (Np x ny)。
     * @param current_x 当前状态向量 (nx x 1)。
     * @param u_last 上一时刻的控制输入 (nu x 1)。如果为空，则使用内部存储的值。
     * @return
     */
Eigen::VectorXd MPC::solve(const Eigen::MatrixXd& ref_horizon,
                      const Eigen::VectorXd& current_x,
                      const Eigen::VectorXd& u_last)
{
    if (incremental_) {
        return solve_incremental(ref_horizon, current_x, u_last);
    } else {
        return solve_direct(ref_horizon, current_x);
    }
}

/**
 * @brief 求解 MPC 问题，预测并返回最优控制输入序列。
 * 此方法将构建二次规划 (QP) 问题并尝试求解。
 * @param current_x 当前状态向量 (nx x 1)。
 * @param ref_horizon 预测时域内的参考轨迹 (Np x ny)。
 * @return 最优控制输入序列 (Nc x nu)。在实际中通常只取第一个控制量。
 */
Eigen::VectorXd MPC::solve_direct(const Eigen::MatrixXd& ref_horizon,
                           const Eigen::VectorXd& current_x)
{
    try {
        // 严格的类型和维度检查
        if (current_x.rows() != nx_ || current_x.cols() != 1) {
            throw std::invalid_argument("MPC::solve: current_x must be an (nx x 1) column vector.");
        }
        // ref_horizon 是 N_p x ny
        if (ref_horizon.rows() != N_p_ || ref_horizon.cols() != ny_) { // 检查预测时域 N_p_ 和输出维度 ny_
            throw std::invalid_argument("MPC::solve: ref_horizon must be an (Np x ny) matrix.");
        }

        Eigen::MatrixXd H_qp;
        Eigen::VectorXd f_qp;
        // 构建 QP 问题的 H 和 f 矩阵
        build_mpc_qp_matrices(current_x, ref_horizon, H_qp, f_qp);

        // 调用 QP 求解器
        Eigen::VectorXd u_horizon_optimized = solve_qp(H_qp, f_qp, N_c_ * nu_); // 传递控制时域的总输入变量数 N_c_ * nu_

        // ####################################################################################################################################
        // 打印最终的成本
        Eigen::MatrixXd J = u_horizon_optimized.transpose() * H_qp * u_horizon_optimized / 2.0 + f_qp.transpose() * u_horizon_optimized;
        std::cout << "MPC Cost: " << J(0, 0) << std::endl;

        // // 将优化后的控制输入向量重塑为 (N_c x nu) 矩阵
        // Eigen::MatrixXd optimal_U_sequence(N_c_, nu_);
        // for (int i = 0; i < N_c_; ++i) {
        //     optimal_U_sequence.row(i) = u_horizon_optimized.segment(i * nu_, nu_).transpose();
        // }
        // ####################################################################################################################################

        // 更新内部存储的上一时刻控制输入（使用第一个优化控制）
        // u_last_ = u_horizon_optimized.row(0).transpose();
        u_last_ = u_horizon_optimized.head(nu_); // 只取第一个控制输入，更新内部存储的上一时刻控制输入

        return u_horizon_optimized;

    } catch (const std::exception& e) {
        std::cerr << "[MPC::solve] 发生异常: " << e.what() << std::endl;
        return Eigen::MatrixXd::Zero(N_c_, nu_); // 发生错误时返回零矩阵，保持正确维度 N_c_ x nu_
    }
}

Eigen::VectorXd MPC::solve_incremental(const Eigen::MatrixXd& ref_horizon,
                                       const Eigen::VectorXd& current_x,
                                       const Eigen::VectorXd& u_last)
{
    try {
        // 严格的类型和维度检查
        if (current_x.rows() != nx_ || current_x.cols() != 1) {
            throw std::invalid_argument("MPC::solve: current_x must be an (nx x 1) column vector.");
        }
        // ref_horizon 是 N_p x ny
        if (ref_horizon.rows() != N_p_ || ref_horizon.cols() != ny_) { // 检查预测时域 N_p_ 和输出维度 ny_
            throw std::invalid_argument("MPC::solve: ref_horizon must be an (Np x ny) matrix.");
        }

        // 使用提供的 u_prev 或内部存储的值
        Eigen::VectorXd u_last_to_use;
        if (u_last.size() > 0) {
            if (u_last.rows() != nu_ || u_last.cols() != 1) {
                throw std::invalid_argument("MPC::solve: u_prev must be an (nu x 1) column vector.");
            }
            u_last_to_use = u_last;
        } else {
            u_last_to_use =  Eigen::VectorXd::Zero(nu_);
        }

        Eigen::MatrixXd H_qp_delta_u;
        Eigen::VectorXd f_qp_delta_u;
        // 构建 QP 问题的 H 和 f 矩阵
        build_incremental_qp_matrices(current_x, ref_horizon, u_last_to_use, H_qp_delta_u, f_qp_delta_u);

        // 调用 QP 求解器
        Eigen::VectorXd delta_u_optimized = solve_qp(H_qp_delta_u, f_qp_delta_u, N_c_ * nu_); // 传递控制时域的总输入变量数 N_c_ * nu_

        // ####################################################################################################################################
        // 打印最终的成本
        Eigen::MatrixXd J = delta_u_optimized.transpose() * H_qp_delta_u * delta_u_optimized / 2.0 + f_qp_delta_u.transpose() * delta_u_optimized;
        std::cout << "MPC Cost: " << J(0, 0) << std::endl;

        // // 将优化后的控制输入向量重塑为 (N_c x nu) 矩阵
        // Eigen::MatrixXd optimal_U_sequence(N_c_, nu_);
        // for (int i = 0; i < N_c_; ++i) {
        //     optimal_U_sequence.row(i) = delta_u_optimized.segment(i * nu_, nu_).transpose();
        // }
        // ####################################################################################################################################

        return delta_u_optimized;


    } catch (const std::exception& e) {
        std::cerr << "[MPC::solve_incremental] 发生异常: " << e.what() << std::endl;
        return Eigen::MatrixXd::Zero(N_c_, nu_); // 发生错误时返回零矩阵，保持正确维度 N_c_ x nu_
    }
}

/**
 * @brief 构建 MPC 问题的 H 和 f 矩阵，用于 QP 求解器。
 * MPC 目标函数: J = sum_{i=0}^{Np-1} (y_i - r_i)^T Q (y_i - r_i) + u_i^T R u_i
 * 转换为 QP 形式: min 0.5 * U_horizon^T * H * U_horizon + f^T * U_horizon
 * @param current_x 当前状态。
 * @param ref_horizon 预测时域内的参考轨迹。
 * @param H_qp QP 问题中的 H 矩阵引用 (将被赋值为 H_qp_)。
 * @param f_qp QP 问题中的 f 向量引用。
 */
void MPC::build_mpc_qp_matrices(const Eigen::VectorXd& current_x,
                                const Eigen::MatrixXd& ref_horizon,
                                Eigen::MatrixXd& H_qp,
                                Eigen::VectorXd& f_qp) const
{
    // H 矩阵已经预计算好，直接赋值
    H_qp = H_qp_;

    // 将 ref_horizon (Np x ny) 展开成一个长向量 (Np*ny x 1)
    Eigen::VectorXd R_vec = Eigen::VectorXd::Zero(N_p_ * ny_);
    for (int i = 0; i < N_p_; ++i) {
        R_vec.segment(i * ny_, ny_) = ref_horizon.row(i).transpose();
    }

    // 构建 QP 问题的 f 向量
    // f_qp = G_^T * Q_bar_ * (Phi_ * current_x - R_vec);
    Eigen::VectorXd error_free = Phi_ * current_x - R_vec;
    f_qp = G_.transpose() * Q_bar_ * error_free;
}


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
void MPC::build_incremental_qp_matrices(const Eigen::VectorXd& current_x,
                                   const Eigen::MatrixXd& ref_horizon,
                                   const Eigen::VectorXd& u_last,
                                   Eigen::MatrixXd& H_qp_delta_u,
                                   Eigen::VectorXd& f_qp_delta_u) const
{
    // H_qp_delta_u 已经预计算好，直接赋值
    H_qp_delta_u = H_qp_delta_u_;

    // 将 ref_horizon (Np x ny) 展开成一个长向量 (Np*ny x 1)
    Eigen::VectorXd R_vec = Eigen::VectorXd::Zero(N_p_ * ny_);
    for (int i = 0; i < N_p_; ++i) {
        R_vec.segment(i * ny_, ny_) = ref_horizon.row(i).transpose();
    }

    // 构建增量控制优化的 f 向量
    Eigen::VectorXd error_free = Phi_ * current_x + S_ * u_last - R_vec;
    f_qp_delta_u = G_incremental_.transpose() * Q_bar_ * error_free;

}

/**
 * @brief 使用 qpOASES 求解二次规划 (QP) 问题。
 * 此函数集成了 qpOASES 库以找到最优控制序列。
 * @param H_qp QP 问题中的 H 矩阵。
 * @param f_qp QP 问题中的 f 向量。
 * @param nu_horizon 控制输入序列的总维度 (Nc * nu)。
 * @return 优化后的控制输入序列 (Nc*nu x 1) 向量。
 */
Eigen::VectorXd MPC::solve_qp(const Eigen::MatrixXd& H_qp,
                              const Eigen::VectorXd& f_qp,
                              int nu_horizon)
{
    std::cout << "  Calling qpOASES QP solver...\n";

    // 定义求解结果
    Eigen::VectorXd result_optimized(nu_horizon);

    // qpOASES 需要 C-style 数组 (原始数据指针)
    // Eigen 默认是列主序，qpOASES 的 H 需要是列主序的数组
    qpOASES::real_t* H_data = const_cast<qpOASES::real_t*>(H_qp.data());
    qpOASES::real_t* f_data = const_cast<qpOASES::real_t*>(f_qp.data());

    // 创建一个 qpOASES QProblem 实例
    // num_variables: 总的决策变量数
    // num_constraints: 0 (本例中为无约束 QP)
    qpOASES::QProblem qp_solver(nu_horizon, 0);

    // 设置 qpOASES 选项
    qpOASES::Options options;
    options.printLevel = qpOASES::PL_LOW; // 抑制详细输出，只显示基本信息
    qp_solver.setOptions(options);

    // 工作集重新计算的最大次数 (nWSR)
    int nWSR = 500;


    qpOASES::returnValue status = qp_solver.init(H_data, f_data, nullptr, nullptr, nullptr, nullptr, nullptr, nWSR);

    if (status != qpOASES::SUCCESSFUL_RETURN) {
        // std::cerr << "  [MPC::solve_qp] qpOASES solver initialization failed, status: " << qpOASES::getSimpleStatusMessage(status) << std::endl;
        throw std::runtime_error("QP 求解器初始化失败。");
    }

    // 获取原始解 (最优控制序列)
    qpOASES::real_t* sol_data = result_optimized.data();
    qp_solver.getPrimalSolution(sol_data);

    std::cout << "  qpOASES QP solver finished.\n";
    return result_optimized;
}

/**
 * @brief 模拟 MPC 控制下的系统预测轨迹和成本。
 * 这是一个辅助函数，用于验证给定控制输入序列的效果，不进行优化。
 * @param current_x 初始状态。
 * @param u_horizon 要应用的控制输入序列 (Nc x nu)。
 * @param ref_horizon 参考轨迹 (Np x ny)。
 * @return 包含预测输出轨迹 (MatrixXd) 和总成本 (double) 的 pair。
 */
std::pair<Eigen::MatrixXd, double> MPC::simulate_prediction(
    const Eigen::VectorXd& current_x,
    const Eigen::MatrixXd& u_horizon,
    const Eigen::MatrixXd& ref_horizon) const
{
    try {
        // 严格的类型和维度检查
        if (current_x.rows() != nx_ || current_x.cols() != 1) {
            throw std::invalid_argument("MPC::simulate_prediction: current_x must be an (nx x 1) column vector.");
        }
        if (u_horizon.rows() != N_c_ || u_horizon.cols() != nu_) { // u_horizon 维度为 Nc x nu
            throw std::invalid_argument("MPC::simulate_prediction: u_horizon must be an (Nc x nu) matrix.");
        }
        if (ref_horizon.rows() != N_p_ || ref_horizon.cols() != ny_) { // ref_horizon 维度为 Np x ny
            throw std::invalid_argument("MPC::simulate_prediction: ref_horizon must be an (Np x ny) matrix.");
        }

        Eigen::MatrixXd predicted_outputs(N_p_, ny_); // 预测输出矩阵，维度 Np x ny
        double total_cost = 0.0;
        Eigen::VectorXd x_k_pred = current_x; // 预测的起始状态

        for (int i = 0; i < N_p_; ++i) { // 遍历预测时域 N_p_
            Eigen::VectorXd u_k_i;
            if (i < N_c_) { // 在控制时域内，使用实际优化后的输入
                u_k_i = u_horizon.row(i).transpose(); // 当前控制输入 (nu x 1)
            } else { // 超出控制时域的步长，使用控制时域的最后一个输入
                u_k_i = u_horizon.row(N_c_ - 1).transpose();
            }

            Eigen::VectorXd r_k_i = ref_horizon.row(i).transpose(); // 当前参考 (ny x 1)

            // 预测输出 y(k+i|k) = C * x(k+i|k)
            Eigen::VectorXd y_k_i = C_ * x_k_pred;
            predicted_outputs.row(i) = y_k_i.transpose(); // 将预测输出存储为一行

            // 计算输出误差成本: (y - r)^T Q (y - r)
            Eigen::VectorXd error_y = y_k_i - r_k_i;
            total_cost += error_y.transpose() * Q_ * error_y;

            // 计算控制输入成本: u^T R u
            if (i < N_c_) { // 仅惩罚控制时域内的输入
                total_cost += u_k_i.transpose() * R_ * u_k_i;
            }

            // 预测下一个状态: x(k+i+1|k) = Ad * x(k+i|k) + Bd * u(k+i)
            x_k_pred = Ad_ * x_k_pred + Bd_ * u_k_i;
        }
        return {predicted_outputs, total_cost};

    } catch (const std::exception& e) {
        std::cerr << "[MPC::simulate_prediction] 发生异常: " << e.what() << std::endl;
        // 发生错误时返回零矩阵和零成本
        return {Eigen::MatrixXd::Zero(N_p_, ny_), 0.0};
    }
}

/**
 * @brief 利用 Phi_ 和 G_ 计算预测输出序列。
 * @param x_current 当前状态 (nx x 1)。
 * @param u_horizon 控制输入序列 (Nc*nu x 1)。
 * @return 预测输出序列 (Np*ny x 1)。
 */
Eigen::VectorXd MPC::predict_y_horizon(const Eigen::VectorXd& x_current,
                                       const Eigen::VectorXd& u_horizon,
                                       const Eigen::VectorXd& u_last) const
{
    // 检查维度
    if (x_current.rows() != nx_ || x_current.cols() != 1) {
        throw std::invalid_argument("predict_y_horizon: x_current 必须是 (nx x 1) 列向量。");
    }
    // u_horizon 维度为 (Nc*nu x 1)
    if (u_horizon.rows() != N_c_ * nu_ || u_horizon.cols() != 1) { // 检查 N_c_
        throw std::invalid_argument("predict_y_horizon: u_horizon 必须是 (Nc*nu x 1) 列向量。");
    }

    if(incremental_) {
        // 使用提供的 u_prev 或内部存储的值
        Eigen::VectorXd u_last_to_use;
        if (u_last.size() > 0) {
            if (u_last.rows() != nu_ || u_last.cols() != 1) {
                throw std::invalid_argument("MPC::solve: u_prev must be an (nu x 1) column vector.");
            }
            u_last_to_use = u_last;
        } else {
            u_last_to_use =  Eigen::VectorXd::Zero(nu_);
        }
        // 如果启用了增量控制优化，使用 G_incremental_ 和 S_ 计算预测输出
        return Phi_ * x_current + S_ * u_last_to_use + G_incremental_ * u_horizon;
    }else{
        // 计算 y_horizon = Phi_ * x_current + G_ * u_horizon
        return Phi_ * x_current + G_ * u_horizon;// 结果维度为 (Np*ny x 1)
    }

}
