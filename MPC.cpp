#include "MPC.h"
#include <iostream>
#include <stdexcept> // 必需，用于 std::runtime_error 和 std::invalid_argument
#include <cmath>     // 必需，用于 std::fabs（Pade 近似）
#include <limits>    // 必需，用于 std::numeric_limits


Eigen::MatrixXd matrix_exponential_pade(const Eigen::MatrixXd& M, int order) {
    if (M.rows() != M.cols()) {
        throw std::invalid_argument("matrix_exponential_pade: 输入矩阵必须是方阵。");
    }
    if (order < 0) {
        throw std::invalid_argument("matrix_exponential_pade: Pade 近似阶数必须是非负数。");
    }

    int n = static_cast<int>(M.rows()); // 显式转换为 int
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


    double c0 = 1.0;
    double c1 = 1.0 / 2.0;
    double c2 = 1.0 / 8.0;
    double c3 = 1.0 / 48.0;
    double c4 = 1.0 / 384.0;
    double c5 = 1.0 / 3840.0;
    double c6 = 1.0 / 46080.0;

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
MPC::MPC(int N,
         const Eigen::MatrixXd& Ac,
         const Eigen::MatrixXd& Bc,
         const Eigen::MatrixXd& C,
         const Eigen::MatrixXd& Q,
         const Eigen::MatrixXd& R,
         double dt)
    : N_(N), dt_(dt), C_(C), Q_(Q), R_(R)
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
        if (N_ <= 0 || dt_ <= std::numeric_limits<double>::epsilon()) { // 检查是否大于一个小的 epsilon
            throw std::invalid_argument("预测时域 N 必须为正，采样时间 dt 必须为正。");
        }

        discretize_model(Ac, Bc, dt_, Ad_, Bd_);

        precompute_mpc_matrices();

        // 打印初始化信息
        std::cout << "MPC controller initialized:\n";
        std::cout << "  Prediction horizon (N): " << N_ << "\n";
        std::cout << "  Sampling time (dt): " << dt_ << "\n";
        std::cout << "  State dimension (nx): " << nx_ << "\n";
        std::cout << "  Input dimension (nu): " << nu_ << "\n";
        std::cout << "  Output dimension (ny): " << ny_ << "\n";
        std::cout << "  Ad:\n" << Ad_ << "\n";
        std::cout << "  Bd:\n" << Bd_ << "\n";
        std::cout << "  C:\n" << C_ << "\n";
        std::cout << "  Q:\n" << Q_ << "\n";
        std::cout << "  R:\n" << R_ << "\n";
        std::cout << "  Precomputed Psi:\n" << Psi_ << "\n";
        std::cout << "  Precomputed Gamma:\n" << Gamma_ << "\n";
        std::cout << "  Precomputed H_qp:\n" << H_qp_ << "\n";


    } catch (const std::exception& e) {
        std::cerr << "[MPC::MPC] 初始化错误: " << e.what() << std::endl;
        throw; // 重新抛出异常，指示构造失败
    }
}

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

void MPC::precompute_mpc_matrices() {

    Psi_ = Eigen::MatrixXd::Zero(N_ * ny_, nx_);
    Gamma_ = Eigen::MatrixXd::Zero(N_ * ny_, N_ * nu_);

    Eigen::MatrixXd Ad_pow_i = Eigen::MatrixXd::Identity(nx_, nx_); // Ad^0 = I
    Eigen::MatrixXd Ad_pow_ij; // 用于存储 Ad^(i-j)

    // 计算冗余，可优化，但实时计算时不需要计算Gamma_和Psi_，因此可以不优化。
    for (int i = 0; i < N_; ++i) { // 遍历预测步长 (0 到 N-1)

        Psi_.block(i * ny_, 0, ny_, nx_) = C_ * Ad_pow_i;

        Ad_pow_ij = Eigen::MatrixXd::Identity(nx_, nx_); // 为内层循环的 Ad^(i-j) 计算重置

        for (int j = i; j >= 0; --j) { // 逆向遍历预测时域内的控制输入

            if (j == i) {
                Gamma_.block(i * ny_, j * nu_, ny_, nu_) = C_ * Bd_;
            } else {
                Ad_pow_ij *= Ad_; // 迭代计算 Ad^(i-j)
                Gamma_.block(i * ny_, j * nu_, ny_, nu_) = C_ * Ad_pow_ij * Bd_;
            }
        }
        Ad_pow_i *= Ad_; // 更新 Ad^i 用于下一次迭代 (Ad^(i+1))
    }

    Q_bar_ = Eigen::MatrixXd::Zero(N_ * ny_, N_ * ny_);
    R_bar_ = Eigen::MatrixXd::Zero(N_ * nu_, N_ * nu_);

    for (int i = 0; i < N_; ++i) {
        Q_bar_.block(i * ny_, i * ny_, ny_, ny_) = Q_;
        R_bar_.block(i * nu_, i * nu_, nu_, nu_) = R_;
    }

    // 构建 QP 问题的 H 矩阵
    H_qp_ = Gamma_.transpose() * Q_bar_ * Gamma_ + R_bar_;

    if (H_qp_.rows() != N_ * nu_ || H_qp_.cols() != N_ * nu_) {
        throw std::runtime_error("预计算的 H_qp 矩阵维度不正确。");
    }
}


// 核心的 MPC 求解方法
Eigen::MatrixXd MPC::solve(const Eigen::VectorXd& current_x,
                           const Eigen::MatrixXd& ref_horizon)
{
    try {
        // 严格的类型和维度检查
        if (current_x.rows() != nx_ || current_x.cols() != 1) {
            throw std::invalid_argument("MPC::solve: current_x 必须是一个 (nx x 1) 列向量。");
        }
        if (ref_horizon.rows() != N_ || ref_horizon.cols() != ny_) {
            throw std::invalid_argument("MPC::solve: ref_horizon 必须是一个 (N x ny) 矩阵。");
        }

        Eigen::MatrixXd H_qp;
        Eigen::VectorXd f_qp;
        // 构建 QP 问题的 H 和 f 矩阵
        build_mpc_qp_matrices(current_x, ref_horizon, H_qp, f_qp);

        // 调用 QP 求解器
        Eigen::VectorXd u_horizon_optimized = solve_qp(H_qp, f_qp, N_ * nu_);

        Eigen::MatrixXd optimal_U_sequence(N_, nu_);
        for (int i = 0; i < N_; ++i) {
            optimal_U_sequence.row(i) = u_horizon_optimized.segment(i * nu_, nu_).transpose();
        }
        // // 替代方法
        // Eigen::MatrixXd optimal_U_sequence = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        //                          u_horizon_optimized.data(), N_, nu_)
        //                          .array();

        return optimal_U_sequence;

    } catch (const std::exception& e) {
        std::cerr << "[MPC::solve] 发生异常: " << e.what() << std::endl;
        return Eigen::MatrixXd::Zero(N_, nu_);
    }
}

// 构建MPC问题的H和f矩阵
void MPC::build_mpc_qp_matrices(const Eigen::VectorXd& current_x,
                                const Eigen::MatrixXd& ref_horizon,
                                Eigen::MatrixXd& H_qp,
                                Eigen::VectorXd& f_qp) const
{
    H_qp = H_qp_;

    // // 将 ref_horizon (N*ny) 展开成一个长向量 (N*ny x 1)
    // Eigen::VectorXd R_vec = Eigen::VectorXd::Zero(N_ * ny_);
    // for (int i = 0; i < N_; ++i) {
    //     R_vec.segment(i * ny_, ny_) = ref_horizon.row(i).transpose();
    // }

    Eigen::VectorXd R_vec = ref_horizon.reshaped().eval();

    // 构建 QP 问题的 f 向量
    f_qp = Gamma_.transpose() * Q_bar_ * (Psi_ * current_x - R_vec);

}


// 使用 qpOASES 的 QP 求解器函数
Eigen::VectorXd MPC::solve_qp(const Eigen::MatrixXd& H_qp,
                              const Eigen::VectorXd& f_qp,
                              int nu_horizon)
{
    std::cout << "  Calling qpOASES QP solver...\n";

    Eigen::VectorXd u_horizon_optimized = Eigen::VectorXd::Zero(nu_horizon);

    // 确保 H_qp 是对称的 (qpOASES 期望对称 H)
    // H_qp = 0.5 * (H_qp + H_qp.transpose()); // 如果 H_qp 因为数值问题不完全对称，可能需要此行

    // qpOASES 需要原始数据指针
    qpOASES::real_t* H_data = const_cast<qpOASES::real_t*>(H_qp.data()); // qpOASES 期望 qpOASES::real_t*
    qpOASES::real_t* f_data = const_cast<qpOASES::real_t*>(f_qp.data()); // qpOASES 期望 qpOASES::real_t*

    // 创建一个 qpOASES QProblem 实例
    // num_variables: 总的决策变量数
    // num_constraints: 0 (本例中为无约束 QP)
    qpOASES::QProblem qp_solver(nu_horizon, 0);

    // 设置 qpOASES 选项
    qpOASES::Options options;
    // options.setTo=qpOASES::QPOASES_SETTING_DEFAULT; // 使用默认设置
    options.printLevel = qpOASES::PL_LOW; // 抑制详细输出，但显示基本信息
    qp_solver.setOptions(options);

    // 工作集重新计算的最大次数 (nWSR) - 对性能很重要
    // 典型值为 10-100；500 是一个安全的上限。
    int nWSR = 500; // 工作集重新计算的最大次数


    qpOASES::returnValue status = qp_solver.init(H_data, f_data, nullptr, nullptr, nullptr, nullptr, nullptr, nWSR);

    if (status != qpOASES::SUCCESSFUL_RETURN) {
        std::cerr << "  [MPC::solve_qp] qpOASES 求解器初始化失败，状态: " << status << std::endl;
        // 根据应用，您可以抛出异常、返回零向量或使用备用策略。
        throw std::runtime_error("QP 求解器初始化失败。");
    }

    // 获取原始解 (最优控制序列)
    qp_solver.getPrimalSolution(u_horizon_optimized.data());

    std::cout << "  qpOASES QP solver finished.\n";
    return u_horizon_optimized;
}

// 模拟MPC控制下的系统预测轨迹和成本
std::pair<Eigen::MatrixXd, double> MPC::simulate_prediction(
    const Eigen::VectorXd& current_x,
    const Eigen::MatrixXd& u_horizon,
    const Eigen::MatrixXd& ref_horizon) const
{
    try {
        // 严格的类型和维度检查
        if (current_x.rows() != nx_ || current_x.cols() != 1) {
            throw std::invalid_argument("MPC::simulate_prediction: current_x 必须是一个 (nx x 1) 列向量。");
        }
        if (u_horizon.rows() != N_ || u_horizon.cols() != nu_) {
            throw std::invalid_argument("MPC::simulate_prediction: u_horizon 必须是一个 (N x nu) 矩阵。");
        }
        if (ref_horizon.rows() != N_ || ref_horizon.cols() != ny_) {
            throw std::invalid_argument("MPC::simulate_prediction: ref_horizon 必须是一个 (N x ny) 矩阵。");
        }

        Eigen::MatrixXd predicted_outputs(N_, ny_); // N 行, ny 列的预测输出
        double total_cost = 0.0;
        Eigen::VectorXd x_k_pred = current_x; // 预测的起始状态

        for (int i = 0; i < N_; ++i) {
            Eigen::VectorXd u_k_i = u_horizon.row(i).transpose(); // 当前控制输入 (nu x 1)
            Eigen::VectorXd r_k_i = ref_horizon.row(i).transpose(); // 当前参考 (ny x 1)

            // 预测输出 y(k+i|k) = C * x(k+i|k)
            Eigen::VectorXd y_k_i = C_ * x_k_pred;
            predicted_outputs.row(i) = y_k_i.transpose(); // 将预测输出存储为一行

            // 计算输出误差成本: (y - r)^T Q (y - r)
            Eigen::VectorXd error_y = y_k_i - r_k_i;
            total_cost += error_y.transpose() * Q_ * error_y;

            // 计算控制输入成本: u^T R u
            total_cost += u_k_i.transpose() * R_ * u_k_i;

            // 预测下一个状态: x(k+i+1|k) = Ad * x(k+i|k) + Bd * u(k+i)
            x_k_pred = Ad_ * x_k_pred + Bd_ * u_k_i;
        }
        return {predicted_outputs, total_cost};

    } catch (const std::exception& e) {
        std::cerr << "[MPC::simulate_prediction] 发生异常: " << e.what() << std::endl;
        // 错误时返回零矩阵和零成本
        return {Eigen::MatrixXd::Zero(N_, ny_), 0.0};
    }
}

Eigen::VectorXd MPC::predict_y_horizon(const Eigen::VectorXd& x_current,
                                       const Eigen::VectorXd& u_horizon) const
{
    // 检查维度
    if (x_current.rows() != nx_ || x_current.cols() != 1) {
        throw std::invalid_argument("predict_y_horizon: x_current 必须是 (nx x 1) 列向量。");
    }
    if (u_horizon.rows() != N_ * nu_ || u_horizon.cols() != 1) {
        throw std::invalid_argument("predict_y_horizon: u_horizon 必须是 (N*nu x 1) 列向量。");
    }
    // 计算 y_horizon = Psi_ * x_current + Gamma_ * u_horizon
    return Psi_ * x_current + Gamma_ * u_horizon;// N*ny×1
}
