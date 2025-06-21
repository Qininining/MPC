#include "MPC.h"
#include <iostream>
#include <stdexcept> // 必需，用于 std::runtime_error 和 std::invalid_argument
#include <cmath>     // 必需，用于 std::fabs（Pade 近似）
#include <limits>    // 必需，用于 std::numeric_limits


// 辅助函数：使用 Pade 近似计算矩阵指数
// 这是一个常用的数值方法，用于计算矩阵 M 的指数 exp(M)。
// 更多详情请参考：
// Charles Van Loan. "Nineteen Ways to Compute the Exponential of a Matrix, Twenty-Five Years Later." SIAM Review, 45(1):3–49, 2003.
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

    // (6,6) Pade 近似的系数：R_6(X) = N_6(X) / D_6(X)
    // N_6(X) = I + (1/2)X + (1/8)X^2 + (1/48)X^3 + (1/384)X^4 + (1/3840)X^5 + (1/46080)X^6
    // D_6(X) = I - (1/2)X + (1/8)X^2 - (1/48)X^3 + (1/384)X^4 - (1/3840)X^5 + (1/46080)X^6
    // 这些系数是从 exp(x) 的泰勒级数展开导出的。
    // 注意：对于一般的 (p,q) Pade 近似，系数更复杂，但对于 exp(X) 的 (k,k) 阶近似，
    // 它们通常简化为这些对称形式。

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
    // 调整初始化列表顺序以匹配成员变量在类中的声明顺序，消除警告 C26432
    : N_(N), dt_(dt), C_(C), Q_(Q), R_(R)
{
    // 获取矩阵维度并进行显式类型转换，消除警告 C4244
    nx_ = static_cast<int>(Ac.rows()); // rows 是行数
    nu_ = static_cast<int>(Bc.cols()); // cols 是列数
    ny_ = static_cast<int>(C.rows());

    try {
        // 验证输入矩阵的维度是否一致
        if (Ac.cols() != nx_ || Bc.rows() != nx_ || C.cols() != nx_ ||
            Q.rows() != ny_ || Q.cols() != ny_ || R.rows() != nu_ || R.cols() != nu_) {
            throw std::invalid_argument("输入矩阵维度不一致。");
        }
        // 验证预测时域 N 和采样时间 dt
        if (N_ <= 0 || dt_ <= std::numeric_limits<double>::epsilon()) { // 检查是否大于一个小的 epsilon
            throw std::invalid_argument("预测时域 N 必须为正，采样时间 dt 必须为正。");
        }

        // 离散化连续时间模型
        discretize_model(Ac, Bc, dt_, Ad_, Bd_);

        // 预计算 MPC 的核心矩阵 (Psi, Gamma, Q_bar, R_bar, H_qp)
        precompute_mpc_matrices();

        // 打印初始化信息
        std::cout << "MPC控制器已初始化:\n";
        std::cout << "  预测时域 (N): " << N_ << "\n";
        std::cout << "  采样时间 (dt): " << dt_ << "\n";
        std::cout << "  状态维度 (nx): " << nx_ << "\n";
        std::cout << "  输入维度 (nu): " << nu_ << "\n";
        std::cout << "  输出维度 (ny): " << ny_ << "\n";
        std::cout << "  Ad:\n" << Ad_ << "\n";
        std::cout << "  Bd:\n" << Bd_ << "\n";
        std::cout << "  C:\n" << C_ << "\n";
        std::cout << "  Q:\n" << Q_ << "\n";
        std::cout << "  R:\n" << R_ << "\n";
        std::cout << "  预计算的 Psi:\n" << Psi_ << "\n";
        std::cout << "  预计算的 Gamma:\n" << Gamma_ << "\n";
        std::cout << "  预计算的 H_qp:\n" << H_qp_ << "\n";


    } catch (const std::exception& e) {
        std::cerr << "[MPC::MPC] 初始化错误: " << e.what() << std::endl;
        throw; // 重新抛出异常，指示构造失败
    }
}

// 将连续时间模型离散化 (零阶保持 ZOH)
// M = [Ac_cont, Bc_cont; 0, 0]
// exp(M * dt) = [Ad_disc, Bd_disc; 0, I]
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

    // 构建增广矩阵 M
    // M = [Ac_cont, Bc_cont;
    //      0_nu_nx, 0_nu_nu]
    // 积分 exp(Ac*tau)*Bc d_tau 从 0 到 dt 得到 Bd。
    // 这通过计算 exp([Ac Bc; 0 0] * dt) 并取其右上角块得到。
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nx_ + nu_, nx_ + nu_);
    M.block(0, 0, nx_, nx_) = Ac_cont;
    M.block(0, nx_, nx_, nu_) = Bc_cont;
    // 最后 nu 行和列保持为零（根据 ZOH 增广）

    // 计算 M * dt 的矩阵指数
    // 明确传递 order 参数，解决“函数不接受 1 个参数”的编译错误
    Eigen::MatrixXd expMdt = matrix_exponential_pade(M * dt, 6); // 明确传入默认的阶数 6

    // 从结果中提取 Ad 和 Bd
    Ad_disc = expMdt.block(0, 0, nx_, nx_);
    Bd_disc = expMdt.block(0, nx_, nx_, nu_);
}

// 预计算MPC的核心矩阵 (Psi, Gamma, Q_bar, R_bar, H_qp)
void MPC::precompute_mpc_matrices() {
    // 预测时域的扩展矩阵
    // Y_horizon = Psi * x_k + Gamma * U_horizon
    // Psi: (N*ny x nx), Gamma: (N*ny x N*nu)

    Psi_ = Eigen::MatrixXd::Zero(N_ * ny_, nx_);
    Gamma_ = Eigen::MatrixXd::Zero(N_ * ny_, N_ * nu_);

    Eigen::MatrixXd Ad_pow_i = Eigen::MatrixXd::Identity(nx_, nx_); // Ad^0 = I
    Eigen::MatrixXd Ad_pow_ij; // 用于存储 Ad^(i-j)

    for (int i = 0; i < N_; ++i) { // 遍历预测步长 (0 到 N-1)
        // Psi 矩阵 (初始状态 x_k 对输出的影响)
        // y_{k+i} = C * Ad^i * x_k + ...
        Psi_.block(i * ny_, 0, ny_, nx_) = C_ * Ad_pow_i;

        // Gamma 矩阵 (未来控制输入对输出的影响)
        // y_{k+i} = ... + C * Ad^(i-j) * Bd * u_j + ...
        Ad_pow_ij = Eigen::MatrixXd::Identity(nx_, nx_); // 为内层循环的 Ad^(i-j) 计算重置

        for (int j = i; j >= 0; --j) { // 逆向遍历预测时域内的控制输入
            // Gamma_ij = C * Ad^(i-j) * Bd
            // 如果 j == i, Ad^(i-j) = Ad^0 = I
            // 如果 j < i, Ad^(i-j) 通过 Ad_pow_ij 乘以 Ad_ 迭代计算。
            // 这比从头重新计算幂次更高效。

            if (j == i) {
                Gamma_.block(i * ny_, j * nu_, ny_, nu_) = C_ * Bd_;
            } else {
                // Ad_pow_ij 从 I 开始。
                // 对于 j = i-1, Ad_pow_ij 变为 Ad_ (Ad^(i-(i-1)) = Ad^1)
                // 对于 j = i-2, Ad_pow_ij 变为 Ad_^2 (Ad^(i-(i-2)) = Ad^2) 等。
                Ad_pow_ij *= Ad_; // 迭代计算 Ad^(i-j)
                Gamma_.block(i * ny_, j * nu_, ny_, nu_) = C_ * Ad_pow_ij * Bd_;
            }
        }
        Ad_pow_i *= Ad_; // 更新 Ad^i 用于下一次迭代 (Ad^(i+1))
    }

    // 构建块对角扩展权重矩阵 (Q_bar 和 R_bar)
    Q_bar_ = Eigen::MatrixXd::Zero(N_ * ny_, N_ * ny_);
    R_bar_ = Eigen::MatrixXd::Zero(N_ * nu_, N_ * nu_);

    for (int i = 0; i < N_; ++i) {
        Q_bar_.block(i * ny_, i * ny_, ny_, ny_) = Q_;
        R_bar_.block(i * nu_, i * nu_, nu_, nu_) = R_;
    }

    // 构建 QP 问题的 H 矩阵
    H_qp_ = Gamma_.transpose() * Q_bar_ * Gamma_ + R_bar_;
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

        // 从优化后的序列中提取控制输入矩阵 (N x nu)
        // u_horizon_optimized 是一个 (N*nu x 1) 的列向量
        // 我们需要将其重塑为 (N x nu) 的矩阵
        Eigen::MatrixXd optimal_U_sequence(N_, nu_);
        for (int i = 0; i < N_; ++i) {
            // segment(offset, length) 从向量中提取子向量
            // transpose() 将列向量转换为行向量，以适应 optimal_U_sequence.row(i) 的赋值
            optimal_U_sequence.row(i) = u_horizon_optimized.segment(i * nu_, nu_).transpose();
        }
        return optimal_U_sequence;

    } catch (const std::exception& e) {
        std::cerr << "[MPC::solve] 发生异常: " << e.what() << std::endl;
        // 返回一个零矩阵作为错误处理的默认或占位符
        return Eigen::MatrixXd::Zero(N_, nu_);
    }
}

// 构建MPC问题的H和f矩阵
void MPC::build_mpc_qp_matrices(const Eigen::VectorXd& current_x,
                                const Eigen::MatrixXd& ref_horizon,
                                Eigen::MatrixXd& H_qp,
                                Eigen::VectorXd& f_qp) const
{
    // H_qp 直接使用预计算的成员变量
    H_qp = H_qp_;

    // 将 ref_horizon 展开成一个长向量 (N*ny x 1)
    Eigen::VectorXd R_vec = Eigen::VectorXd::Zero(N_ * ny_);
    for (int i = 0; i < N_; ++i) {
        R_vec.segment(i * ny_, ny_) = ref_horizon.row(i).transpose();
    }

    // 构建 QP 问题的 f 向量
    // f_qp = Gamma_.transpose() * Q_bar_ * (Psi_ * current_x - R_vec);
    f_qp = Gamma_.transpose() * Q_bar_ * (Psi_ * current_x - R_vec);

    // H_qp 理论上应该是对称正定的。为了数值稳定性，
    // 有时可以添加一个小的正对角矩阵（正则化），
    // 特别是当 H_qp 接近奇异或源自病态问题时。
    // H_qp += 1e-6 * Eigen::MatrixXd::Identity(total_u_dim, total_u_dim);
}


// 使用 qpOASES 的 QP 求解器函数
Eigen::VectorXd MPC::solve_qp(const Eigen::MatrixXd& H_qp,
                              const Eigen::VectorXd& f_qp,
                              int nu_horizon)
{
    std::cout << "  正在调用 qpOASES QP 求解器...\n";

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

    // 初始化并求解 QP 问题
    // 参数: H_qp.data(), f_qp.data(), A_约束矩阵, lb_约束, ub_约束, lb_变量, ub_变量, nWSR
    // 对于无约束 QP，A, lbA, ubA 为 nullptr。
    // 如果需要，可以在此处添加控制输入 (u) 的下界和上界。
    // 示例:
    // Eigen::VectorXd lb = Eigen::VectorXd::Constant(nu_horizon, -10.0); // 示例下界
    // Eigen::VectorXd ub = Eigen::VectorXd::Constant(nu_horizon, 10.0);  // 示例上界
    // qpOASES::returnValue status = qp_solver.init(H_data, f_data, nullptr, nullptr, nullptr, lb.data(), ub.data(), nWSR);
    //
    // 目前，我们使用 nullptr 作为边界（无约束）
    qpOASES::returnValue status = qp_solver.init(H_data, f_data, nullptr, nullptr, nullptr, nullptr, nullptr, nWSR);

    if (status != qpOASES::SUCCESSFUL_RETURN) {
        std::cerr << "  [MPC::solve_qp] qpOASES 求解器初始化失败，状态: " << status << std::endl;
        // 根据应用，您可以抛出异常、返回零向量或使用备用策略。
        throw std::runtime_error("QP 求解器初始化失败。");
    }

    // 获取原始解 (最优控制序列)
    qp_solver.getPrimalSolution(u_horizon_optimized.data());

    std::cout << "  qpOASES QP 求解器完成。\n";
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
