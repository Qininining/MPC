#include "MPC.h"
#include <iostream>
#include <stdexcept> // Required for std::runtime_error and std::invalid_argument

// MPC 控制器的构造函数
MPC::MPC(int N,
         const Eigen::MatrixXd& Ac,
         const Eigen::MatrixXd& Bc,
         const Eigen::MatrixXd& C,
         const Eigen::MatrixXd& Q,
         const Eigen::MatrixXd& R,
         double dt)
    : N_(N), C_(C), Q_(Q), R_(R), dt_(dt)
{
    // 获取矩阵维度
    nx_ = Ac.rows();
    nu_ = Bc.cols();
    ny_ = C.rows();

    try {
        // 验证输入矩阵的维度是否一致
        if (Ac.cols() != nx_ || Bc.rows() != nx_ || C.cols() != nx_ ||
            Q.rows() != ny_ || Q.cols() != ny_ || R.rows() != nu_ || R.cols() != nu_) {
            throw std::invalid_argument("Input matrix dimensions are inconsistent.");
        }
        // 验证预测时域 N 和采样时间 dt
        if (N_ <= 0 || dt_ <= 0) {
            throw std::invalid_argument("Prediction horizon N and sampling time dt must be positive.");
        }

        // 离散化连续时间模型
        discretize_model(Ac, Bc, dt_, Ad_, Bd_);

        // 打印初始化信息
        std::cout << "MPCController initialized:\n";
        std::cout << "  Prediction Horizon (N): " << N_ << "\n";
        std::cout << "  Sampling Time (dt): " << dt_ << "\n";
        std::cout << "  State Dim (nx): " << nx_ << "\n";
        std::cout << "  Input Dim (nu): " << nu_ << "\n";
        std::cout << "  Output Dim (ny): " << ny_ << "\n";
        std::cout << "  Ad:\n" << Ad_ << "\n";
        std::cout << "  Bd:\n" << Bd_ << "\n";
        std::cout << "  C:\n" << C_ << "\n";
        std::cout << "  Q:\n" << Q_ << "\n";
        std::cout << "  R:\n" << R_ << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[MPC::MPC] Initialization Error: " << e.what() << std::endl;
        // 在实际应用中，这里可能需要清理资源或设置错误标志
        throw; // 重新抛出异常，指示构造失败
    }
}

// 将连续时间模型离散化 (零阶保持 ZOH)
void MPC::discretize_model(const Eigen::MatrixXd& Ac_cont,
                           const Eigen::MatrixXd& Bc_cont,
                           double dt,
                           Eigen::MatrixXd& Ad_disc,
                           Eigen::MatrixXd& Bd_disc)
{
    // 对于 ZOH 离散化，通常涉及矩阵指数 exp(Ac*dt) 和积分。
    // 这里使用一个简单的近似：Ad = I + Ac*dt, Bd = Bc*dt
    // 这种近似在采样时间 dt 很小的情况下是合理的。
    // 更精确的 ZOH 离散化需要计算矩阵指数的积分，例如通过以下矩阵的指数：
    // M = [Ac_cont, Bc_cont; 0, 0]
    // exp(M * dt) = [Ad_disc, Bd_disc; 0, I]
    // Eigen 库本身不直接提供 exp(MatrixXd) 的功能。如果需要精确的 ZOH 离散化，
    // 可以考虑使用外部库，或者实现数值近似方法（如pade近似、泰勒级数展开等）。
    // 此处仍保留简单近似以维持代码自包含性。

    Ad_disc = Eigen::MatrixXd::Identity(nx_, nx_) + Ac_cont * dt;
    Bd_disc = Bc_cont * dt;
}

// 核心的 MPC 求解方法
Eigen::MatrixXd MPC::solve(const Eigen::VectorXd& current_x,
                           const Eigen::MatrixXd& ref_horizon)
{
    try {
        // 严格的类型和维度检查
        if (current_x.rows() != nx_ || current_x.cols() != 1) {
            throw std::invalid_argument("MPC::solve: current_x must be an (nx x 1) column vector.");
        }
        if (ref_horizon.rows() != N_ || ref_horizon.cols() != ny_) {
            throw std::invalid_argument("MPC::solve: ref_horizon must be an (N x ny) matrix.");
        }

        Eigen::MatrixXd H_qp;
        Eigen::VectorXd f_qp;
        // 构建 QP 问题的 H 和 f 矩阵
        build_mpc_qp_matrices(current_x, ref_horizon, H_qp, f_qp);

        int num_vars_u_horizon = N_ * nu_; // U_horizon 的总变量数

        // 调用 QP 求解器
        // =========================================================================
        // ** Important Note: **
        // At this point, you need to integrate an external Quadratic Programming (QP) solver library.
        // Common open-source C++ QP solvers include:
        // - qpOASES: Optimized for real-time applications, good performance.
        // - OSQP: Another high-performance ADMM-based QP solver.
        // - Eigen::QuadProg (unofficial Eigen module): If you're already using Eigen and have less strict requirements.
        //
        // The following code is a placeholder that simulates a solver returning an optimized control sequence.
        // In a real application, you would call the API of your chosen QP library here.
        // For example, if using qpOASES:
        // qpOASES::QProblem qp_solver(num_vars_u_horizon, 0); // num_variables, num_constraints
        // qpOASES::Options options;
        // options.setTo=qpOASES::QPOASES_SETTING_DEFAULT;
        // options.printLevel = qpOASES::PL_NONE; // Suppress output
        // qp_solver.setOptions(options);
        //
        // // Assuming no bounds or linear constraints, or they are incorporated into H, f
        // int nWSR = 10; // Number of working set recalculations
        // qp_solver.init(H_qp.data(), f_qp.data(), nullptr, nullptr, nullptr, nullptr, nWSR);
        // qp_solver.getPrimalSolution(u_horizon_optimized.data());
        //
        // If there are bound constraints lb <= u <= ub:
        // Eigen::VectorXd lb = Eigen::VectorXd::Constant(num_vars_u_horizon, -5.0); // Example lower bounds
        // Eigen::VectorXd ub = Eigen::VectorXd::Constant(num_vars_u_horizon, 5.0);   // Example upper bounds
        // qp_solver.init(H_qp.data(), f_qp.data(), nullptr, nullptr, lb.data(), ub.data(), nWSR);
        // qp_solver.getPrimalSolution(u_horizon_optimized.data());
        // =========================================================================

        Eigen::VectorXd u_horizon_optimized = solve_qp_placeholder(H_qp, f_qp, num_vars_u_horizon);

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
        std::cerr << "[MPC::solve] Exception occurred: " << e.what() << std::endl;
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
    // 定义 QP 变量 U_horizon = [u_0; u_1; ...; u_{N-1}] (N*nu x 1)
    int total_u_dim = N_ * nu_; // Total number of decision variables in U_horizon

    // Extended matrices for Prediction Horizon
    // Y_horizon = Psi * x_k + Gamma * U_horizon
    // Psi: (N*ny x nx), Gamma: (N*ny x N*nu)

    Eigen::MatrixXd Psi = Eigen::MatrixXd::Zero(N_ * ny_, nx_);
    Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(N_ * ny_, N_ * nu_);

    Eigen::MatrixXd Ad_pow = Eigen::MatrixXd::Identity(nx_, nx_); // Ad^0 = I (current Ad_ is Ad^1)

    for (int i = 0; i < N_; ++i) { // Iterating through prediction steps (0 to N-1)
        // Psi matrix (effect of initial state x_k on outputs)
        // y_{k+i} = C * Ad^i * x_k + ...
        Psi.block(i * ny_, 0, ny_, nx_) = C_ * Ad_pow;

        // Gamma matrix (effect of future control inputs on outputs)
        // y_{k+i} = ... + C * Ad^(i-j) * Bd * u_j + ...
        for (int j = 0; j <= i; ++j) { // Iterating through past control inputs within the horizon
            // Gamma_ij = C * Ad^(i-j) * Bd
            if (j == i) { // Ad^0 term, i.e., direct effect of u_i on y_i
                Gamma.block(i * ny_, j * nu_, ny_, nu_) = C_ * Bd_;
            } else { // Ad^(i-j) term
                Eigen::MatrixXd Ad_inner_pow = Eigen::MatrixXd::Identity(nx_, nx_);
                for (int k = 0; k < (i - j); ++k) { // Calculate Ad^(i-j)
                    Ad_inner_pow *= Ad_;
                }
                Gamma.block(i * ny_, j * nu_, ny_, nu_) = C_ * Ad_inner_pow * Bd_;
            }
        }
        Ad_pow *= Ad_; // Update Ad^i for the next iteration (Ad^(i+1))
    }

    // Unroll ref_horizon into a long vector (N*ny x 1)
    Eigen::VectorXd R_vec = Eigen::VectorXd::Zero(N_ * ny_);
    for (int i = 0; i < N_; ++i) {
        R_vec.segment(i * ny_, ny_) = ref_horizon.row(i).transpose();
    }

    // Build block-diagonal extended weight matrices (Q_bar and R_bar)
    Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(N_ * ny_, N_ * ny_);
    Eigen::MatrixXd R_bar = Eigen::MatrixXd::Zero(N_ * nu_, N_ * nu_);

    for (int i = 0; i < N_; ++i) {
        Q_bar.block(i * ny_, i * ny_, ny_, ny_) = Q_;
        R_bar.block(i * nu_, i * nu_, nu_, nu_) = R_;
    }

    // Construct H and f matrices for the QP problem
    // Objective function: J = 0.5 * U_horizon^T * H * U_horizon + f^T * U_horizon
    H_qp = Gamma.transpose() * Q_bar * Gamma + R_bar;
    f_qp = Gamma.transpose() * Q_bar * (Psi * current_x - R_vec);

    // H_qp should theoretically be symmetric positive definite. For numerical stability,
    // a small positive diagonal matrix is sometimes added (regularization).
    // H_qp += 1e-6 * Eigen::MatrixXd::Identity(total_u_dim, total_u_dim);
}


// QP 求解器占位符函数
Eigen::VectorXd MPC::solve_qp_placeholder(const Eigen::MatrixXd& H_qp,
                                          const Eigen::VectorXd& f_qp,
                                          int nu_horizon)
{
    std::cout << "  (Placeholder) Calling a dummy QP solver...\n";
    // This is a highly simplified example, purely for demonstrating the QP solver
    // interface's inputs and outputs.
    // In an unconstrained scenario, the optimal solution U = -H_inv * f.
    // However, H_qp might not be invertible, or there might be constraints.
    // A real QP solver would handle constraints (e.g., A*U <= b, l <= U <= u).

    // For demonstration, we simply return a zero vector.
    // ** Note: This will NOT produce correct optimization results; it's a placeholder! **

    Eigen::VectorXd u_horizon_optimized = Eigen::VectorXd::Zero(nu_horizon);

    // ** This is where the actual integration of a QP solver would happen **
    // Example with qpOASES (requires qpOASES library):
    qpOASES::QProblem qp_solver(nu_horizon, 0); // num_variables, num_constraints
    qpOASES::Options options;
    // options.setTo=qpOASES::QPOASES_SETTING_DEFAULT;
    // options.printLevel = qpOASES::PL_NONE; // Suppress output
    // qp_solver.setOptions(options);
    //
    // // Assuming no bounds or linear constraints, or they are merged into H, f
    // int nWSR = 10; // Number of working set recalculations (for qpOASES)
    // qp_solver.init(H_qp.data(), f_qp.data(), nullptr, nullptr, nullptr, nullptr, nWSR);
    // qp_solver.getPrimalSolution(u_horizon_optimized.data());
    //
    // If boundary constraints lb <= u <= ub are present:
    // Eigen::VectorXd lb = Eigen::VectorXd::Constant(nu_horizon, -5.0); // Example lower bounds
    // Eigen::VectorXd ub = Eigen::VectorXd::Constant(nu_horizon, 5.0);   // Example upper bounds
    // qp_solver.init(H_qp.data(), f_qp.data(), nullptr, nullptr, lb.data(), ub.data(), nWSR);
    // qp_solver.getPrimalSolution(u_horizon_optimized.data());


    std::cout << "  (Placeholder) QP solver finished. Returning zeros for demo purposes.\n";
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
            throw std::invalid_argument("MPC::simulate_prediction: current_x must be an (nx x 1) column vector.");
        }
        if (u_horizon.rows() != N_ || u_horizon.cols() != nu_) {
            throw std::invalid_argument("MPC::simulate_prediction: u_horizon must be an (N x nu) matrix.");
        }
        if (ref_horizon.rows() != N_ || ref_horizon.cols() != ny_) {
            throw std::invalid_argument("MPC::simulate_prediction: ref_horizon must be an (N x ny) matrix.");
        }

        Eigen::MatrixXd predicted_outputs(N_, ny_); // N rows, ny columns for predicted outputs
        double total_cost = 0.0;
        Eigen::VectorXd x_k_pred = current_x; // Starting state for prediction

        for (int i = 0; i < N_; ++i) {
            Eigen::VectorXd u_k_i = u_horizon.row(i).transpose(); // Current control input (nu x 1)
            Eigen::VectorXd r_k_i = ref_horizon.row(i).transpose(); // Current reference (ny x 1)

            // Predict output y(k+i|k) = C * x(k+i|k)
            Eigen::VectorXd y_k_i = C_ * x_k_pred;
            predicted_outputs.row(i) = y_k_i.transpose(); // Store predicted output as a row

            // Calculate output error cost: (y - r)^T Q (y - r)
            Eigen::VectorXd error_y = y_k_i - r_k_i;
            total_cost += error_y.transpose() * Q_ * error_y;

            // Calculate control input cost: u^T R u
            total_cost += u_k_i.transpose() * R_ * u_k_i;

            // Predict next state: x(k+i+1|k) = Ad * x(k+i|k) + Bd * u(k+i)
            x_k_pred = Ad_ * x_k_pred + Bd_ * u_k_i;
        }
        return {predicted_outputs, total_cost};

    } catch (const std::exception& e) {
        std::cerr << "[MPC::simulate_prediction] Exception occurred: " << e.what() << std::endl;
        // Return a zero matrix and zero cost in case of error
        return {Eigen::MatrixXd::Zero(N_, ny_), 0.0};
    }
}
