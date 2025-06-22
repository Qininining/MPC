#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    BlueSeries = new QLineSeries();
    BlueSeries->setColor(Qt::blue);
    QPen bluePen = BlueSeries->pen(); // Get the current pen
    bluePen.setWidth(2); // Set the desired width (e.g., 2 pixels)
    BlueSeries->setPen(bluePen); // Apply the modified pen

    RedSeries = new QLineSeries();
    RedSeries->setColor(Qt::red);
    QPen redPen = RedSeries->pen();
    redPen.setWidth(2); // You can use different widths for different lines
    RedSeries->setPen(redPen);

    GreenSeries = new QLineSeries();
    GreenSeries->setColor(Qt::green);
    QPen greenPen = GreenSeries->pen();
    greenPen.setWidth(3); // Example: a slightly thicker green line
    GreenSeries->setPen(greenPen);

    BlackSeries = new QLineSeries();
    BlackSeries->setColor(Qt::darkGray);
    QPen blackPen = BlackSeries->pen();
    blackPen.setWidth(1); // Example: a thinner black line
    BlackSeries->setPen(blackPen);


    chart = new QChart();
    chart->legend()->hide(); // 隐藏图例
    chart->addSeries(GreenSeries);
    chart->addSeries(BlackSeries);;
    chart->createDefaultAxes(); // 创建默认轴
    chart->setTitle("y");
    ui->charts->setChart(chart);
    ui->charts->setRenderHint(QPainter::Antialiasing); // 抗锯齿

    chart2 = new QChart();
    chart2->legend()->hide(); // 隐藏图例
    chart2->addSeries(BlueSeries);
    chart2->createDefaultAxes(); // 创建默认轴
    chart2->setTitle("u");
    ui->charts_2->setChart(chart2);
    ui->charts_2->setRenderHint(QPainter::Antialiasing); // 抗锯齿

    chart3 = new QChart();
    chart3->legend()->hide(); // 隐藏图例
    chart3->addSeries(RedSeries);
    chart3->createDefaultAxes(); // 创建默认轴
    chart3->setTitle("e");
    ui->charts_3->setChart(chart3);
    ui->charts_3->setRenderHint(QPainter::Antialiasing); // 抗锯齿


    Eigen::MatrixXd Ac(2, 2);
    Ac << 0.00, 1,
          -50, -5;
    Eigen::MatrixXd Bc(2, 1);
    Bc << 0,
          5;
    Eigen::MatrixXd C(1, 2);
    C << 1, 0;
    Eigen::MatrixXd Q(1, 1);
    Q << 100; // 输出误差权重矩阵
    Eigen::MatrixXd R(1, 1);
    R << 0.01; // 控制输入权重矩阵
    double dt = 0.02; // 采样时间
    int Np = 150; // 预测时域长度
    int Nc = 100; // 控制时域长度

    mpcController = new MPC(Np, Nc, Ac, Bc, C, Q, R, dt); // dt

    int nx = static_cast<int>(Ac.rows()); // rows 是行数
    // int nu = static_cast<int>(Bc.cols()); // cols 是列数
    int ny = static_cast<int>(C.rows());


    Eigen::VectorXd current_x = Eigen::VectorXd::Zero(nx); // 当前状态
    current_x << 0, 0; // 初始状态

    Eigen::MatrixXd ref_horizon = Eigen::MatrixXd::Ones(Np, ny) * 1; // 参考轨迹
    for (int i = 0; i < Nc / 3; ++i) {
        ref_horizon(i, 0) = 1;
    }
    for (int i = Nc / 3; i < Nc * 2 / 3; ++i) {
        ref_horizon(i, 0) = 1;
    }
    for (int i = Nc * 2 / 3; i < Nc-1; ++i) {
        ref_horizon(i, 0) = 1;
    }

    // // 填充线性信号
    // for (int i = 0; i < Nc-1; ++i) {
    //     double t = i * dt;
    //     ref_horizon(i, 0) = 49/Nc * i; // 线性增长
    // }

    // // 填充脉冲信号
    // for (int i = 0; i < Np; ++i) {
    //     double t = i * dt;
    //     if (t < 0.25) {
    //         ref_horizon(i, 0) = 0.0; // 前1秒为0
    //     } else if (t < 0.5) {
    //         ref_horizon(i, 0) = 1.0; // 1秒到2秒为1
    //     } else if (t < 1.0) {
    //         ref_horizon(i, 0) = 0.0; // 2秒到3秒为0
    //     } else if (t < 1.5) {
    //         ref_horizon(i, 0) = -1.0; // 3秒到4秒为-1
    //     } else if (t < 2.0) {
    //         ref_horizon(i, 0) = 0.0; // 4秒到5秒为0
    //     } else if (t < 2.5) {
    //         ref_horizon(i, 0) = 1.0; // 5秒到6秒为1
    //     } else if (t < 3.0) {
    //         ref_horizon(i, 0) = 0.0; // 6秒到7秒为0
    //     } else if (t < 3.5) {
    //         ref_horizon(i, 0) = -1.0; // 7秒到8秒为-1
    //     } else {
    //         ref_horizon(i, 0) = 0.0; // 后续为0
    //     }
    // }





    Eigen::MatrixXd result = mpcController->solve(current_x, ref_horizon);

    Eigen::VectorXd u_predict = result.reshaped().eval();
    // std::cout << "u_predict :\n" << u_predict << std::endl;

    Eigen::VectorXd y_predict = mpcController->predict_y_horizon(current_x, u_predict);
    // std::cout << "y_predict:\n" << y_predict << std::endl;

    Eigen::VectorXd ref_ = ref_horizon.reshaped().eval();
    Eigen::VectorXd error = ref_ - y_predict;
    // std::cout << "error:\n" << error << std::endl;




    // Draw ref_ to BlackSeries
    BlackSeries->clear();
    for (int i = 0; i < Np; ++i) {
        double t = i * dt;
        double y = ref_(i);
        BlackSeries->append(t, y);
    }
    // Draw predicted output to GreenSeries
    GreenSeries->clear();
    for (int i = 0; i < Np; ++i) {
        double t = i * dt;
        double y = y_predict(i);
        GreenSeries->append(t, y);
    }
    // Calculate combined Y-axis range for both series
    double combinedMinY = std::min(ref_.minCoeff(), y_predict.minCoeff());
    double combinedMaxY = std::max(ref_.maxCoeff(), y_predict.maxCoeff());
    // Handle cases where combinedMinY == combinedMaxY to avoid a zero-range Y-axis
    if (std::abs(combinedMaxY - combinedMinY) < 1e-8) {
        combinedMinY -= 1.0; // Extend range by a small arbitrary value
        combinedMaxY += 1.0;
    }
    // Reset and configure axes
    chart->removeAxis(chart->axisX());
    chart->removeAxis(chart->axisY());
    QValueAxis *axisX = new QValueAxis;
    axisX->setRange(0, (Np - 1) * dt);
    axisX->setTitleText("Time (s)");
    QValueAxis *axisY = new QValueAxis;
    // Y-axis self-adaptation with a 10% margin
    double yMargin = 0.1 * std::abs(combinedMaxY - combinedMinY); // Calculate margin based on range
    axisY->setRange(combinedMinY - yMargin, combinedMaxY + yMargin);
    axisY->setTitleText("Output");
    // Attach axes to both series for correct scaling
    chart->addAxis(axisX, Qt::AlignBottom); // Add axis to chart before setting for series
    chart->addAxis(axisY, Qt::AlignLeft);
    GreenSeries->attachAxis(axisX);
    GreenSeries->attachAxis(axisY);
    BlackSeries->attachAxis(axisX);
    BlackSeries->attachAxis(axisY);



    // 绘制预测输出到 BlueSeries
    BlueSeries->clear();
    double Blue_minY = u_predict.minCoeff();
    double Blue_maxY = u_predict.maxCoeff();
    // 若minY==maxY，避免Y轴范围为0
    if (std::abs(Blue_maxY - Blue_minY) < 1e-8) {
        Blue_minY -= 1.0;
        Blue_maxY += 1.0;
    }
    for (int i = 0; i < Nc; ++i) {
        double t = i * dt;
        double y = u_predict(i);
        BlueSeries->append(t, y);
    }
    for(int i = Nc; i < Np; ++i) {
        double t = i * dt;
        double y = u_predict(Nc-1); // 填充剩余部分为最后一个控制输入);
        BlueSeries->append(t, y); // 填充剩余部分为0
    }
    // 重新设置坐标轴范围以适应数据
    chart2->removeAxis(chart2->axisX());
    chart2->removeAxis(chart2->axisY());
    QValueAxis *axisX2 = new QValueAxis;
    axisX2->setRange(0, (Np-1)*dt);
    axisX2->setTitleText("Time (s)");
    QValueAxis *axisY2 = new QValueAxis;
    // Y轴自适应，留出10%边距
    double Blue_yMargin = 0.1 * std::max(std::abs(Blue_minY), std::abs(Blue_maxY));
    axisY2->setRange(Blue_minY - Blue_yMargin, Blue_maxY + Blue_yMargin);
    axisY2->setTitleText("Control Input");
    chart2->setAxisX(axisX2, BlueSeries);
    chart2->setAxisY(axisY2, BlueSeries);

    // 绘制误差到 RedSeries
    RedSeries->clear();
    double Red_minY = error.minCoeff();
    double Red_maxY = error.maxCoeff();
    // 若minY==maxY，避免Y轴范围为0
    if (std::abs(Red_maxY - Red_minY) < 1e-8) {
        Red_minY -= 1.0;
        Red_maxY += 1.0;
    }
    for (int i = 0; i < Np; ++i) {
        double t = i * dt;
        double e = error(i);
        RedSeries->append(t, e);
    }
    // 重新设置坐标轴范围以适应数据
    chart3->removeAxis(chart3->axisX());
    chart3->removeAxis(chart3->axisY());
    QValueAxis *axisX3 = new QValueAxis;
    axisX3->setRange(0, (Np-1)*dt);
    axisX3->setTitleText("Time (s)");
    QValueAxis *axisY3 = new QValueAxis;
    // Y轴自适应，留出10%边距
    double Red_yMargin = 0.1 * std::max(std::abs(Red_minY), std::abs(Red_maxY));
    axisY3->setRange(Red_minY - Red_yMargin, Red_maxY + Red_yMargin);
    axisY3->setTitleText("Error");
    chart3->setAxisX(axisX3, RedSeries);
    chart3->setAxisY(axisY3, RedSeries);





    // 设置图表标题
    chart3->setTitle("Error (e)");
    chart2->setTitle("Control Input (u)");
    chart->setTitle("Predicted Output (y)");



}

MainWindow::~MainWindow()
{
    delete ui;
}
