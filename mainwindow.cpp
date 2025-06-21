#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    BlueSeries = new QLineSeries();
    BlueSeries->setColor(Qt::blue);
    RedSeries = new QLineSeries();
    RedSeries->setColor(Qt::red);
    GreenSeries = new QLineSeries();
    GreenSeries->setColor(Qt::green);


    chart = new QChart();
    chart->legend()->hide(); // 隐藏图例
    chart->addSeries(BlueSeries);
    chart->createDefaultAxes(); // 创建默认轴
    chart->setTitle("Force (μN)");
    ui->charts->setChart(chart);
    ui->charts->setRenderHint(QPainter::Antialiasing); // 抗锯齿
}

MainWindow::~MainWindow()
{
    delete ui;
}
