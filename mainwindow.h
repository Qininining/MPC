#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineSeries>
#include <QChart>

#include "MPC.h" // 包含 MPC 类头文件

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


private:
    Ui::MainWindow *ui;

    QLineSeries *BlueSeries;  // 用于绘制实际系统输出
    QLineSeries *RedSeries;   // 用于绘制参考轨迹
    QLineSeries *GreenSeries; // 用于绘制控制输入（可选）

    QChart *chart;
};
#endif // MAINWINDOW_H
