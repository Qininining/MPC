QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    MPC.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    MPC.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

INCLUDEPATH += \
    $$PWD/eigen-3.4.0 \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/opencv/x64/vc16/lib/ -lopencv_world4100
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/opencv/x64/vc16/lib/ -lopencv_world4100d
INCLUDEPATH += $$PWD/opencv/include
DEPENDPATH += $$PWD/opencv/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/qpOASES-releases-3.2.2/build/libs/Release/ -lqpOASES
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/qpOASES-releases-3.2.2/build/libs/Debug/ -lqpOASES
INCLUDEPATH += \
    $$PWD/qpOASES-releases-3.2.2/include \
    $$PWD/qpOASES-releases-3.2.2/build/libs/Release \
    $$PWD/qpOASES-releases-3.2.2/build/libs/Debug
DEPENDPATH += \
    $$PWD/qpOASES-releases-3.2.2/include \
    $$PWD/qpOASES-releases-3.2.2/build/libs/Release \
    $$PWD/qpOASES-releases-3.2.2/build/libs/Debug

