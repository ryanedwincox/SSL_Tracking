#-------------------------------------------------
#
# Project created by QtCreator 2014-12-12T10:54:32
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SSL_Tracking
TEMPLATE = app

LIBS += -lOpenCL
LIBS += `pkg-config opencv --libs`

INCLUDEPATH += '/usr/local/cuda-6.5/include'\
        '/opt/AMDAPP/include'

SOURCES += main.cpp\
    search.cpp \
    holdpoints.cpp

HEADERS += \
    search.h \
    holdpoints.h

OTHER_FILES += cl/findSSL.cl\



