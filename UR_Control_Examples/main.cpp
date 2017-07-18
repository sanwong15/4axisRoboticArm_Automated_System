#include <QCoreApplication>

#include "urcontrolstamp.hpp"

int main(int argc, char *argv[])
{
//    QCoreApplication a(argc, argv);

    URControlStamp urcs;
    urcs.ForceTouch();
//    urcs.Touch();

            return 0;
//    return a.exec();
}
