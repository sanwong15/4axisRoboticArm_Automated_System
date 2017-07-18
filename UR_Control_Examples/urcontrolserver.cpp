#include "urcontrolserver.h"
#include <QDebug>
#include <iostream>
URControlServer::URControlServer(quint16 port, int timeOutms, QObject *parent) : QObject(parent)
{
    server = new QTcpServer(this);

    // whenever a user connects, it will emit signal
    connect(server, SIGNAL(newConnection()), this, SLOT(acceptConnection()));

    if(server->listen(QHostAddress::Any,port))
    {
       //        qDebug() << "Server started!";
        //        qDebug() << server->serverPort();
                server->waitForNewConnection(timeOutms);
    }
    else
    {
        qDebug() << "Server could not start";
    }
}

void URControlServer::acceptConnection(){
    qDebug() << "connection accepted";
    socket = server->nextPendingConnection();
    //    connect(socket, SIGNAL(readyRead()),this, SLOT(readData()));
    socket->waitForReadyRead();
    readData();
}

void URControlServer::readData()
{
    qDebug() << "try receive";
    response = socket->readAll();
    qDebug(response.constData());
    socket->close();
    //    if(std::string(response.constData()).compare("COMPLETE")==0)
    server->close();
}
