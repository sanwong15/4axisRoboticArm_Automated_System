#ifndef URCONTROLSERVER_H
#define URCONTROLSERVER_H

#include <QObject>
#include <QTcpServer>
#include <QTcpSocket>

class URControlServer : public QObject
{
    Q_OBJECT
public:
    explicit URControlServer(quint16 port, int timeOutms=30000, QObject *parent = 0);

signals:

public slots:
    void readData();
    void acceptConnection();

public:
    QTcpServer *server;
    QTcpSocket *socket;    
    QByteArray response;
};

#endif // URCONTROLSERVER_H
