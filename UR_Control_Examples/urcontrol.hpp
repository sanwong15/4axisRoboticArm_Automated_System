#ifndef URCONTROL_HPP
#define URCONTROL_HPP
#include <QTcpSocket>
//#include <QThread>
#include <iostream>
#include <fstream>
#include <sstream>

#include <thread>


class URControl
{
public:
    //protected:
    //https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/modbus-server-16377/
    static void ReadMultipleRegister(QTcpSocket &tcpSocket, unsigned char registerNum, unsigned short startAddress, short *registerValue)
    {
        QByteArray cmd(12,0);
        cmd[0]=0;
        cmd[1]=1;
        cmd[2]=0;
        cmd[3]=0;
        cmd[4]=0;
        cmd[5]=6;
        cmd[6]=0;
        cmd[7]=3;
        cmd[8]=startAddress>>8;
        cmd[9]=startAddress;
        cmd[10]=0;
        cmd[11]=registerNum;
        tcpSocket.write(cmd);
        tcpSocket.waitForBytesWritten(-1);
        tcpSocket.waitForReadyRead(-1);
        QByteArray rs=tcpSocket.readAll();
        uchar *pt=(uchar*)(rs.data())+9;
        uchar *rss=new uchar[2*registerNum];
        std::reverse_copy(pt,pt+2*registerNum,rss);
        short *rsss=(short*)rss;
        std::reverse_copy(rsss,rsss+registerNum,registerValue);
        delete rss;
    }


    static bool LoadScript(std::string &script, std::string filePath)
    {
        std::ifstream ifs(filePath.c_str());
        if(ifs.is_open())
        {
            script.clear();
            std::string str;
            while (std::getline(ifs, str))
                script+=str+"\n";
            ifs.close();
            return true;
        }
        std::cout<<"LoadScript "<<filePath<<" error\n";
        return false;
    }

    static bool Play(std::string ip, unsigned short port=29999, const std::string file="/programs/ddd.urp")
    {
        QTcpSocket socket;
        socket.connectToHost(QString(ip.c_str()),port);
        if(socket.waitForConnected())
        {
            socket.waitForReadyRead(-1);
            {
                QByteArray response = socket.readAll();
                std::cout<<response.constData()<<"\n"<<std::flush;
            }


            std::string script="load "+file;
            std::cout<<script<<"\n"<<std::flush;
            socket.write(script.c_str());
            socket.waitForBytesWritten(-1);
            std::cout<<"wait read\n"<<std::flush;
            //            socket.waitForReadyRead(-1);
            {
                //                QByteArray response = socket.readAll();
                //            std::cout<<response.constData()<<"\n"<<std::flush;
            }


            script="play";
            std::cout<<script<<"\n"<<std::flush;
            socket.write(script.c_str());
            socket.waitForBytesWritten(-1);
            std::cout<<"wait read\n"<<std::flush;
            //            socket.waitForReadyRead(-1);
            {
                //                QByteArray response = socket.readAll();
                //            std::cout<<response.constData()<<"\n"<<std::flush;
            }

            for(int i=0;i<10;i++)
            {
                script="programState";
                std::cout<<script<<"\n"<<std::flush;
                socket.write(script.c_str());
                socket.waitForBytesWritten(-1);
                std::cout<<"wait read\n"<<std::flush;
                socket.waitForReadyRead(-1);
                {
                    QByteArray response = socket.readAll();
                    std::cout<<response.constData()<<"\n"<<std::flush;
                }
            }

            socket.disconnectFromHost();
            return true;
        }

        return false;
    }

    static bool SendScript(std::string ip, unsigned short port, const std::string &script)
    {
        QTcpSocket socket;
        socket.connectToHost(QString(ip.c_str()),port);
        if(socket.waitForConnected())
        {
            socket.write(script.c_str());
            socket.waitForBytesWritten(-1);
            socket.waitForReadyRead(-1);
            socket.disconnectFromHost();
            return true;
        }
        std::cout<<"connectToHost "<<ip<<":"<<port<<" error\n";
        return false;
    }

    std::string ip;
    unsigned short port;
    unsigned short modbusPort;

public:
    template<typename T>
    static std::string SetTCPPoseScript(T x, T y, T z, T rx, T ry, T rz, T a=0.02, T v=0.1)
    {
        std::ostringstream oss;
        oss<<"movel(p["
          <<x<<","
         <<y<<","
        <<z<<","
        <<rx<<","
        <<ry<<","
        <<rz<<"], a="
        <<a<<", v="<<v<<")\n";
        return oss.str();
    }

public:
    URControl(std::string ip0="192.168.10.134", unsigned short port0=30003, unsigned short modbusPort0=502)
        : ip(ip0)
        , port(port0)
        , modbusPort(modbusPort0)
    {

    }

    ~URControl()
    {

    }

    bool GetModbusRegister(unsigned char registerNum, unsigned short startAddress, short *registerValue)
    {
        QTcpSocket tcpSocket;
        tcpSocket.connectToHost(QString(ip.c_str()),modbusPort);
        if(tcpSocket.waitForConnected())
        {
            ReadMultipleRegister(tcpSocket,registerNum, startAddress, registerValue);
            tcpSocket.disconnectFromHost();
            return true;
        }
        return false;
    }

    template<typename T>
    bool GetPos(T &x, T &y, T &z, T &rx, T &ry, T &rz)
    {
        short pos[6];
        if(GetModbusRegister(6,400,pos))
        {
            x=pos[0]/10000.0;
            y=pos[1]/10000.0;
            z=pos[2]/10000.0;
            rx=pos[3]/1000.0;
            ry=pos[4]/1000.0;
            rz=pos[5]/1000.0;
            return true;
        }
        return false;
    }

    bool SendScript(std::string script)
    {
        std::cout<<script<<"\n"<<std::flush;
        return SendScript(ip, port, script);
    }

    bool SendScriptFile(std::string scriptPath)
    {
        std::string script;
        if(LoadScript(script,scriptPath))
            return SendScript(script);
        return false;
    }
};

#endif // URCONTROL_HPP
