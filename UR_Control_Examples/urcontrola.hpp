#ifndef URCONTROLA_HPP
#define URCONTROLA_HPP


#include "urcontrol.hpp"
#include "urcontrolserver.h"
#include <thread>

class URControlA : public URControl
{
public:
    static std::string GetTCPPoseScript(std::string ip, unsigned short port, std::string functionName="fc")
    {
        std::ostringstream oss;
        oss<<"def "<<functionName<<"():\n"
          <<"socket_open(\""<<ip<<"\","<<port<<")\n"
         <<"w=get_actual_tcp_pose()\n"
        <<"socket_send_line(w)\n"
        <<"end\n";
        return oss.str();
    }

    template<typename T>
    static std::string SetTCPPoseResScript(T x, T y, T z, T rx, T ry, T rz, std::string ip, unsigned short port, T a=2.2, T v=5.8, std::string functionName="fc")
    {
        std::ostringstream oss;
        oss<<"def "<<functionName<<"():\n"
          <<"socket_open(\""<<ip<<"\","<<port<<")\n"
         <<SetTCPPoseScript(x,y,z,rx,ry,rz,a,v)
        <<"w=get_actual_tcp_pose()\n"
        <<"socket_send_line(w)\n"
        <<"end\n";
        return oss.str();
    }

    static std::string SampleScript2(std::string ip, unsigned short port, std::string functionName="fc")
    {
        std::ostringstream oss;
        oss<<"def "<<functionName<<"():\n"

             //             <<"set_standard_analog_input_domain(0, 1)\n"
             //             <<"set_standard_analog_input_domain(1, 1)\n"
             //             <<"set_tool_analog_input_domain(0, 1)\n"
             //             <<"set_tool_analog_input_domain(1, 1)\n"
             //             <<"set_analog_outputdomain(0, 0)\n"
             //             <<"set_analog_outputdomain(1, 0)\n"
             //             <<"set_tool_voltage(24)\n"
             //             <<"set_standard_digital_input_action(0, \"default\")\n"
             //             <<"set_standard_digital_input_action(1, \"default\")\n"
             //             <<"set_standard_digital_input_action(2, \"default\")\n"
             //             <<"set_standard_digital_input_action(3, \"default\")\n"
             //             <<"set_standard_digital_input_action(4, \"default\")\n"
             //             <<"set_standard_digital_input_action(5, \"default\")\n"
             //             <<"set_standard_digital_input_action(6, \"default\")\n"
             //             <<"set_standard_digital_input_action(7, \"default\")\n"
             //             <<"set_tool_digital_input_action(0, \"default\")\n"
             //             <<"set_tool_digital_input_action(1, \"default\")\n"


          <<"set_tcp(p[0.0,0.0,0.0,0.0,0.0,0.0])\n"
         <<"set_payload(0.74)\n"
        <<"set_gravity([0.0, 0.0, 9.82])\n"
          //        <<"rtde_set_watchdog(\"speed_slider_mask\", 10.0, \"ignore\")\n"

        <<"socket_open(\""<<ip<<"\","<<port<<")\n"

          //        <<"while (True):\n"
        <<"movel(p[0.271, 0.295, 0.324, " << acos(-1) << ", 0, 0], a=0.02, v=0.1)\n"
        <<"sync()\n"
          //        <<"force_mode( tool_pose(), [0, 0, 1, 0, 0, 0], [0.0, 0.0, 15.0, 0.0, 0.0, 0.0], 2, [0.01, 0.01, 0.015, 0.3490658503988659, 0.3490658503988659, 0.3490658503988659] )\n"
        <<"movel(p[0.271, 0.295, 0.234, " << acos(-1) << ", 0, 0], a=0.02, v=0.1)\n"
          //        <<"end_force_mode()\n"
        <<"stopl(5.0)\n"
          //          <<"end\n"

        <<"movel(p[0.271, 0.295, 0.324, " << acos(-1) << ", 0, 0], a=0.02, v=0.1)\n"

        <<"socket_send_line(tool_pose())\n"
        <<"socket_send_line(\"COMPLETE\")\n"
        <<"end\n";
        return oss.str();
    }

    static void recvTh(std::string &response, unsigned short port)
    {
        URControlServer urcs(port, 20000);
        response=urcs.response.constData();
    }

public:
    std::string ipServer;
    unsigned short portServer;
public:
    URControlA(std::string ipPC="192.168.1.14", std::string ipUR="192.168.1.13", unsigned short portPC=1235, unsigned short portUR=30003, unsigned short modbusPort=502)
        : URControl(ipUR, portUR, modbusPort)
        , ipServer(ipPC)
        , portServer(portPC)
    {

    }

    std::string GetTCPPoseScript(std::string functionName="fc")
    {
        return GetTCPPoseScript(ipServer,portServer,functionName);
    }

    template<typename T>
    bool GetPose(T &x, T &y, T &z, T &rx, T &ry, T &rz)
    {
        std::string res;
        std::thread th(recvTh, std::ref(res), portServer);
        if(!SendScript(GetTCPPoseScript()))
            return false;
        th.join();
        std::cout<<"th end\n"<<std::flush;
        res.erase(0,2);
        char seg;
        std::istringstream iss(res);
        iss>>x>>seg;
        iss>>y>>seg;
        iss>>z>>seg;
        iss>>rx>>seg;
        iss>>ry>>seg;
        iss>>rz>>seg;
        return true;
    }

    template<typename T>
    bool SetPose(T x, T y, T z, T rx, T ry, T rz, T v)
    {
        std::string res;
        std::thread th(recvTh, std::ref(res), portServer);
        if(!SendScript(SetTCPPoseResScript(x,y,z,rx,ry,rz,ipServer,portServer, 4.2, v)))
            return false;
        th.join();
        std::cout<<"th end\n"<<std::flush;
        return true;
    }

    bool LoadAndPlay(std::string urpfile, unsigned short dashboardPort=29999)
    {
        std::string res;
        std::thread th(recvTh, std::ref(res), portServer);
        if(SendScript(ip,dashboardPort,"load /programs/"+urpfile+".urp"))
        {
            if(SendScript(ip,dashboardPort,"play"))
            {
                th.join();
                std::cout<<"th end\n"<<std::flush;
                return true;
            }
        }
        return false;
    }

};

#endif // URCONTROLA_HPP
