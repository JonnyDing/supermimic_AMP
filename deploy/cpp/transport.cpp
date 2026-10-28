// Native SDK2 transport matching RoboMimic_Deploy_bjc/deploy_real message contracts.
// No MotionSwitcher calls, no hands, no implicit policy startup.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
namespace py = pybind11;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using Clock = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
template<class T> uint32_t crc(const T& msg) {
    static const auto table = [] { std::array<uint32_t,256> a{};
        for (uint32_t i=0;i<256;i++) { uint32_t v=i<<24; for(int j=0;j<8;j++) v=(v<<1)^((v&0x80000000)?0x04c11db7:0); a[i]=v; } return a; }();
    uint32_t out=0xffffffff;
    const auto* bytes=reinterpret_cast<const unsigned char*>(&msg);
    for(size_t i=0;i<sizeof(T)-4;i+=4) {uint32_t word;std::memcpy(&word,bytes+i,4);
        for(int shift=24;shift>=0;shift-=8) out=(out<<8)^table[((out>>24)^(word>>shift))&255];}
    return out;
}
static void init(int domain,const std::string& net) {
    static bool done=false;static int first;static std::string iface;
    if(done) {if(first!=domain||iface!=net) throw std::runtime_error("One DDS domain/interface per process");return;}
    ChannelFactory::Instance()->Init(domain,net);first=domain;iface=net;done=true;
}
class Bridge {
    std::shared_ptr<ChannelSubscriber<LowState_>> sub;
    std::shared_ptr<ChannelPublisher<LowCmd_>> pub;
    std::mutex mutex,send_mutex;
    LowState_ state{};bool have=false,active=false,stopped=false;
    double received=0,last_send=0,stop_time=0;
    uint64_t count=0,bad=0,duplicates=0,sent=0;
    std::string reason;
    std::array<float,29> damping{};
    std::atomic<bool> running{true};std::thread watchdog;
    void stop_locked(const std::string& why) {if(!stopped) {stopped=true;reason=why;stop_time=now();}}
    LowCmd_ make_locked(const std::vector<float>& q,const std::vector<float>& kp,const std::vector<float>& kd) {
        LowCmd_ cmd{};cmd.mode_pr()=0;cmd.mode_machine()=state.mode_machine();
        for(int i=0;i<29;i++) {auto& m=cmd.motor_cmd()[i];m.mode()=1;m.q()=q[i];m.dq()=0;m.tau()=0;m.kp()=kp[i];m.kd()=kd[i];}
        cmd.crc()=crc(cmd);return cmd;
    }
    LowCmd_ damp_locked() {std::vector<float> z(29,0),kd(damping.begin(),damping.end());return make_locked(z,z,kd);}
    void publish(const LowCmd_& cmd) {pub->Write(cmd);std::lock_guard<std::mutex> g(mutex);sent++;}
public:
    Bridge(const std::string& net,int domain,bool read_only,const std::vector<float>& kd) {
        if(kd.size()!=29) throw std::runtime_error("29 damping gains required");
        for(int i=0;i<29;i++) {if(!std::isfinite(kd[i])||kd[i]<0) throw std::runtime_error("Invalid damping");damping[i]=kd[i];}
        init(domain,net);
        if(!read_only) {pub=std::make_shared<ChannelPublisher<LowCmd_>>("rt/lowcmd");pub->InitChannel();}
        sub=std::make_shared<ChannelSubscriber<LowState_>>("rt/lowstate");
        sub->InitChannel([this](const void* raw) {
            const auto& msg=*static_cast<const LowState_*>(raw);
            bool valid=crc(msg)==msg.crc();
            std::lock_guard<std::mutex> guard(mutex);
            if(!valid) {bad++;return;}
            uint16_t buttons=msg.wireless_remote()[2]|(msg.wireless_remote()[3]<<8);
            if(buttons&(1<<9)) stop_locked("emergency stop (B)");
            if(buttons&(1<<3)) stop_locked("stop (select)");
            if(have && msg.tick()==state.tick()) {duplicates++;return;}
            state=msg;have=true;received=now();count++;
        },0); // Native DDS callback, latest-state overwrite, no Python/GIL/queue.
        watchdog=std::thread([this] {
            while(running) {
                {
                    std::lock_guard<std::mutex> writer(send_mutex);
                    LowCmd_ cmd{};bool emit=false;
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        if(active) {
                            if(now()-received>.1) stop_locked("LowState stale > 100 ms");
                            if(now()-last_send>.1) stop_locked("Python command stale > 100 ms");
                            if(stopped && pub) {if(now()-stop_time<1.) {cmd=damp_locked();emit=true;}else active=false;}
                        }
                    }
                    if(emit)publish(cmd);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }
    py::dict snapshot() {
        std::lock_guard<std::mutex> guard(mutex);py::dict d;
        d["have"]=have;d["stopped"]=stopped;d["reason"]=reason;d["age"]=have?now()-received:1e9;
        d["tick"]=state.tick();d["mode_machine"]=state.mode_machine();
        std::vector<float> q(29),dq(29);for(int i=0;i<29;i++){q[i]=state.motor_state()[i].q();dq[i]=state.motor_state()[i].dq();}
        d["q"]=q;d["dq"]=dq;d["quat"]=state.imu_state().quaternion();d["gyro"]=state.imu_state().gyroscope();
        d["remote"]=py::bytes(reinterpret_cast<const char*>(state.wireless_remote().data()),40);
        d["received"]=count;d["invalid_crc"]=bad;d["duplicates"]=duplicates;d["sent"]=sent;
        return d;
    }
    void send(const std::vector<float>& q,const std::vector<float>& kp,const std::vector<float>& kd,bool damp) {
        for(auto* v:{&q,&kp,&kd}) {if(v->size()!=29) throw std::runtime_error("Expected 29 motor values");for(auto x:*v) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite motor command");}
        std::lock_guard<std::mutex> writer(send_mutex);
        LowCmd_ cmd{};
        {
            std::lock_guard<std::mutex> guard(mutex);
            if(!pub) throw std::runtime_error("Read-only transport cannot send");
            if(!have) throw std::runtime_error("No LowState");
            if(damp) {stop_locked("requested damping");cmd=damp_locked();}
            else {
                if(now()-received>.1) stop_locked("LowState stale > 100 ms");
                if(stopped) throw std::runtime_error(reason);
                cmd=make_locked(q,kp,kd);last_send=now();active=true;
            }
        }
        publish(cmd);
    }

    void close() {
        if(!running.exchange(false)) return;
        if(watchdog.joinable())watchdog.join();
        // Destruction alone never enables or starts a robot command stream.
        if(sub)sub->CloseChannel();if(pub)pub->CloseChannel();
    }
    ~Bridge(){close();}
};
PYBIND11_MODULE(wmrl_transport,m){
    py::class_<Bridge>(m,"Bridge").def(py::init<const std::string&,int,bool,const std::vector<float>&>())
      .def("snapshot",&Bridge::snapshot).def("send",&Bridge::send,py::call_guard<py::gil_scoped_release>())
      .def("close",&Bridge::close,py::call_guard<py::gil_scoped_release>());
}
