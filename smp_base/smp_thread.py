"""smp_base: Basic threaded class. We use it for having a main loop side by side with ROS callbacks"""

import threading, signal, time, sys
import numpy as np

import rospy
from std_msgs.msg import Float32, Float32MultiArray
from smp_msgs.msg import reservoir

class smp_thread(threading.Thread):
    def __init__(self, loop_time = 0.1):
        # super init
        threading.Thread.__init__(self)
        self.name = str(self.__class__).split(".")[-1].replace("'>", "")
        signal.signal(signal.SIGINT, self.shutdown_handler)
        # print self.__class__
        # print "self.name", self.name
        # 20170314: remove this from base smp_thread because is is NOT ros
        # rospy.init_node(self.name, anonymous=True)
        # rospy.init_node(self.name, anonymous=False)
        # initialize pub sub
        self.pub_sub()
        # initialize services
        self.srv()

        self.isrunning = True
        self.cnt_main = 0
        self.loop_time = loop_time

    def pub_sub(self):
        self.sub = {}
        self.pub = {}

    def srv(self):
        self.srv = {}

    def shutdown_handler(self, signum, frame):
        print(('smp_thread: Signal handler called with signal', signum))
        self.isrunning = False
        # for sub in self.sub:
        #     self.sub.shutdown()
        #     # print sub.unregister()
        # for pub in self.pub:
        #     self.pub.shutdown()
        # rospy.signal_shutdown("ending")
        # sys.exit(0)
        
    def run(self):
        while self.isrunning:
            time.sleep(self.loop_time)


class smp_thread_ros(smp_thread):
    def __init__(self, loop_time = 0.1, pubs = {}, subs = {}):
        """init args: pubs: dict with topic / [type,], subs: dict with topic / [type, callback]"""
        smp_thread.__init__(self, loop_time = loop_time)
        # now init ros node
        rospy.init_node(self.name, anonymous=True)
        # loop frequency / sampling rate
        self.rate = rospy.Rate(1./self.loop_time)
        # local pub / sub
        self.default_queue_size_pub = 2
        self.default_queue_size_sub = 2
        if len(pubs) == 0 and len(subs) == 0:
            self.pub_sub_local_legacy()
        else:
            self.pub_sub_local(pubs, subs)
        # print "smp_thread_ros pubs", self.pub
        # print "smp_thread_ros subs", self.sub
    def __del__(self):
        del self.pub
        del self.sub

    def pub_sub_local(self, pubs = {}, subs = {}):
        # pass
        for k, v in list(pubs.items()):
            self.pub[k.replace("/", "_")] = rospy.Publisher(k, v[0], queue_size = self.default_queue_size_pub)
        for k, v in list(subs.items()):
            self.sub[k.replace("/", "_")] = rospy.Subscriber(k, v[0], v[1])

    def pub_sub_local_legacy(self):
        self.pub["motor"]         = rospy.Publisher("/motor", Float32MultiArray)
        # learning signals
        # FIXME: change these names to /learner/...
        self.pub["learn_dw"]         = rospy.Publisher("/learner/dw", Float32MultiArray)
        self.pub["learn_w"]          = rospy.Publisher("/learner/w", Float32MultiArray)
        self.pub["learn_perf"]    = rospy.Publisher("/learner/perf", reservoir)
        self.pub["learn_perf_lp"] = rospy.Publisher("/learner/perf_lp", reservoir)
        # learning control
        self.sub["ctrl_eta"]      = rospy.Subscriber("/learner/ctrl/eta", Float32, self.sub_cb_ctrl)
        self.sub["ctrl_theta"]      = rospy.Subscriber("/learner/ctrl/theta", Float32, self.sub_cb_ctrl)
        self.sub["ctrl_target"]   = rospy.Subscriber("/learner/ctrl/target", Float32, self.sub_cb_ctrl)
        # state
        self.pub["learn_x_raw"]   = rospy.Publisher("/learner/x_raw", reservoir)
        self.pub["learn_x"]       = rospy.Publisher("/learner/x", reservoir)
        self.pub["learn_r"]       = rospy.Publisher("/learner/r", reservoir)
        self.pub["learn_y"]       = rospy.Publisher("/learner/y", reservoir)
        
    def sub_cb_ctrl(self, msg):
        """Set learning parameters"""
        topic = msg._connection_header["topic"].split("/")[-1]
        # print "topic", topic
        # print msg
        if topic == "eta":
            # self.eta_init = msg.data
            self.cfg.eta_EH = msg.data
            print(("eta_EH", self.cfg.eta_EH))
        elif topic == "target":
            self.cfg.target = msg.data
            print(("target", self.cfg.target))
        elif topic == "theta":
            self.cfg.theta = msg.data
            self.res.set_theta(self.cfg.theta)
            # self.res.theta = self.cfg.theta
            print(("theta", self.res.theta, self.res.theta_amps))

    def pub_all(self):
        # ros publish
        
        msg = Float32MultiArray()
        msg.data = [np.linalg.norm(self.res.wo[:,i], 2) for i in range(self.cfg.odim)]

        # for i in range(self.cfg.odim):
        #     msg.data.append(np.linalg.norm(self.res.wo[:,i], 2))
        
        # msg.data.append(np.linalg.norm(self.res.wo, 2))
        self.pub["learn_w"].publish(msg)
        self.pub["learn_x_raw"].publish(self.iosm.x_raw)
        self.pub["learn_x"].publish(self.iosm.x)
        self.pub["learn_y"].publish(self.iosm.z)
        self.pub["learn_r"].publish(self.res.r)
        # msg.data = []
        # msg.data.append(LA.norm(dw, 2))
        # self.pub["learn_dw"].publish(msg)
        self.pub["learn_perf"].publish(self.rew.perf)
        self.pub["learn_perf_lp"].publish(self.rew.perf_lp)
        # pass

    def local_hooks(self):
        print("implement: local_hooks()")

    def prepare_inputs(self):
        print("implement: prepare_inputs()")

    def controller(self):
        print("implement: controller()")

    def prepare_output(self, z, zn):
        print(("implement: prepare_output(z, zn)", z, zn))

    def savelogs(self):
        print("implement: save logfiles")

    def local_post_hooks(self):
        pass

    def controller(self):
        print("Implement controller()")
        return (0,0)
                        
    def run(self):
        """Generic run method for learners"""
        print("starting")
        while self.isrunning:
            # print "smp_thread: running"
            # call any local computations
            self.local_hooks()

            # prepare input for local conditions
            self.prepare_inputs()

            # execute network / controller            
            # FIXME: callback count based learning control: washout, learning, testing,
            #        dynamic switching and learning rate modulation
            if self.cnt_main > self.cfg.lag:
                (z, zn) = self.controller()
                # (z, zn) = (0., 0.)
            else:
                z = np.zeros_like(self.iosm.z)
                zn = np.zeros_like(self.iosm.z)
            # print "z/zn.shape", self.iosm.z.shape, self.iosm.zn.shape
            
            # local: adjust generic network output to local conditions
            self.prepare_output(z, zn)

            # post hooks
            self.local_post_hooks()
            # write to memory
            self.memory_pushback()
            
            # publish all state data
            self.pub_all()
            
            # count
            self.cnt_main += 1 # incr counter

            # check if finished
            if self.cnt_main == self.cfg.len_episode:
                # self.savelogs()
                self.isrunning = False
                # generates problem with batch mode
                rospy.signal_shutdown("ending")
                print("ending")
            
            # time.sleep(self.loop_time)
            self.rate.sleep()

# class Terminator(object):
#     def __init__(self):
#         signal.signal(signal.SIGINT, self.handler)
#         # pass

#     def handler(self, signum, frame):
#         print ('class Signal handler called with signal', signum)
#         # al.savelogs()
#         l.isrunning = False
#         rospy.signal_shutdown("ending")
#         sys.exit(0)
#         # raise IOError("Couldn't open device!")

            
if __name__ == "__main__":
    l = smp_thread()

    # t = Terminator()    
    # def handler(signum, frame):
    #     print ('Signal handler called with signal', signum)
    #     # al.savelogs()
    #     l.isrunning = False
    #     rospy.signal_shutdown("ending")
    #     sys.exit(0)
    #     # raise IOError("Couldn't open device!")
    
    # signal.signal(signal.SIGINT, handler)

    l.start()
    while True:
        time.sleep(1)
