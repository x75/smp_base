
import numpy as np

from smp_graphs.common import set_attr_from_dict

################################################################################
# helper funcs
def dtanh(x):
    return 1 - np.tanh(x)**2

def idtanh(x):
    return 1./dtanh(x) # hm?    

def get_cb_dict(func):
    return {"func": func, "cnt": 0}

################################################################################
# main homeostasis, homeokinesis class based on smp_thread_ros
class HK():
    modes = {"hs": 0, "hk": 1, "eh_pi_d": 2}

    # def __init__(self, idim = 1, odim = 1, minlag = 1, maxlag = 2, laglen = 1, mode="hs"):
    def __init__(self, conf):
        set_attr_from_dict(self, conf)
                
        self.isrunning = True
        
        # self.name = "lpzros"
        self.mode = HK.modes[self.mode]
        self.cnt = 0
    
        ############################################################
        # model + meta params
        self.numsen_raw = self.idim
        self.numsen = self.idim
        self.nummot = self.odim
        # buffer size accomodates causal minimum 1 + lag time steps
        self.bufsize = self.maxlag + 1 # 1 + self.robot.lag # 2
        
        # self.creativity = 0.5
        # self.epsA = 0.2
        # self.epsA = 0.02
        # self.epsA = 0.001
        # self.epsC = 0.001
        # self.epsC = 0.001
        # self.epsC = 0.01
        # self.epsC = 0.1
        # self.epsC = 0.3
        # self.epsC = 0.5
        # self.epsC = 0.9
        # self.epsC = 1.0
        # self.epsC = 2.0

        ############################################################
        # forward model
        # self.A = np.eye(self.numsen) * 1.
        self.A  = np.zeros((self.numsen, self.nummot))
        # self.A[range(self.nummot),range(self.nummot)] = 1. # diagonal
        self.b = np.zeros((self.numsen,1))
        # controller
        # self.C  = np.eye(self.nummot) * 0.4
        self.C  = np.zeros((self.nummot, self.numsen))
        self.C[range(self.nummot),range(self.nummot)] = 1 # * 0.4
        # self.C  = np.random.uniform(-1e-2, 1e-2, (self.nummot, self.numsen))
        print "self.C", self.C
        self.h  = np.zeros((self.nummot,1))
        self.g  = np.tanh # sigmoidal activation function
        self.g_ = dtanh # derivative of sigmoidal activation function
        # state
        self.x = np.ones ((self.numsen, self.bufsize))
        self.y = np.zeros((self.nummot, self.bufsize))
        self.z = np.zeros((self.numsen, 1))
        # auxiliary variables
        # self.L     = np.zeros((self.numsen, self.nummot))
        self.v     = np.zeros((self.numsen, 1)) 
        self.v_avg = np.zeros((self.numsen, 1)) 
        self.xsi   = np.zeros((self.numsen, 1))

        self.imu_vec  = np.zeros((3 + 3 + 3, 1))
        self.imu_smooth = 0.8 # coef
        
        # # expansion
        # self.exp_size = self.numsen
        # self.exp_hidden_size = 100
        # self.res = Reservoir(N = self.exp_hidden_size, p = 0.1, g = 1.5, tau = 0.1, input_num = self.numsen_raw, input_scale = 5.0)
        # self.res_wo_expand     = np.random.randint(0, self.exp_hidden_size, self.exp_size)
        # self.res_wo_expand_amp = np.random.uniform(0, 1, (self.exp_size, 1)) * 0.8
        # self.res_wi_expand_amp = np.random.uniform(0, 1, (self.exp_size, self.numsen_raw)) * 1.0
        
    def expansion_random_system(self, x, dim_target = 1):
        # dim_source = x.shape[0]
        # print "x", x.shape
        self.res.execute(x)
        # print "self.res.r", self.res.r.shape
        a = self.res.r[self.res_wo_expand]
        # print "a.shape", a.shape
        b = a * self.res_wo_expand_amp
        # print "b.shape", b.shape
        c = b + np.dot(self.res_wi_expand_amp, x)
        return c
        
    def brain(self, msg):
        """lpz sensors callback: receive sensor values, sos algorithm attached, FloatArray input msg"""
        # FIXME: fix the timing
        # print "msg", self.cnt, msg
        now = 0
        # self.msg_motors.data = []
        self.x = np.roll(self.x, 1, axis=1) # push back past
        self.y = np.roll(self.y, 1, axis=1) # push back past
        # update with new sensor data
        self.x[:,now] = msg[:,-1].copy() # np.array(msg)
        # self.msg_inputs.data = self.x[:,now].flatten().tolist()
        # self.pubs["_lpzros_x"].publish(self.msg_inputs)
        
        # self.x[[0,1],now] = 0.
        # print "msg", msg
        
        # xa = np.array([msg.data]).T
        # self.x[:,[0]] = self.expansion_random_system(xa, dim_target = self.numsen)
        # self.msg_sensor_exp.data = self.x.flatten().tolist()
        # self.pub_sensor_exp.publish(self.msg_sensor_exp)
        
        # compute new motor values
        x_tmp = np.atleast_2d(self.x[:,now]).T + self.v_avg * self.creativity
        # print "x_tmp.shape", x_tmp.shape
        # print self.g(np.dot(self.C, x_tmp) + self.h)
        m1 = np.dot(self.C, x_tmp)
        # print "m1.shape", m1.shape
        t1 = self.g(m1 + self.h).reshape((self.nummot,))
        self.y[:,now] = t1

        self.cnt += 1
        if self.cnt <= 2: return

        # print "x", self.x
        # print "y", self.y
        
        # local variables
        x = np.atleast_2d(self.x[:,self.minlag]).T
        # x = np.atleast_2d(self.x[:,1]).T
        # this is better
        y = np.atleast_2d(self.y[:,self.minlag]).T
        x_fut = np.atleast_2d(self.x[:,now]).T

        # print "x", x.shape, x, x_fut.shape, x_fut
        z = np.dot(self.C, x + self.v_avg * self.creativity) + self.h
        # z = np.dot(self.C, x)
        # print z.shape, x.shape
        # print z - x

        g_prime = dtanh(z) # derivative of g
        g_prime_inv = idtanh(z) # inverse derivative of g

        # print "g_prime", self.cnt, g_prime
        # print "g_prime_inv", self.cnt, g_prime_inv

        # forward prediction error xsi
        # FIXME: include state x in forward model
        self.xsi = x_fut - (np.dot(self.A, y) + self.b)
        # print "xsi =", np.linalg.norm(self.xsi)
        
        # forward model learning
        dA = self.epsA * np.dot(self.xsi, y.T) + (self.A * -0.0003) # * 0.1
        self.A += dA
        db = self.epsA * self.xsi              + (self.b * -0.0001) # * 0.1
        self.b += db

        # print "A", self.cnt, self.A
        # print "b", self.b

        if self.mode == 1: # TLE / homekinesis
            eta = np.dot(np.linalg.pinv(self.A), self.xsi)
            zeta = np.clip(eta * g_prime_inv, -1., 1.)
            # print "eta", self.cnt, eta
            # print "zeta", self.cnt, zeta
            # print "C C^T", np.dot(self.C, self.C.T)
            # mue = np.dot(np.linalg.pinv(np.dot(self.C, self.C.T)), zeta)
            # changed params + noise shape
            lambda_ = np.eye(self.nummot) * np.random.uniform(-0.01, 0.01, (self.nummot, self.nummot))
            mue = np.dot(np.linalg.pinv(np.dot(self.C, self.C.T) + lambda_), zeta)
            self.v = np.clip(np.dot(self.C.T, mue), -1., 1.)
            self.v_avg += (self.v - self.v_avg) * 0.1
            # print "v", self.cnt, self.v
            # print "v_avg", self.cnt, self.v_avg
            EE = 1.0

            # print EE, self.v
            if True: # logarithmic error
                # EE = .1 / (np.sqrt(np.linalg.norm(self.v)) + 0.001)
                EE = .1 / (np.square(np.linalg.norm(self.v)) + 0.001)
            # print "EE", np.linalg.norm(EE)
            # print "eta", eta
            # print "zeta", zeta
            # print "mue", mue
            
            dC = (np.dot(mue, self.v.T) + (np.dot((mue * y * zeta), -2 * x.T))) * EE * self.epsC
            dh = mue * y * zeta * -2 * EE * self.epsC

            # pass
            # dC = np.zeros_like(self.C)
            # dh = np.zeros_like(self.h)
            
        elif self.mode == 0: # homestastic learning
            eta = np.dot(self.A.T, self.xsi)
            # print "eta", self.cnt, eta.shape, eta
            dC = np.dot(eta * g_prime, x.T) * self.epsC
            dh = eta * g_prime * self.epsC
            # print dC, dh
            # self.C +=

        # FIXME: ???
        self.h += np.clip(dh, -.1, .1)
        self.C += np.clip(dC, -.1, .1)
        # self.h += np.clip(dh, -10, 10)
        # self.C += np.clip(dC, -10, 10)

        # print "C", self.C
        # print "h", self.h
        # self.msg_motors.data.append(m[0])
        # self.msg_motors.data.append(m[1])
        # self.msg_motors.data = self.y[:,0].tolist()
        # print("sending msg", msg)
        # self.pub_motors.publish(self.msg_motors)
        # time.sleep(0.1)
        # if self.cnt > 20:
        #     rospy.signal_shutdown("stop")
        #     sys.exit(0)

    def local_hooks(self):
        pass
        
    def prepare_inputs(self):
        return self.robot.prepare_inputs()
    
    def prepare_output(self):
        return self.robot.prepare_output(self.y[:,0])

    def step(self, x):
        """HK (homeokinesis)

        step runs fit forward / fit inverse"""
        if self.isrunning:
            # print "smp_thread: running"
            # call any local computations
            self.local_hooks()

            # prepare input for local conditions
            # inputs = self.prepare_inputs()

            # execute network / controller
            self.brain(x) # inputs)
            
            # local: adjust generic network output to local conditions
            # self.prepare_output()

            # post hooks
            # self.local_post_hooks()
            # write to memory
            # self.memory_pushback()
            
            # publish all state data
            # self.pub_all()
            
            # count
            # self.cnt_main += 1 # incr counter
    
            # print "%s.run isrunning %d" % (self.__class__.__name__, self.isrunning) 
            
            # # check if finished
            # if self.cnt_main == 100000: # self.cfg.len_episode:
            #     # self.savelogs()
            #     self.isrunning = False
            #     # generates problem with batch mode
            #     rospy.signal_shutdown("ending")
            #     print("ending")
            
            # self.rate.sleep()
            # print "self.y", self.y[:,[-1]]
            return self.y[:,[0]]
