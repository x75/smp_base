"""smp_base.lr

.. moduleauthor:: Oswald Berthold

learning rules (lr)
"""

from smp_base.impl import smpi

logging = smpi('logging')
get_module_logger = smpi('smp_base.common', 'get_module_logger')
logger = get_module_logger(modulename = 'lr', loglevel = logging.DEBUG)

np = smpi('numpy')
rlspy = smpi('rlspy')
# if rlspy is None:
#     print("Dont have rlspy, exiting")
#     sys.exit()

def isdefined(obj):
    return obj in globals() or obj in locals()

class smplr(object):
    """smplr

    learning rules
    """
    def __init__(self, *args, **kwargs):
        logger.debug('smplr.init args = {0}, kwargs = {1}'.format(len(args), len(kwargs)))

class smplrRLS(smplr):
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)

        # x0, P0, dim, noise
        for i, param in enumerate(['x0', 'P0', 'dim', 'noise']):
            if param in kwargs:
                setattr(self, param, kwargs[param])
            else:
                setattr(self, param, args[i])
        
        # self.rls_estimator = rlspy.data_matrix.Estimator(np.zeros(shape=(self.dim, 1)) ,(1.0/self.alpha)*np.eye(self.dim))
        # self.rls_estimator = rlspy.data_matrix.Estimator(np.random.uniform(0, 0.0001, size=(self.dim, 1)) , np.eye(self.dim))
        
        # logger.debug('x0 shape {0}'.format(self.x0.shape))
        
        if isdefined('rlspy') and rlspy is not None:
            self.init_rlspy()
            self.update = self.update_rlspy
        else:
            logger.warn('rlspy is not defined, failed import.')
            self.init_smp()
            self.update = self.update_smp

    def init_rlspy(self):
        if self.P0 is None:
            logger.debug("smplrRLS.init: random initialization for RLS setup")
            # self.rls_estimator = rlspy.data_matrix.Estimator(np.random.uniform(0, 0.1, size=(self.dim, 1)) , np.eye(self.dim))
            self.rls_estimator = rlspy.data_matrix.Estimator(
                np.random.uniform(0, 0.01, size=(self.dim, 1)),
                np.eye(self.dim)
            )
        else:
            logger.debug('smplrRLS: taking arguments as initialization for RLS setup')
            # self.wo = wo_init
            self.rls_estimator = rlspy.data_matrix.Estimator(self.x0, self.P0)
            
    def init_smp(self):
        self.rls_estimator = rlsEstimator(self.x0, self.P0)
            
    def update_smp(self, target, r, noise, z, x, **kwargs):
        logger.warn('smplrRLS not implemented')
        self.err = self.rls_estimator.y
        return self.rls_estimator.dx
        
    def update_rlspy(self, target, r, noise, z, x, **kwargs):
        if noise is not None:
            self.noise = noise

        if x is not None and z is not None:
            e = self.mdn_loss(x, r, z, target)
            target = z - e
            
        # print "%s.learnRLS, target.shape = %s" % (self.__class__.__name__, target.shape)
        # self.rls_estimator.update(self.r.T, target.T, self.theta_state)
        self.rls_estimator.single_update(r.T, target.T, self.noise)
        self.err = self.rls_estimator.y
        return self.rls_estimator.dx


class rlsEstimator(object):
    def __init__(self, x0, P0):
        """Returns new RLS estimator."""
        self.x = x0
        self.P = P0
        self.y = np.zeros((1, x0.shape[1]))
        self.dx = np.zeros_like(self.y)
        # identity matrix same size as P for convenience
        self.I = np.identity(len(x0)) 

    def single_update(self, A, b, v):
        logger.warn('rlsEstimator.single_update: implement me')
    
    def update(self, A, b, V):
        logger.warn('rlsEstimator.update: implement me')
