"""smp_base.common

common and frequently required small patterns and util functions

.. TODO:: consolidate with :mod:`smp_graphs.common`
"""

from smp_base.impl import smpi

logging = smpi('logging')
reduce = smpi('functools', 'reduce')

def get_module_logger(modulename = 'experiment', loglevel = logging.INFO):
    """get a logging.logger instance with reasonable defaults

    Create a new logger and configure its name, loglevel, formatter
    and output stream handling.
    1. initialize a logger with name from arg 'modulename'
    2. set loglevel from arg 'loglevel'
    3. configure matching streamhandler
    4. set formatting swag
    5. return the logger
    """
    loglevels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARNING}
    if type(loglevel) is str:
        try:
            loglevel = loglevels[loglevel]
        except:
            loglevel = logging.INFO
            
    if modulename.startswith('smp_graphs'):
        modulename = '.'.join(modulename.split('.')[1:])
        
    if len(modulename) > 20:
        modulename = modulename[-20:]
    
    # create logger
    logger = logging.getLogger(modulename)
    logger.setLevel(loglevel)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)

    # create formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)8s: %(name)20s: %(message)s')
    # formatter = logging.Formatter('{levelname:8}s: %(name)20s: %(message)s')
    # formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    # suppress double log output 
    logger.propagate = False
    return logger

# is defined
def isdefined(obj):
    """isdefined

    obj is a string actually
    """
    return obj in globals() or obj in locals()

# function composition
# https://mathieularose.com/function-composition-in-python/
def compose2(f, g):
    """Compose two functions 'f' and 'g'

    Returns:
     - f(g(x): f return type (object)
    """
    return lambda x: f(g(x))

def compose(*functions):
    return reduce(lambda f, g: lambda *x: f(g(*x)), functions, lambda *x: x)

# def compose(*functions):
#     # def compose2(f, g):
#     #     return lambda x: f(g(x))
#     return reduce(compose2, functions, lambda x: x)

################################################################################
# dictionary helper functions
def dict_to_attr(obj, dictionary):
    set_attr_from_dict(obj, dictionary)
    
def set_attr_from_dict(obj, dictionary):
    """set object attribute 'k' = v from a dictionary's k, v for all dict items

    Transfer configuration dictionaries into an object's member
    namespace (self.__dict__) with :func:`setattr`.
    """
    for k,v in list(dictionary.items()):
        setattr(obj, k, v)

def dict_search_recursive(d, k):
    """smp_base.common.dict_search_recursive

    From smp_graphs.common.dict_search_recursive

    Search for key `k` recursively over nested smp_graph config dicts
    """
    # FIXME: make it generic recursive search over nested graphs and move to smp_base

    # print "#" * 80
    # print "searching k = %s " % (k,),
    if k in d:
        # print "found k = %s, params = %s" % (k, d[k]['params'].keys())
        return d[k]
    else:
        # print "d.keys()", d.keys()
        for k_, v_ in list(d.items()):
            # if v_[
            if 'graph' in v_['params']: #  or v_['params'].has_key('subgraph'):
                # print "k_", k_, "v_", v_['params'].keys()
                return dict_search_recursive(v_['params']['graph'], k)
    # None found
    return None

def dict_search_value_as_key(d, v):
    """smp_base.common.dict_search_value_as_key

    Search for value `v` in dict `d` and return its key `k` if found.
    """
    for k_, v_ in list(d.items()):
        if v == v_: return k_
    return None
        
################################################################################
# main
def main():
    print('smp_base/{0} {1}'.format(__file__, __name__))

if __name__ == '__main__':

    main()
