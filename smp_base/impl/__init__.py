import importlib

# from smp_base.common import dict_search_value_as_key

# required imports
smp_imports_req = {'numpy': 'np', 'smp_base.common': 'smpcom'}

# optional imports
# - these require wrapping by smp to make sure there is a fallback
smp_imports_opt = {'matplotlib.pyplot': 'plt'}

# modules = list(map(__import__, smp_imports_req))

# print('smp_imports_req = {0}'.format(smp_imports_req))
# print('modules = {0}'.format(modules))

class smpcls(object):
    def __init__(self):
        pass

def smp_load_module_as(module, name):
    print('    smp_load_module_as trying to load {0} as {1}'.format(module, name))
    # globals()[name] = importlib.import_module(module)
    # print('loaded {0} as {1}'.format(module, name, name in globals()))
    # return globals()
    return importlib.import_module(module)

def smp_import_modules(_imports):
    """Test code

.. code-block:: python

    from functools import partial
    from smp_base.impl import imp

    gu = partial(globals().update)

    _imports_req = ['np', 'logging', 'argparse', 'smpcom']
    gu(imp(_imports_req))

    _imports_opt = {'matplotlib.pyplot': 'plt', 'sklearn.preprocessing': 'normalize'}
    gu(imp(_imports_opt))

    import logging
    from smp_base.common import get_module_logger

    """
    _modules = {}
    # make sure it's a dict
    if type(_imports) is list:
        _imports = dict(zip(_imports, _imports))

    def dict_search_value_as_key(d, v):
        for k_, v_ in d.values():
            if v == v_: return k_
        return None
        
    # iterate dict and load modules as names
    for module, name in _imports.items():
        if module in smp_imports_req.values():
            module = dict_search_value_as_key(smp_imports_req, module)
            print('smp_import_modules rewriting name {0} to module {1}'.format(smp_imports_req[module], module))
            
        try:
            _modules[name] = smp_load_module_as(module, name)
        except (LookupError, ImportError) as e:
            print('load import {0} failed. Resource {1} was not available w/ {2}'.format(module, name, e))
            _modules[name] = None
            
            # if data not found (not already installed), download it
            # print("Tried to load: '%s'. Resource '%s' was not available and is being downloaded.\n" % (module, name))
            # nltk.download(name)
            # try_load(module, name)
            # return

        assert name in _modules, 'Module {0} failed to load as {1}'.format(module, name)
        
    # # for _ in _modules:
    # exec(compile('global ' + ','.join(list(_imports.values())), filename='impl.imp', mode='exec'), globals(), locals())
    # globals().update(_modules)
    # return {}
    
    return _modules

def imp(_imports):
    return smp_import_modules(_imports)
    
def smpi(module, name=None):
    """Test code

.. code-block:: python

    from smp_base.impl import smp, smpi

    argparse = smpi('argparse')
    logging = smpi('logging')
    np = smpi('numpy')
    normalize = smpi('sklearn.preprocessing', 'normalize')
    get_module_logger = smpi('smp_base.common', 'get_module_logger')
    plot_gennoise = smpi('smp_base.plot_models', 'plot_gennoise')

    logger = get_module_logger(modulename = 'gennoise', loglevel = logging.INFO)
    """

    _module = importlib.import_module(module)
    if name is not None:
        _attr = getattr(_module, name)
        return _attr
    
    return _module
