from collections import OrderedDict

################################################################################
# dynamic code and function mangling
def code_compile_and_run(code = '', gv = {}, lv = {}, return_keys = []):
    """Compile and run the code given in the str 'code',
    optionally including the global and local environments given
    by 'gv', and 'lv'.

    FIXME: check if 'rkey' is in str code and convert to '%s = %s' % (rkey, code) if it is not

    Returns:
    - r(dict): lv if |return_keys| = 0, lv[return_keys] otherwise
    """
    code_ = compile(code, "<string>", "exec")
    exec(code, gv, lv)
    # no keys given, return entire local variables dict
    if len(return_keys) < 1:
        return lv
    # single key given, return just the value of this entry
    elif len(return_keys) == 1:
        if return_keys[0] in lv:
            return lv[return_keys[0]]
    # several keys given, filter local variables dict by these keys and return
    else:
        return dict([(k, lv[k]) for k in return_keys if k in lv])

def get_input(inputs, inkey):
    """smp_graphs.common.get_input

    An smp_graphs bus operation: return the 'val' field of the inputs' item at 'inkey'
    """
    assert type(inputs) in [dict, OrderedDict]
    assert inkey is not None
    assert inkey in inputs
    return inputs[inkey]['val']
