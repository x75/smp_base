def set_attr_from_dict(obj, dictionary):
    """set object attribute k = v from a dictionary's k, v for all dict items"""
    for k,v in dictionary.items():
        setattr(obj, k, v)

