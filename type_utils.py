def ensure(truthy, msg = None, *args):
    if not truthy:
        raise ValueError("ensure truthy error", *([msg] or []), truthy, *args)


def ensure_equal(val, expected, msg = None, *args):
    if val != expected:
        raise ValueError("ensure equal error", *([msg] or []), val, expected, *args)


def ensure_type(item, expected, msg = None, *args):
    try:
        expected = list(expected)
    except:
        expected = [expected]

    for t in expected:
        if isinstance(item, t):
            return

    raise ValueError("ensure type error", *([msg] or []), type(item), expected, item, *args)
