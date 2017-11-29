import os
def increment_path(base):
    if not os.path.exists(base):
        return base
    for idx in range(100):
        path = base + "--{:02}".format(idx)
        if not os.path.exists(path):
            return path
    else:
        raise RuntimeError("100 paths already stored under name {}; use a"
                           " different basepath, or expand"
                           " capacity".format(base))


