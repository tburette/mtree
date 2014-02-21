def count(d):
    """decorator counting calls
    """
    def counting_d(*args, **kwargs):
        counting_d.count += 1
        return d(*args, **kwargs)
    counting_d.count = 0
    return counting_d
