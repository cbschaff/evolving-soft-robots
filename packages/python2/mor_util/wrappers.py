from mor.reduction import container


class ObjToAnimate(container.ObjToAnimate):

    def __init__(self, location, animFct='defaultShaking', item=None, save_period=1.0,
                 duration=1.0, dt=0.01, **params):
        if 'incr' in params or 'rangeOfAction' in params or 'incrPeriod' in params:
            container.ObjToAnimate(self, location, animFct, item,
                                   duration, **params)
        else:
            params['rangeOfAction'] = duration
            params['incr'] = save_period * dt
            params['incrPeriod'] = save_period
            container.ObjToAnimate.__init__(self, location, animFct, item,
                                            duration, **params)
