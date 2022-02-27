class BatteryParams:

    def __init__(self, c10, p1d, p2d, p3d, p4d, p5d, p1c, p2c, p3c, p4c, p5c, i_d):
        self._c10 = c10
        self._p1d = p1d
        self._p2d = p2d
        self._p3d = p3d
        self._p4d = p4d
        self._p5d = p5d
        self._p1c = p1c
        self._p2c = p2c
        self._p3c = p3c
        self._p4c = p4c
        self._p5c = p5c
        self._i_d = i_d

    # getter method 

    @property
    def c10(self):
        return self._c10

    @property
    def p1d(self):
        return self._p1d

    @property
    def p2d(self):
        return self._p2d

    @property
    def p3d(self):
        return self._p3d

    @property
    def p4d(self):
        return self._p4d

    @property
    def p5d(self):
        return self._p5d

    @property
    def p1c(self):
        return self._p1c

    @property
    def p2c(self):
        return self._p2c

    @property
    def p3c(self):
        return self._p3c

    @property
    def p4c(self):
        return self._p4c

    @property
    def p5c(self):
        return self._p5c

    @property
    def i_d(self):
        return self._i_d

    # setter method 

    @c10.setter
    def c10(self, x):
        self._c10 = x

    @p1d.setter
    def p1d(self, x):
        self._p1d = x

    @p2d.setter
    def p2d(self, x):
        self._p2d = x

    @p3d.setter
    def p3d(self, x):
        self._p3d = x

    @p4d.setter
    def p4d(self, x):
        self._p4d = x

    @p5d.setter
    def p5d(self, x):
        self._p5d = x

    @p1c.setter
    def p1c(self, x):
        self._p1c = x

    @p2c.setter
    def p2c(self, x):
        self._p2c = x

    @p3c.setter
    def p3c(self, x):
        self._p3c = x

    @p4c.setter
    def p4c(self, x):
        self._p4c = x

    @p5c.setter
    def p5c(self, x):
        self._p5c = x

    @i_d.setter
    def i_d(self, x):
        self._i_d = x
