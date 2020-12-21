from RLA.easy_log import logger

class BasicUnitTest(object):
    def __init__(self):
        pass

class TestType(object):
    SINGLE_TIME = 'single_time'
    FREQUENT = 'freq'
    KEY_MAP = 'hash'

class EmbUnitController(BasicUnitTest):
    def __init__(self):
        BasicUnitTest.__init__(self)
        self.test_obj_dict = {}
        self.test_type = TestType

    def add_test(self, test_func_name, test_func, test_type, freq=-1, keys=None):
        test_obj = EmbUnitTestObj(test_func, test_type, freq, keys)
        if test_func_name in self.test_obj_dict:
            return
        else:
            self.test_obj_dict[test_func_name] = test_obj

    def do_test(self, func_name, key=None, *args, **kwargs):
        assert func_name in self.test_obj_dict
        self.test_obj_dict[func_name](key, func_name, *args, **kwargs)
    #
    # def build_test(self, func):
    #     if hasattr(self, str(id(func))):
    #         pass
    #     else:
    #         def wrap_func(*args, **kwargs):
    #             if self.__getattribute__(str(id(func))):
    #                 pass
    #             else:
    #                 should_pass = func(*args, **kwargs)
    #                 if should_pass:
    #                     self.__setattr__(str(id(func)), True)
    #                     print("[pass test]")
    #         return wrap_func
    #

class EmbUnitTestObj(object):

    def __init__(self, test_func, test_type, freq=-1, keys=None):
        self.test_type = test_type
        self.name = test_func.__name__
        self.test_func = test_func
        self.__do_test_var = {}
        self.__do_test_var_init(freq, keys)

    def __do_test_var_init(self, freq, keys):
        if self.test_type == TestType.SINGLE_TIME:
            self.__do_test_var['res'] = True
        elif self.test_type == TestType.FREQUENT:
            assert type(freq) is int
            self.__do_test_var = {"count": 0, "freq": freq}
        elif self.test_type == TestType.KEY_MAP:
            assert keys is not None
            for k in keys:
                self.__do_test_var[k] = True

    def __if_do_test(self, key=None):
        if self.test_type == TestType.SINGLE_TIME:
            if self.__do_test_var['res']:
                self.__do_test_var['res'] = False
                return True
            else:
                return False
        elif self.test_type == TestType.FREQUENT:
            assert type(self.__do_test_var) is dict
            self.__do_test_var['count'] += 1
            return self.__do_test_var['count'] % self.__do_test_var['freq'] == 0
        elif self.test_type == TestType.KEY_MAP:
            if not self.__do_test_var[key]:
                self.__do_test_var[key] = True
                return True
            else:
                return False

    def __call__(self, key=None, func_name=None, *args, **kwargs):
        if self.__if_do_test(key):
            logger.info("do test {}, {} - {}".format(self.test_type, self.name, func_name))
            self.test_func(*args, **kwargs)







ut_controller = EmbUnitController()

