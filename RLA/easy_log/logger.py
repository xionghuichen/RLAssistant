import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
import numpy as np
from collections import defaultdict, deque


from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
from contextlib import contextmanager
from RLA.const import DEFAULT_X_NAME, FRAMEWORK

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
BACKUP = 60

DISABLED = 50

class KVWriter(object):
    def __init__(self):
        self.format_name = None
    """
    Key Value writer
    """
    def writekvs(self, kvs):
        """
        write a dictionary to file

        :param kvs: (dict)
        """
        raise NotImplementedError

class SeqWriter(object):
    """
    sequence writer
    """
    def writeseq(self, seq):
        """
        write an array to file

        :param seq: (list)
        """
        raise NotImplementedError

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        """
        log to a file, in a human readable format

        :param filename_or_file: (str or File) the file to write the log to
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'at')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'write'), 'Expected file or str, got {}'.format(filename_or_file)
            self.file = filename_or_file
            self.own_file = False
        super(HumanOutputFormat).__init__()

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn('Tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 40)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()


    def _truncate(self, s):
        return s[:40] + '...' if len(s) > 80 else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1: # add space unless this is the last one
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        if self.own_file:
            self.file.close()

class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        """
        log to a file, in the JSON format

        :param filename: (str) the file to write the log to
        """
        self.file = open(filename, 'at')
        super(JSONOutputFormat).__init__()

    def writekvs(self, kvs):
        for key, value in sorted(kvs.items()):
            if hasattr(value, 'dtype'):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    kvs[key] = float(value)
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    kvs[key] = value.tolist()
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        self.file.close()

class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        """
        log to a file, in a CSV format

        :param filename: (str) the file to write the log to
        """
        self.filename = filename
        self.file = open(filename, 'a+t')
        self.file.seek(0)
        keys = self.file.readline()
        if keys != '':
            keys = keys[:-1] # skip '\n'
            keys = keys.split(',')
            self.keys = keys
        else:
            self.keys = []
        self.file = open(filename, 'a+t')
        self.sep = ','
        super(CSVOutputFormat).__init__()

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file = open(self.filename, 'w+t')
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(key)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
            self.file = open(self.filename, 'a+t')
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            value = kvs.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        self.file.close()


def summary_val(key, value):
    """
    :param key: (str)
    :param value: (float)
    """
    kwargs = {'tag': key, 'simple_value': float(value)}
    import tensorflow as tf
    return tf.Summary.Value(**kwargs)


def valid_float_value(value):
    """
    Returns True if the value can be successfully cast into a float

    :param value: (Any) the value to check
    :return: (bool)
    """
    try:
        float(value)
        return True
    except TypeError:
        return False


def get_tbx_writer():
    tb_fmt = None
    for fmt in Logger.CURRENT.output_formats:
        if isinstance(fmt, TensorBoardOutputFormat):
            tb_fmt = fmt 
    assert tb_fmt is not None, "cannot find TensorBoard-format output in the database. " \
                               "Please check the key LOG_USED in rla_config.yaml"
    assert tb_fmt.framework == FRAMEWORK.torch, "tensorboardX writer is constructed in torch framework"
    return tb_fmt.tbx_writer


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, dir, framework):
        os.makedirs(dir, exist_ok=True)
        self.framework = framework
        self.dir = dir
        self.step = 1
        prefix = 'events'
        path = osp.join(osp.abspath(dir), prefix)
        if self.framework == FRAMEWORK.tensorflow:
            import tensorflow as tf
            self.tb_writer = tf.summary.FileWriter(path)
            self.tbx_writer = None
            from tensorflow.python import pywrap_tensorflow
            from tensorflow.core.util import event_pb2
            from tensorflow.python.util import compat
            self.tf = tf
            self.event_pb2 = event_pb2
            self.pywrap_tensorflow = pywrap_tensorflow
        elif self.framework == FRAMEWORK.torch:
            self.tb_writer = None
            from tensorboardX import SummaryWriter
            self.tbx_writer = SummaryWriter(path)
        else:
            raise NotImplementedError

        super(TensorBoardOutputFormat).__init__()
        # try:
        #     # self.writer = tf.summary.FileWriter(path) # pywrap_tensorflow.EventsWriter(compat.as_bytes(path))
        # except Exception:
        #     self.tb_writer = None
        #     from tensorboardX import SummaryWriter
        #     self.tbx_writer = SummaryWriter(path)

        # import tensorflow as tf

    @property
    def writer(self):
        return self.tbx_writer if self.tb_writer is None else self.tb_writer

    def add_hyper_params_to_tb(self, hyper_param, metric_dict=None):
        if self.framework == FRAMEWORK.tensorflow:
            import tensorflow as tf
            with tf.Session(graph=tf.Graph()) as sess:
                hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in hyper_param.items()]
                summary = sess.run(tf.summary.text('hyperparameters', tf.stack(hyperparameters)))
                self.tb_writer.add_summary(summary, self.step)
        elif self.framework == FRAMEWORK.torch:
            import pprint
            if metric_dict is None:
                pp = pprint.PrettyPrinter(indent=4)
                self.writer.add_text('hyperparameters', pp.pformat(hyper_param))
            else:
                self.writer.add_hparams(hyper_param, metric_dict)
        else:
            raise NotImplementedError

    def writekvs(self, kvs):
        if self.framework == FRAMEWORK.tensorflow:
            def summary_val(k, v):
                kwargs = {'tag': k, 'simple_value': float(v)}
                return self.tf.Summary.Value(**kwargs)
            summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
            event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
            event.step = self.step  # is there any reason why you'd want to specify the step?
            self.writer.add_event(event)
            self.writer.flush()
        elif self.framework == FRAMEWORK.torch:
            def summary_val(k, v):
                kwargs = {'tag': k, 'scalar_value': float(v), 'global_step': self.step}
                self.writer.add_scalar(**kwargs)
                # return self.tf.Summary.Value(**kwargs)
            for k, v in kvs.items():
                summary_val(k, v)
        else:
            raise NotImplementedError


    #
    # def writekvs(self, kvs):
    #     def summary_val(k, v):
    #         kwargs = {'tag': k, 'scalar_value': float(v), 'global_step': self.step}
    #         self.writer.add_scalar(**kwargs)
    #         # return self.tf.Summary.Value(**kwargs)
    #     for k, v in kvs.items():
    #         summary_val(k, v)
        # summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        # event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        # event.step = self.step # is there any reason why you'd want to specify the step?
        # self.writer.add_event(event)
        # self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix='', framework='tensorflow'):
    """
    return a logger for the requested format

    :param _format: (str) the requested format to log to ('stdout', 'log', 'json', 'csv' or 'tensorboard')
    :param ev_dir: (str) the logging directory
    :param log_suffix: (str) the suffix for the log file
    :return: (KVWrite) the logger
    """
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        parsed_format =  HumanOutputFormat(sys.stdout)
    elif format == 'log':
        parsed_format = HumanOutputFormat(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    elif format == 'warn':
        parsed_format = HumanOutputFormat(osp.join(ev_dir, 'warn%s.txt' % log_suffix))
    elif format == 'backup':
        parsed_format = HumanOutputFormat(osp.join(ev_dir, 'backup%s.txt' % log_suffix))
    elif format == 'json':
        parsed_format = JSONOutputFormat(osp.join(ev_dir, 'progress%s.json' % log_suffix))
    elif format == 'csv':
        parsed_format = CSVOutputFormat(osp.join(ev_dir, 'progress%s.csv' % log_suffix))
    elif format == 'tensorboard':
        parsed_format = TensorBoardOutputFormat(osp.join(ev_dir, 'tb%s' % log_suffix), framework)
    else:
        raise ValueError('Unknown format specified: %s' % (format,))
    parsed_format.format_name = format
    return parsed_format

# ================================================================
# API
# ================================================================

def timestep():
    from RLA.easy_log.time_step import time_step_holder
    return time_step_holder.get_time()
    # for fmt in Logger.CURRENT.output_formats:
    #     if isinstance(fmt, TensorBoardOutputFormat):
    #         return fmt.step
    # raise NotImplementedError


ma_dict = {}


def ma_record_tabular(key, val, record_len, ignore_nan=False, exclude:Optional[Union[str, Tuple[str, ...]]]=None):
    if key not in ma_dict:
        ma_dict[key] = deque(maxlen=record_len)
    if ignore_nan:
        if val != np.nan:
            ma_dict[key].append(val)
    else:
        ma_dict[key].append(val)
    if len(ma_dict[key]) == record_len:
        record_tabular(key, np.mean(ma_dict[key]), exclude)

def logkv(key, val, exclude:Optional[Union[str, Tuple[str, ...]]]=None):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.

    :param key: (Any) save to log this key
    :param val: (Any) save to log this value
    """
    get_current().logkv(key, val, exclude)


def log_from_tf_summary(summary):
    """
    add summary of tensorflow to the logger system
    """
    from tensorflow.core.framework import summary_pb2
    summ = summary_pb2.Summary()
    summ.ParseFromString(summary)
    list_field = summ.ListFields()
    # log scale values
    def recursion_util(inp_field):
        if hasattr(inp_field, "__getitem__"):
            for inp in inp_field:
                recursion_util(inp)
        elif hasattr(inp_field, 'simple_value'):
            record_tabular(inp_field.tag, inp_field.simple_value)
        else:
            pass
    recursion_util(list_field)
    # log other format of values
    for fmt in Logger.CURRENT.output_formats:
        if isinstance(fmt, TensorBoardOutputFormat):
            if fmt.tb_writer is not None:
                fmt.tb_writer.add_summary(summary, fmt.step)
                fmt.tb_writer.flush()
            else:
                warn("Failed to find tb_writer.")

def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)

def logkvs(d, exclude:Optional[Union[str, Tuple[str, ...]]]=None):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v, exclude)


def log_key_value(keys, values, prefix_name=''):
    """
    Log a dictionary of key-value pairs
    """
    for k, v in zip(keys, values):
        logkv(prefix_name + k, v)
    # for (k, v) in d.itemsms():
    #     logkv(k, v)

def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    if get_current().name2val != dict():
        try:
            get_current().logkv(DEFAULT_X_NAME, timestep())
        except NotImplementedError as e:
            print(e)
            for fmt in Logger.CURRENT.output_formats:
                print(fmt)
    return get_current().dumpkvs()

def getkvs():
    return get_current().name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)

def debug(*args):

    log(*args, level=DEBUG)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    args = list(args)
    get_current()
    args[0] = "[WARN] {} : {}".format(timestep(), args[0])
    args = tuple(args)
    log(*args, level=WARN)

def backup(*args):
    args = list(args)
    get_current()
    args[0] = "[BACKUP] {} : {}".format(timestep(), args[0])
    args = tuple(args)
    log(*args, level=BACKUP)

def error(*args):

    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)

def set_comm(comm):
    get_current().set_comm(comm)

def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()

record_tabular = logkv
dump_tabular = dumpkvs

class ProfileKV:
    def __init__(self, name):
        """
        Usage:
        with logger.ProfileKV("interesting_scope"):
            code

        :param name: (str) the profiling name
        """
        self.name = "wait_" + name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, _type, value, traceback):
        Logger.CURRENT.name2val[self.name] += time.time() - self.start_time


def profile(name):
    """
    Usage:
    @profile("my_func")
    def my_func(): code

    :param name: (str) the profiling name
    :return: (function) the wrapped function
    """
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with ProfileKV(name):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name



# ================================================================
# Backend
# ================================================================

def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, warn_output_formats, backup_output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.exclude_name = defaultdict(str)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.warn_output_log = warn_output_formats
        self.backup_output_log = backup_output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val, exclude:Optional[Union[str, Tuple[str, ...]]]=None):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: (Any) save to log this key
        :param val: (Any) save to log this value
        """
        self.name2val[key] = val
        self.exclude_name[key] = exclude

    def logkv_mean(self, key, val, exclude:Optional[Union[str, Tuple[str, ...]]]=None):
        """
        The same as logkv(), but if called many times, values averaged.

        :param key: (Any) save to log this key
        :param val: (Number) save to log this value
        """
        if val is None:
            self.name2val[key] = None
            self.exclude_name[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1
        self.exclude_name[key] = exclude

    def dumpkvs(self):
        d = self.name2val
        out = d.copy() # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                d2 = d.copy()
                for k, v in d.items():
                    if self.exclude_name[k] is not None and fmt.format_name in self.exclude_name[k]:
                        del d2[k]
                fmt.writekvs(d2)
        self.name2val.clear()
        self.name2cnt.clear()
        self.exclude_name.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)
        if level > INFO and level < BACKUP:
            self.warn_output_log.writeseq(map(str, args))
        elif level == BACKUP:
            self.backup_output_log.writeseq(map(str, args))
        # else:
        #     raise NotImplementedError


    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

def configure(dir=None, format_strs=None, comm=None, framework='tensorflow'):
    """
    configure the current logger

    :param folder: (str) the save location (if None, $OPENAI_LOGDIR, if still None, tempdir/openai-[date & time])
    :param format_strs: (list) the output logging format
        (if None, $OPENAI_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    """
    if dir is None:
        dir = os.getenv('OPENAI_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    log_suffix = ''
    if format_strs is None:
        format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix, framework) for f in format_strs]
    warn_output_formats = make_output_format('warn', dir, log_suffix, framework)
    backup_output_formats = make_output_format('backup', dir, log_suffix, framework)

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, warn_output_formats=warn_output_formats,
                            backup_output_formats=backup_output_formats,
                            comm=comm)
    log('Logging to %s'%dir)

def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT

def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')

@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger

# ================================================================

def _demo():
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    dir = "/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see b = -33.3")

    logkv("b", -2.5)
    dumpkvs()

    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()


# ================================================================
# Readers
# ================================================================

def read_json(fname):
    import pandas
    ds = []
    with open(fname, 'rt') as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pandas.DataFrame(ds)

def read_csv(fname):
    import pandas
    return pandas.read_csv(fname, index_col=False, comment='#')

def read_tb(path):
    """
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    import pandas
    import numpy as np
    from glob import glob
    import tensorflow as tf
    if osp.isdir(path):
        fnames = glob(osp.join(path, "events.*"))
    elif osp.basename(path).startswith("events."):
        fnames = [path]
    else:
        raise NotImplementedError("Expected tensorboard file or directory containing them. Got %s"%path)
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for v in summary.summary.value:
                    pair = (summary.step, v.simple_value)
                    tag2pairs[v.tag].append(pair)
                maxstep = max(summary.step, maxstep)
    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx,tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step-1, colidx] = value
    return pandas.DataFrame(data, columns=tags)

if __name__ == "__main__":
    _demo()
