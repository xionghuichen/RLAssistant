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

import tensorflow as tf

from contextlib import contextmanager
from RLA.const import DEFAULT_X_NAME

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

class KVWriter(object):
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
        self.keys = []
        self.sep = ','

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


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = 'events'
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = tf.summary.FileWriter(path) # pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return self.tf.Summary.Value(**kwargs)
        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step # is there any reason why you'd want to specify the step?
        self.writer.add_event(event)
        self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=''):
    """
    return a logger for the requested format

    :param _format: (str) the requested format to log to ('stdout', 'log', 'json', 'csv' or 'tensorboard')
    :param ev_dir: (str) the logging directory
    :param log_suffix: (str) the suffix for the log file
    :return: (KVWrite) the logger
    """
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        return HumanOutputFormat(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    elif format == 'warn':
        return HumanOutputFormat(osp.join(ev_dir, 'warn%s.txt' % log_suffix))
    elif format == 'json':
        return JSONOutputFormat(osp.join(ev_dir, 'progress%s.json' % log_suffix))
    elif format == 'csv':
        return CSVOutputFormat(osp.join(ev_dir, 'progress%s.csv' % log_suffix))
    elif format == 'tensorboard':
        return TensorBoardOutputFormat(osp.join(ev_dir, 'tb%s' % log_suffix))
    else:
        raise ValueError('Unknown format specified: %s' % (format,))

# ================================================================
# API
# ================================================================

def timestep():
    for fmt in Logger.CURRENT.output_formats:
        if isinstance(fmt, TensorBoardOutputFormat):
            return fmt.step
    raise NotImplementedError


ma_dict = {}


def ma_record_tabular(key, val, record_len, ignore_nan=False):
    if key not in ma_dict:
        ma_dict[key] = deque(maxlen=record_len)
    if ignore_nan:
        if val != np.nan:
            ma_dict[key].append(val)
    else:
        ma_dict[key].append(val)
    if len(ma_dict[key]) == record_len:
        record_tabular(key, np.mean(ma_dict[key]))

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.

    :param key: (Any) save to log this key
    :param val: (Any) save to log this value
    """
    get_current().logkv(key, val)

def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)

def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)

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

    def __init__(self, dir, output_formats, warn_output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.warn_output_log = warn_output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: (Any) save to log this key
        :param val: (Any) save to log this value
        """
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        """
        The same as logkv(), but if called many times, values averaged.

        :param key: (Any) save to log this key
        :param val: (Number) save to log this value
        """
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        d = self.name2val
        out = d.copy() # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)
        if level > INFO:
            self.warn_output_log.writeseq(map(str, args))


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

def configure(dir=None, format_strs=None, comm=None):
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

    def mpi_rank_or_zero():
        """
        Return the MPI rank if mpi is installed. Otherwise, return 0.
        :return: (int)
        """
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD.Get_rank()
        except ImportError:
            return 0

    rank = mpi_rank_or_zero()
    log_suffix = ''
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ['PMI_RANK', 'OMPI_COMM_WORLD_RANK']:
        if varname in os.environ:
            rank = int(os.environ[varname])
    if rank > 0:
        log_suffix = "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
        else:
            format_strs = os.getenv('OPENAI_LOG_FORMAT_MPI', 'log').split(',')
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]
    warn_output_formats = make_output_format('warn', dir, log_suffix)

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, warn_output_formats=warn_output_formats, comm=comm)
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
