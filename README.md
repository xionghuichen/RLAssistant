# rl_training_assistant
## easy_log: Tools to record your experiments.
### Usage

#### initialization
Step1: config config.yaml
config.yaml is used to defined the work flow of easy_log. It is nessary to config before use RLA.

See ./example/config.yaml for more details.

Step2: config the Tester object in your main file, which is a manger of RLA.

```python
from RLA.easy_log.tester import tester
task_name = 'demo_task'
private_config_path = './example/config.yaml'
tester.configure(task_name=task_name, private_config_path=private_config_path)
```

Step3: record hyperparameters

```python
from RLA.easy_log.tester import tester
kwargs = {"hp_a": 1, "hp_b": 2, "info": "description of the experiment"}
tester.set_hyper_param(**kwargs)
# add the hyperparameters to track.
tester.add_record_param(["info", "hp_a"])
```

Step4: initialize log files:

```python
from RLA.easy_log.tester import tester
tester.log_files_gen()

```

#### structure of logs and usages
For each experiment, RLA generate the following logs:


**source code**: the source code of your project when running the experiment, which can config in "BACKUP_CONFIG" of config.yaml. 

The source code can be found in "LOG_ROOT/code/log_name/".

The backup process is done after "tester.log_files_gen()".


**checkpoint**: the model parameters is saved in "LOG_ROOT/checkpoint/log_name/". You can update/initialze your checkpoint by:

```python
from RLA.easy_log.tester import tester
# after graph initialize
tester.new_saver(var_prefix='', max_to_keep=1)
# after training
tester.save_checkpoint()
```

**record variables**: you can record any scale into tensorboard, csv file and standard output by the flowwing code:

```python
from RLA.easy_log import logger
reward = 10
logger.logkv("performance/reward", reward)
logger.dump_tabular()
```

these files are stored in "LOG_ROOT/log/log_name/"

figures: you can record figures easily by the api in RLA.easy_log.simple_mat_plot

```python
from RLA.easy_log.simple_mat_plot import *
simple_plot(name="test_plot", data=[[1,2,3,4,5,6]])
```
the figure will be stored in "LOG_ROOT/results/log_name/"

hyper-parameters: the hyperparameters will be record in the "text" tab of tensorboard and a pickle files of Tester object.
The Tester object is stored in "LOG_ROOT/archive_tester/log_name/"

"log_name" above is a formatted string contraining "task_name/datetime/ip/record_param"


## An example
you can find a project demo from the example directory.

# TODO
1. be compatible to torch;  