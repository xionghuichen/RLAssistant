# RL training assistant

RLA is a tool for managing your RL experiments automatically (e.g., your hyper-parameters, logs, checkpoints, figures, and code, etc.).  RLA has decoupled to the training code although some additional configuration codes and predetermined directory structures are still needed (see Quickstart and the example project). 

PS: The repo is inspired by [openai/baselines](https://github.com/openai/baselines).

## Quickstart

#### Initialization
Step1: config config.yaml
config.yaml is used to define the workflow of easy_log. It is necessary to config before use RLA.

See ./example/config.yaml for more details.

Step2: config the Tester object in your main file, which is a manager of RLA.

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

#### Structure of logs and usages
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
logger.record_tabular("performance/reward", reward)
logger.dump_tabular()
```

these files are stored in "LOG_ROOT/log/log_name/"

figures: you can record figures easily by the api in RLA.easy_log.simple_mat_plot

```python
from RLA.easy_log.simple_mat_plot import *
simple_plot(name="test_plot", data=[[1,2,3,4,5,6]])
```
the figure will be stored in "LOG_ROOT/results/log_name/"

hyper-parameters: the hyperparameters will be recorded in the "text" tab of Tensorboard and a pickle file of Tester object.
The Tester object is stored in "LOG_ROOT/archive_tester/log_name/"

"log_name" above is a formatted string contraining "task_name/datetime/ip/record_param"


## An example
you can find a project demo from the example directory.

# TODO
1. to be compatible with Pytorch;  
2. add comments and documents to other functions.
