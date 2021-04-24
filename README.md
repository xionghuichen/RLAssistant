# RL training assistant

RLA is a tool for managing your RL experiments automatically (e.g., your hyper-parameters, logs, checkpoints, figures, and code, etc.).  RLA has decoupled to the training code although some additional configuration codes and predetermined directory structures are still needed (see Quickstart and the example project). 

The logger function of RLA is forked from and compatible with the logger object in [openai/baselines](https://github.com/openai/baselines). You can transfer your logging system easily from the "baselines.logger" by modifying the import lines (i.e., ```from baselines import logger``` -> ```from RLA.easy_log import logger```).



## Quickstart

Step1: config config.yaml

config.yaml is used to define the workflow of easy_log. It is necessary to config before use RLA.

See ./example/config.yaml for more details.

Step2: config the Tester object in your main file, which is a manager of RLA.

```python
from RLA.easy_log.tester import tester
task_name = 'demo_task'
private_config_path = '../../../rla_config.yaml'
your_main_file_name = 'main.py'
log_root_path = '../'
tester.configure(task_name=task_name, private_config_path=private_config_path, 
run_file=your_main_file_name, log_root=log_root_path)
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

### Structure of logs and usages
In RLA, we name each experiment with a formatted string containing: task_name (named by programmers), datetime of the experiment, ip address, and tracked hyper-param. We call it "log_name".  See the example project for more details. For each experiment, RLA generates the following logs:


**Source code**: Back up the source code of your project, which can config in "BACKUP_CONFIG" of config.yaml. 
Then you can trace back the code-level implementation details for the historical experiment. As we know, any code-level tricks may make big difference to the final performance especially in RL. 

The source code can be found in "LOG_ROOT/code/log_name/".

The backup process is done by "tester.log_files_gen()".


**Checkpoint**: The model parameters is saved in "LOG_ROOT/checkpoint/log_name/". You can update/initialze your checkpoint by:

```python
from RLA.easy_log.tester import tester
# after graph initialized
tester.new_saver(var_prefix='', max_to_keep=1)
# after training
tester.save_checkpoint()
```
NOTE: We only implement the checkpoint in Tensorflow.

**Record variables**: You can record any scale into tensorboard, csv file and standard output by the flowing code:

```python
from RLA.easy_log import logger
reward = 10
logger.record_tabular("performance/reward", reward)
logger.dump_tabular()
```

these files are stored in "LOG_ROOT/log/log_name/"

Figures: you can record figures easily by the api in RLA.easy_log.simple_mat_plot

```python
from RLA.easy_log.simple_mat_plot import *
simple_plot(name="test_plot", data=[[1,2,3,4,5,6]])
```
the figure will be stored in "LOG_ROOT/results/log_name/"

Hyper-parameters: the hyperparameters will be recorded in the "text" tab of Tensorboard and a pickle file of Tester object.
The Tester object is stored in "LOG_ROOT/archive_tester/log_name/"



## An example
you can find a project demo from the "example" directory.

# TODO
- [ ] add a video to visualize the workflow of RLA.
- [ ] add comments and documents to other functions.
- [ ] add an auto integration script.
- [ ] plot function can do average on varied length (time-step) of experiment curves.
- [ ] download / upload experiment logs through timestamp

