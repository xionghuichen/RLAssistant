# RL experiment Assistant

RLA is a tool for managing your RL experiments automatically (e.g., your hyper-parameters, logs, checkpoints, figures, and code, etc.). 
RLA has decoupled to the training code and only some additional configuration are needed. Before using RLA, we recommend you to read the section "Design Principles of RLA", which will be helpful for you to understand the basic logic of the repo. 

[comment]: <> (The logger function of RLA is forked from and compatible with the logger object in [openai/baselines]&#40;https://github.com/openai/baselines&#41;. You can transfer your logging system easily from the "baselines.logger" by modifying the import lines &#40;i.e., ```from baselines import logger``` -> ```from RLA.easy_log import logger```&#41;.)

The project is still in developing. Welcome to join us. :)

## Design Principles of RLA
The core design principle of RLA is regarding all related information of each Experiment As A complex item in a Database (EaaD). 
We design RLA to manage the experiments dataset by 
1. formulating and structuring the item, table and the database；
2. provide tools for adding, deleting, modifying and querying the items in the database.

The following is the detailed designs of EaaD and the implementation of RLA.
### Formulating and Structuring the Experiment Database
After integrating RLA into your project, we create a "database" implicitly configured by `rla_config.yaml`. 
Each experiment we run will be indexed and written as an item into a "table". There include several elements of the system.
1. **Database**: A database is configured by a YAML file `rla_config.yaml`. In our practice, we only create one database in one subject.
2. **Table**: We map the concept of Table in standard database system into the concept of "task" in our research process. There are many similarities between the two concepts. For example, we will create another Table often when:
   1. the structure of two tables are different, e.g., they have different keys and logic for organization. In the research process, different tasks often have total different types of log to record. For example, in offline model-based RL, the first task might pretrain a dynamics model the second task might be policy learning with the learned model. In model learning, we concern the MSE of the model; In policy learning, we concern the rewards of policy. 
   2. The content of a table is too large which might hurt the speed of querying. In the research process, we need large memory to load a Tensorboard if the logdir have many runs.
3. **Data Item**:  We map the concept of the data item to the generated data in an experiment. For each data item, we need to define the index and the value (e.g., columns and values) for each item.
   1. Index: We need a unique index to define the item in a table for item adding, deleting, modifying and querying. In RLA, we define the index of each experiment as: `datetime of the experiment (for uniqueness) + ip address (for keeping uniqueness in distributed training) + tracked hyper-parameters (easy to identify and easy to search)`.
   2. Value: when running an experiment, we generate many data with different structure. Based on our research practice, currently we formulate and store the following data
      1. Codes and hyper-parameters: Every line of the code and the select hyper-parameters to run the experiment. This is a backup for experiment reproducibility.    
      2. Recorded variables: We often record many intermediate variables in the process of experiment, e.g., the rewards, some losses or learning rates. We record the variables in key-value formulation and store them in a tensorboard event and a csv file.   
      3. Model checkpoints: We support weights saving of neural networks and related custom variables in Tensorflow and Pytorch framework. We can use the checkpoints to resume an experiments or use the results of the experiment to complete downstream tasks.
      4. Other data like figures or videos: It essentially is unstructured intermediate variables in the process of the experiment. We might plot the frame-to-frame video of your agent behavior or some curves to check the process of training. We give some common tools in the RL scenario to generate the related variables and store them in a directory. 

Currently, we store the data items in standard file system and manage the relation among data items, tables and database via a predefined directory structure. After running some experiments, the database will be something like this:
![img.png](resource/run-res.png)
- Here we construct a database in the project "sb_ppo_example". 
- The directory "archive_tester" is to store hyper-parameters and related variables for experiment resuming. 
- The directory "results" is to store other data like figures or videos.
- The directory "log" is to store recorded variables.
- The directory "code" is a backup of code for experiment reproducibility.
- The directory "checkpoint" save weights of neural networks.
- We have a table named "demo_task", which is the root directory of log/archive_tester/checkpoint/code/results. 


### Tools to Operate the Database

In standard database systems, the common used operations to manage a database is adding, deleting modifying and querying. We also give similar tools to manage RLA database.

Adding:
1. RLA.easy_log.tester.exp_manager is a global object to create an experiments and manger the data in the process of experiments.
2. RLA.easy_log.logger is a module to add recorded variables.
3. RLA.easy_log.simple_mat_plot is a module to construct other data like figures or videos

Deleting:
1. rla_scripts.delete_log: a tool to delete a data item by regex;

Modifying:
1. resume: RLA.easy_log.exp_loader.ExperimentLoader: a class to resume an experiments with different flexible settings. 

Querying:
1. tensorboard: the recorded variables is added to tensorboard events and can be loaded via standard tensorboard tools.
  ![img.png](resource/tb-img.png)
2. easy_plot: We give some APIs to load and visualize the data in csv files. The results will be something like this: 
    ![](resource/sample-plot.png)


### Other principles

The second design principle is easy for integration. It still has long way to go to achieve it. We give several example project integrating with RLA in the directory example. 

We also list the RL projects using RLA as follows:
1. https://github.com/xionghuichen/MAPLE
2. https://github.com/xionghuichen/CODAS

## Installation
```angular2html
git clone https://github.com/xionghuichen/RLAssistant.git
cd RLAssistant
python setup.py install
```

## Quickstart


We build a example project for integrating RLA, which can be seen in ./example/simplest_code.

Step1: 

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

**Figures**: you can record figures easily by the api in RLA.easy_log.simple_mat_plot

```python
from RLA.easy_log.simple_mat_plot import *
simple_plot(name="test_plot", data=[[1,2,3,4,5,6]])
```
the figure will be stored in "LOG_ROOT/results/log_name/"

**Hyper-parameters**: the hyperparameters will be recorded in the "text" tab of Tensorboard and a pickle file of Tester object.
The Tester object is stored in "LOG_ROOT/archive_tester/log_name/"



## An example
you can find a project demo from the "example" directory.

# TODO
- [ ] add a video to visualize the workflow of RLA.
- [ ] add comments and documents to other functions.
- [ ] add an auto integration script.
- [ ] download / upload experiment logs through timestamp;
- [ ] config default hyper-parameters of RLA automatically. 

