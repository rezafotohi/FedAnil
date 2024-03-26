# FedAnil
FedAnil is a secure Blockchain and Homomorphic encryption-enabled Federated Deep Learning Model to address non-IID data and privacy concerns. This repo hosts a simulation for FedAnil written in Python.

## Introduction
In the traditional DL-based model, the models are jointly trained without leaving the training data from the enterprise's enterprises, and the privacy of this data is preserved. However, existing DL-based methods are vulnerable to poisoning and inference attacks and violate privacy. Also, enterprises' unbalanced and non-IID data harm the model's accuracy and performance. Therefore, to tackle this issue, we proposed FedAnil, a secure blockchain-enabled Federated Deep Learning Model to improve enterprise models' decentralized and tamper-proof properties that include two main phases. The first phase is presented to address the non-IID data challenge. The second phase is presented to address the privacy-preserving challenge.

For detailed explanations, please refer to the [*Decentralized and Robust Privacy-Preserving Model Using Blockchain-Enabled Federated Deep Learning in Intelligent Enterprises*](https://ieeexplore.ieee.org/abstract/document/10128790).


## Instructions to Run
(1) Suggested Environments Setup
```
Python 3.10
PyTorch 2.2.1
```
(2) Clone the repo
```
$ git clone https://github.com/rezafotohi/FedAnil.git
```
(3) Create a new conda environment with Python 3.10
```
$ conda create -n FedAnil python=3.10
$ conda activate FedAnil
```
(4) Install PyTorch and Jupyter
```
$ conda install pytorch torchvision torchaudio -c pytorch
$ conda install -c conda-forge jupyter jupyterlab
```
(5) Is the torch installed successfully or not? Enter the following commands in the terminal:
```
$ python3
$ import torch
```
(6) Install Pycryptodome and Matplotlib
```
$ conda install pycryptodome
$ conda install matplotlib
```
(7) Install Scikit-Learn
```
$ pip3 install scikit-learn-extra
```
(8) Install Bitarray
```
$ pip3 install bitarray
```
(9) Install TenSEAL
```
$ pip install git+https://github.com/OpenMined/TenSEAL.git#egg=tenseal
```

(10) Run FedAnil Simulation
```
$ python3 main.py -nd 100 -max_ncomm 50 -ha 80,10,10 -aio 1 -pow 0 -ko 5 -nm 3 -vh 0.08 -cs 0 -B 64 -mn OARF -iid 0 -lr 0.01 -dtx 1 -le 20
```

## FedAnil arguments

<b>-nd 20</b>: 20 Enterprises.

<b>-max_ncomm 50</b>: maximum 50 communication rounds.

<b>-ha 80,10,10</b>: role assignment hard-assigned to 80 workers, 10 validators, and 10 miners for each communication round. A <b>*</b> in <b>-ha</b> means the corresponding number of roles is not limited. e.g., <b>-ha \*,10,\*</b> means at least 5 validators would be assigned in each communication round, and the rest of the enterprises are dynamically and randomly assigned to any role. <b>-ha \*,\*,\*</b> means the role-assigning in each communication round is completely dynamic and random.

<b>-aio 1</b>: <i>aio</i> means "all in one network", namely, every enterprise in the emulation has every other enterprise in its peer list. This is to simulate that FedAnil runs on a Permissioned blockchain (consortium blockchain). If using <b>-aio 0</b>, the emulation will let an enterprise (registrant) randomly register with another enterprise (register) and copy the register's peer list.

<b>-pow 0</b>: the argument of <b>-pow</b> specifies the proof-of-work difficulty. When using 0, FedAnil runs with FedAnil-PoS consensus to select the winning miner.

<b>-ko 5</b>: this argument means an enterprise is blacklisted after it is identified as malicious after 6 consecutive rounds as a worker.

<b>-nm 3</b>: exactly 3 enterprises will be malicious nodes.

<b>-vh 0.08</b>: validator-threshold is set to 0.08 for all communication rounds. This value may be adaptively learned by validators in a future version.

<b>-cs 0</b>: as the emulation does not include mechanisms to disturb the digital signature of the transactions, this argument turns off signature checking to speed up the execution.

Federated Learning arguments (inherited from https://github.com/WHDY/FedAvg)

<b>-B 64</b>: batch size set to 64.

<b>-mn mnist_cnn</b>: use mnist_cnn model. Another choice is mnist_2nn, or you may put your network inside of <i>Models.py</i> and specify it.

<b>-iid 0</b>: shard the training data set in Non-IID way.

<b>-lr 0.01</b>: learning rate set to 0.01.

Other arguments

<b>-dtx 1</b>: see <b>Known Issue</b>.

Please see <i>main.py</i> for other argument options.

## Emulation Logs
#### Examining the Logs

While running, the program saves the emulation logs inside of the <i>log/\<execution_time\></i> folder. The logs are saved based on communication rounds. In the corresponding round folder, you may find the model accuracy evaluated by each enterprise using the global model at the end of each communication round. You may also find each worker's local training accuracy, the validation-accuracy-difference value of each validator, and the final stake rewarded to each enterprise in this communication round. Outside of the round folders, you may also find the malicious enterprises identification log.

#### Plotting Experimental Results

The code for plotting the experimental results is provided in the <i>plottings</i> folder. The path of the desired log folder has to be specified for the plotting code to run. Please look at the code to determine the argument type or look at the samples in <i>.vscode/launch.json</i>.

The logs used for the figures inside of the paper can be found in <i>plotting_logs.zip</i>.

## Known Issue
If you use a GPU with a ram of less than 16GB, you may encounter the issue of <b>CUDA out of memory</b>. The reason causing this issue may be that the local model updates (i.e., neural network models) stored inside the blocks occupy the CUDA memory and cannot be automatically released because the memory taken in CUDA increases as the communication round progresses. A few solutions have been tried without luck.

A temporary solution is to specify <b>-dtx 1</b>. This argument lets the program delete the transactions stored inside of the last block to release the CUDA memory as much as possible. However, specifying <b>-dtx 1</b> will also turn off the chain-resyncing functionality as the resyncing process requires enterprises to reperform global model updates based on the transactions stored inside of the resynced chain, which has empty transactions in each block. As a result, using GPU should only emulate the situation that FedAnil runs in its most ideal situation, that is, every available transaction would be recorded inside of the block of each round, as specified by the default arguments.

The experimental results shown in the paper were obtained from Google Colab Pro with Nvidia Tesla V100, which in most situations can run 100 communication rounds with 20 enterprises. If you wish to test a more complicated running environment, such as specifying a '--miner_acception_wait_time' to limit the validator-transaction accretion time for miners, then each miner may end up with blocks having different validator-transactions and a forking event will require chain resyncing, then at this moment, please use CPU with a high ram. Fixing's underway.

Please raise other issues and concerns you found. Thank you!

## Acknowledgments
(1) The code of the Blockchain Architecture used in FedAnil is inspired  [*Fully functional blockchain application implemented in Python from scratch*](https://github.com/satwikkansal/python_blockchain_app) by Satwik Kansal.

(2) The code of the Validation and Consensus scheme used in FedAnil is inspired [*VBFL*](https://github.com/hanglearning/VBFL) by Hang Chen.

(3) The code of the FedAvg used in FedAnil is inspired [*WHDY's FedAvg implementation*](https://github.com/WHDY/FedAvg) by WHDY.
