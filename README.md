# Using Offline Data to Speed-up Reinforcement Learning in Procedurally Generated Environments

***Repository for the journal extension under review***

Paper initially presented at [2023 Adaptive and Learning Agents Workshop](https://alaworkshop2023.github.io/#accepted) at AAMAS, London, UK

Preliminar results are available at:
- [Ala Workshop domain](https://alaworkshop2023.github.io/papers/ALA2023_paper_47.pdf)
- [Arxiv](https://arxiv.org/pdf/2304.09825)

## Citation
```
@article{andres2023using,
  title={Using Offline Data to Speed-up Reinforcement Learning in Procedurally Generated Environments},
  author={Andres, Alain and Sch{\"a}fer, Lukas and Villar-Rodriguez, Esther and Albrecht, Stefano V and Del Ser, Javier},
  journal={arXiv preprint arXiv:2304.09825},
  year={2023}
}

```

## Dependencies
We recommend using Python3.10 which is the one used for this project.

Clone and create a virtual/conda environment. For the first, you can do:
```
python -m venv .venv
source venv/bin/activate
```
Then, install the required dependencies set at requirements.txt:
```
pip install -r requirements.txt
```

## Code structure 
The implementation is based on the code provided in [`rl-starter-files`](https://github.com/lcswillems/rl-starter-files) repository, which we subsequently modified for our research purposes. Additionally, we adapted the original TensorFlow [`RAPID`](https://github.com/daochenzha/rapid) implementation to PyTorch. Therefore, while inspired by these sources, the core code is our own.
```
pytorch_pcg_il
├──gym_minigrid                       # gym_minigrid env logic
├──numpyworldfiles                    # Place where different buffers are expected to be placed                      
├──scripts                      
    ├──train.py                       # Used to launch a training. Here the hyperparameters can be found
├── storage                           # Used to store the results of a given run/simulation/experiment
├── torch_ac                          # Contains the whole logic of the code
│   ├── algos                         # Algorithms used in the project
        ├── ppo.py                    # Used for PPO logic and IL updates
        ├── base.py                   # Collects the data and process them to be later processed in ppo.py or train.py
│   └── utils                         # Util algorithmic features
        ├── intrinsic_motivation.py   # Implements tabular BeBold/NovelD logic
        ├── ranking_bufer.py          # Used by RAPID (for baselines and data collection)
├── utils.py                          # Util functions used through code
├── model.py                          # Defines agent's neural network architectures.
```
Other additional modules can be found, but might not be necessary for the current project

## Example of use 

### Baselines
Example of PPO in MN12S10 (```--disable_rapid``` has to be used):
```
python3 -m scripts.train.py --log_dir ppo_MN12S10_0 --env MiniGrid-MultiRoom-N12-S10-v0 --num_train_seeds 10000 --seed 0 --nstep 2048 --frames 1e7 --separated_networks 1 --disable_rapid 
```

Example of PPO in Ninja:
```
python3 -m scripts.train.py --log_dir ppo_ninja_0 --env ninja --disable_rapid --frames 25e6 --nsteps 16384 --epochs 3 --nminibatch 8 --lr 0.0005 --discount 0.999 --separated_networks 0 --num_train_seeds 200 --num_test_seeds 100
```
Example of RAPID in O1Dlhb:
```
python3 -m scripts.train.py --log_dir O1Dlhb_rapid_10klevels_0 --seed 0 --env MiniGrid-ObstructedMaze-1Dlhb-v0 --num_train_seeds 10000 --nstep 2048 --frames 1e7 --separated_networks 1
```

### Data Collection
Train agent with RAPID+BeBold in O2Dlh and store buffers (```--store_demos``` has to be used):
```
python3 -m scripts.train.py --log_dir O2Dlh_rapid_storebuffer_0 --env MiniGrid-ObstructedMaze-2Dlh-v0 --seed 0 --frames 3e7 --separated_networks 1 --nstep 2048 --num_train_seeds 10000 --im_type bebold --intrinsic_motivation 0.005 --store_demos
```
Train and collect data with RAPID in Ninja:
```
python3 -m scripts.train.py --log_dir ninja_datacollection --env ninja  --frames 25e6 --nsteps 16384 --epochs 3 --nminibatch 8 --lr 0.0005 --discount 0.999 --separated_networks 0 --num_train_seeds 200 --num_test_seeds 100 --w1 0 --w2 0 --store_demos 
```
(it would also be possible to collect data solely with PPO updates; in that case, add ```--disable_rapid``` & ```--do_buffer```)

### Only Pre-training
Example to how it would work IL only in pre-training with the 60% Buffer (0.4 return) in MN12S10 (```--disable_rapid``` has to be used to quit IL in the online phase;```--pre_train_epochs``` is used to determine the number of optimization steps):
```
python3 -m scripts.train.py --log_dir pretrain_MN12S10_rapid_avg04_loadbuffer_seed0 --env MiniGrid-MultiRoom-N12-S10-v0 --num_train_seeds 10000 --seed 0 --nstep 2048 --frames 1e7 --separated_networks 1 --load_demos --load_demos_path /numpyworldfiles/MN12S10_rapid_storebuffer_newstored_avgret0.4 --disable_rapid --pre_train_epochs 3000
```
Similarly, for Ninja with the 25M buffer:
```
python3 -m scripts.train.py --log_dir pretrain_ninja_25M_0 --seed 0 --env ninja --load_demos --load_demos_path /numpyworldfiles/ninja_datacollection_steps24900000  --pre_train_epochs 10000 --frames 25e6 --nsteps 16384 --epochs 3 --nminibatch 8 --lr 0.0005 --discount 0.999 --separated_networks 0 --num_train_seeds 200 --num_test_seeds 100 --disable_rapid
```

### Only Concurrent IL
Example to how it would work IL Concurrently (online) with RL with the 10% Buffer (0.1 return) in O1Dlhb:
```
python3 -m scripts.train.py --log_dir online_O1Dlhb_rapid_10klevels_avg01_loadbuffer_seed0 --seed 0 --env MiniGrid-ObstructedMaze-1Dlhb-v0 --num_train_seeds 10000 --nstep 2048 --frames 1e7 --separated_networks 1 --load_demos --load_demos_path /numpyworldfiles/O1Dlhb_rapid_storebuffer_10kseeds_avgret0.1 --use_gpu 1 --gpu_id 0
```
Similarly, for Ninja with the 7M buffer with 1episode per level constraint (remember setting ```--w1=0```,```--w2=0``` and ```--rank_type store_one_episode```:
```
python3 -m scripts.train.py --log_dir concurrentIL_1ep_ninja_7M_0 --seed 0 --env ninja --load_demos --load_demos_path /numpyworldfiles/ninja_datacollection_1ep_steps7500000 --rank_type store_one_episode --frames 25e6 --nsteps 16384 --epochs 3 --nminibatch 8 --lr 0.0005 --discount 0.999 --separated_networks 0 --num_train_seeds 200 --num_test_seeds 100 --w1 0 --w2 0 

```
### Pre-training & Concurrent IL
Example to how it would work IL only in pre-training with the 60% Buffer (0.4 return) in MN12S10 (```--pre_train_epochs``` to determine the number of optimization steps):
```
python3 -m scripts.train.py --log_dir pretrain_keepIL_MN12S10_rapid_avg04_loadbuffer_seed0 --env MiniGrid-MultiRoom-N12-S10-v0 --num_train_seeds 10000 --seed 0 --nstep 2048 --frames 1e7 --separated_networks 1 --load_demos --load_demos_path /numpyworldfiles/MN12S10_rapid_storebuffer_newstored_avgret0.4 --pre_train_epochs 10000
```

### Low-data Regime (pre-training) 
When executing experiments from 60% Buffer (suboptimal) for O1Dlhb with 2 demonstrations/levels with 10,000 optimization steps:
```
python3 -m scripts.train.py --log_dir pretrain_O1Dlhb_2suboptlevels_0 --seed 0 --env MiniGrid-ObstructedMaze-1Dlhb-v0 --num_train_seeds 10000 --nstep 2048 --frames 1e7 --separated_networks 1 --load_demos --load_demos_path /numpyworldfiles/O1Dlhb_2suboptlevels10k_1demoperlevel --disable_rapid --pre_train_epochs 10000
```
