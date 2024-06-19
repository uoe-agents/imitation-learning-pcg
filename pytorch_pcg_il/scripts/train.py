import argparse
import time
import datetime
# import torch_ac
import tensorboardX
import sys
import os
import numpy as np
import utils
import torch
import json

# for Actor-Critic
from model import *
from torch_ac.algos import *
# ******************************************************************************
# Parse arguments
# ******************************************************************************

parser = argparse.ArgumentParser()

################################################################################
# Logs related
################################################################################
parser.add_argument("--log_dir", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--log_interval", type=int, default=10, help="number of updates between two logs (default: 1)")
parser.add_argument("--save_interval", type=int, default=100, help="number of updates between two saves (default: 100, 0 means no saving)")

################################################################################
# Environment dependant
################################################################################
parser.add_argument("--env", default='MiniGrid-MultiRoom-N7-S8-v0', help="name of the environment to train on (REQUIRED)")
parser.add_argument("--env_list", nargs="+" , default=[], help="subset of files that we are going to use")
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument("--init_level_seed", type=int, default=0, help="set the initial levels/seed to train from. 0 sets to be random/infinite.")
parser.add_argument("--num_train_seeds", type=int, default=-1,help="number of seeds/levels used for training. -1 sets to be random/infinite.")
parser.add_argument("--num_test_seeds", type=int, default=10,help="number of seeds used for evaluation")
# parser.add_argument("--carrying_info", help="Used to use carrying info (object+color) in observation", action='store_true')

################################################################################
 ## Select algorithm and generic configuration params
 ################################################################################
parser.add_argument("--algo", default="ppo", help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--procs", type=int, default=1, help="number of processes (default: 1)")
parser.add_argument("--frames", type=float, default=int(1e7), help="number of frames of training (default: 1e7)")
parser.add_argument("--separated_networks", type=int, default=1, help="set if we use two different NN for actor and critic (default 1 for MiniGrid, 0 for Procgen)")

################################################################################
## Parameters for main algorithm (PPO)
################################################################################
parser.add_argument('--disable_ppo', help='Use to just collect experiences and train with Imitation Learning', action='store_true')
parser.add_argument("--epochs", type=int, default=4, help="number of epochs for PPO (default: 4 for MiniGrid, 3 for Procgen)")
parser.add_argument("--nminibatch", type=int, default=4, help="number of minibatches (default: 4 for MiniGrid, 8 for Procgen)")
parser.add_argument("--nsteps", type=int, default=2048, help="number of frames per process before update (default: 2048 for MiniGrid, 16384 for Procgen)")
parser.add_argument("--discount", type=float, default=0.99, help="discount factor (default: 0.99 for MiniGrid, 0.999 for Procgen)")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001 for MiniGrid, 5e-4 for Procgen)")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value_loss_coef", type=float, default=0.5, help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim_eps", type=float, default=1e-5, help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim_alpha", type=float, default=0.99, help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip_eps", type=float, default=0.2, help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--use_recurrence_critic", action='store_true')
parser.add_argument("--use_recurrence_actor", action='store_true')

################################################################################
## INTRINSIC MOTIVATION
################################################################################
parser.add_argument("--intrinsic_motivation", type=float, default=0, help="specify if we use intrinsic motivation (int_coef) to face sparse problems")
parser.add_argument("--im_type", default='counts', help="specify if we use intrinsic motivation, which module/approach to use")
parser.add_argument("--normalize_intrinsic_bonus", type=int, default=0, help="boolean-int variable that set whether we want to normalize the intrinsic rewards or not")
parser.add_argument("--use_episodic_counts", type=int, default=0, help="divide intrinsic rewards with the episodic counts for that given state")
parser.add_argument("--use_only_not_visited", type=int, default=0, help="apply mask to reward only those states that have not been explored in the episode")
# parser.add_argument("--reduced_im_networks", type=int, default=0, help="set if we use FC layers or a more sophisticated CNN architecture for IM embeddings")

################################################################################
# IMITATION LEARNING RELATED, BUFFER & SCORES
################################################################################
parser.add_argument('--w0', help='weight for extrinsic rewards', type=float, default=1.0)
parser.add_argument('--w1', help='weight for local bonus', type=float, default=0.1)
parser.add_argument('--w2', help='weight for global bonus', type=float, default=0.001)
parser.add_argument('--buffer_size', help='the size of the ranking buffer', type=int, default=10000)
parser.add_argument('--sl_batch_size', help='the batch size for SL', type=int, default=256)
parser.add_argument('--do_buffer', help='Habilitate the store buffer (use it to collect data)', action='store_true')
parser.add_argument('--disable_rapid', help='Disable SL, i.e., PPO', action='store_true')
parser.add_argument('--sl_num', help='Number of updated steps of SL', type=int, default=5)
parser.add_argument('--sl_clipgrad', help='Clip gradients of SL loss', type=int, default=0)
parser.add_argument('--ratio_offpolicy_update', help='Number of off-policy updates per on-policy update', type=float, default=1)
parser.add_argument('--use_full_traj', help='Use full trajectories instead of random experiences when sampling', action='store_true')
# store/load demonstrations
parser.add_argument('--store_demos', help='Store demonstration buffer experiences and trajectories', action='store_true')
parser.add_argument('--load_demos', help='Load demonstration buffer experiences and trajectories', action='store_true')
parser.add_argument('--load_counts_dict', help='Load counts dict for the demonstration w2 score', action='store_true')
parser.add_argument('--load_demos_path', help='Path to select the demos to import from', default='')
parser.add_argument('--rank_type', help='Determines the strategy followed in the buffer to keep the experiences', default='rapid',choices=['rapid','fifo','store_one_episode'])

# pre-training
parser.add_argument("--pre_train_epochs", type=int, default=0)

################################################################################
## GPU/CPU Configuration
################################################################################
parser.add_argument("--use_gpu", type=int, default=0,help="Specify to use GPU as device to bootstrap the training")
parser.add_argument("--gpu_id", type=int, default=-1,help="add a GRU to the model to handle text input")

# early stopping
parser.add_argument("--early-stopping", type=int, default=0, help="Stop the algorithm when achieving its optimal score (recommended to let to all the given frames)")

args = parser.parse_args()

# ******************************************************************************
# AssertionError to ensure inconsistency problems
# ******************************************************************************
# LOGIC --> if it is true, no error thrown
assert ((args.env =='MiniGrid-NumpyMapFourRoomsPartialView-v0') and (len(args.env_list)>0)) \
        or args.env != 'MiniGrid-NumpyMapFourRoomsPartialView-v0', \
        'You have selected to use Pre-defined environments for training but no subfile specified'

assert (args.use_gpu==False) or (args.use_gpu and args.gpu_id != -1), \
        'Specify the device id to use GPU'


# ******************************************************************************
# Set run dir
# ******************************************************************************
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.log_dir or default_model_name
model_dir = utils.get_model_dir(model_name)
root_dir = os.path.dirname(os.path.abspath('model.py'))
numpyfiles_dir = root_dir + '/numpyworldfiles/' + model_name
print('Root dir:',root_dir)
print('Numpyfiles dir:',numpyfiles_dir)
print('Model dir:',model_dir)

# ******************************************************************************
# Load loggers and Tensorboard writer
# ******************************************************************************
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# ******************************************************************************
# Set seed for all randomness sources
# ******************************************************************************
utils.seed(args.seed)

# ******************************************************************************
# Set device
# ******************************************************************************
device = torch.device("cuda:"+str(args.gpu_id) if args.use_gpu else "cpu")
txt_logger.info(f"Device: {device}\n")


# ******************************************************************************
# Generate ENVIRONMENT DICT
# ******************************************************************************
env_dict = {}
env_list = [name + '.npy' for name in args.env_list] #add file extension
env_dict[args.env] = env_list
print('Env Dictionary:',env_dict)

# ******************************************************************************
# Load environments
# ******************************************************************************
envs = []
envs_evaluation = []

# procgen possible environments
env_type = 'procgen' if args.env in ["climber","ninja"] else 'minigrid' 

if env_type == 'procgen':

    from utils import make_env_procgen as make_env
    args.score_type = 'continious'
    
    for i in range(args.procs):
        envs.append(make_env(env_dict=env_dict,
                            num_levels = args.num_train_seeds,
                            start_level=0,
                            distribution_mode='easy',
                    )
        )
    # evaluation environment
    envs_evaluation.append(make_env(env_dict=env_dict,
                                    num_levels = 0,
                                    distribution_mode='easy',
                        )       
    )
else:
    from utils import make_env_minigrid as make_env 
    args.score_type = 'discrete'
    
    for i in range(args.procs):
        envs.append(make_env(env_dict, args.seed + 10000 * i))


txt_logger.info("Environments loaded\n")

# Define action_space
ACTION_SPACE = envs[0].action_space.n
txt_logger.info(f"ACTION_SPACE: {ACTION_SPACE}")

# Load observations preprocessor
obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
obs_dim = np.prod(obs_space["image"])
txt_logger.info(f"STATE_SPACE: {obs_space}")
txt_logger.info(f"STATE_DIM: {obs_dim}")

# ******************************************************************************
# Load training status
# ******************************************************************************
try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# ******************************************************************************
# Load models of MAIN algorithm (PPO)
# ******************************************************************************

# Use 1 AC network or separated Actor and Critic
if args.separated_networks:
    actor = ActorModel_RAPID(obs_dim, ACTION_SPACE, args.use_recurrence_actor)
    critic = CriticModel_RAPID(obs_dim, ACTION_SPACE, args.use_recurrence_critic)
    actor.to(device)
    critic.to(device)
    if "model_state" in status:
        actor.load_state_dict(status["model_state"][0])
        critic.load_state_dict(status["model_state"][1])
    txt_logger.info("Models loaded\n")
    txt_logger.info("Actor: {}\n".format(actor))
    txt_logger.info("Critic: {}\n".format(critic))
    # save as tuple
    acmodel = (actor,critic)
    # calculate num of model params
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    total_params = actor_params + critic_params
    print('***PARAMS:\nActor {}\nCritic {}\nTotal {}'.format(actor_params,critic_params,total_params))
else:
    if env_type == 'procgen':
        acmodel = ACModel_PROCGEN(obs_space=obs_space["image"], action_space=ACTION_SPACE)
    else:    
        acmodel = ACModel_LSTM(obs_space=obs_dim, action_space=ACTION_SPACE)
    
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    total_params = sum(p.numel() for p in acmodel.parameters())
    print('***PARAMS UNIQUE AC (RIDE):',total_params)

# ******************************************************************************
# Set Imitation Learning
# ******************************************************************************
from torch_ac.utils.intrinsic_motivation import Counter_Global
from torch_ac.utils.ranking_buffer import RankingBuffer

counter_global = Counter_Global()
disable_rapid = args.disable_rapid
sl_num = args.sl_num
sl_batch_size = args.sl_batch_size
sl_buffer_size = args.buffer_size
sl_score_type = args.score_type
observation_space = envs[0].observation_space.shape if env_type=='procgen' else obs_space["image"]
action_space = envs[0].action_space
sl_clipgrad = args.sl_clipgrad

ranking_buffer = RankingBuffer(
                            rank_type=args.rank_type,
                            ob_space = observation_space,
                            ac_space = action_space,
                            buffer_size = sl_buffer_size,
                            score_type = sl_score_type,
                            w0 = args.w0,
                            w1 = args.w1,
                            w2 = args.w2
                            )

# Pre-load the buffer with demos (if required/necessary)
stored_idx_episode = []
if args.load_demos:
    # 1.load buffer for the buffer (ranking buffer)
    ranking_buffer.loadDemonstrationBuffer(path=root_dir+args.load_demos_path+'_episodes_ranking_buffer.npy')
    print('Buffer pre-loaded with Demos!')

    if args.pre_train_epochs==0 or args.load_counts_dict or not disable_rapid:
        # 2. load the states to keep track in the curiosity part related to W2
        with open(root_dir+args.load_demos_path+'_episodes_dict.json', 'r') as f:
            stored_episodes_dict = json.load(f) #deserialize

        stored_idx_episode = np.unique(list(stored_episodes_dict.keys()))
        # conver str list to int list
        stored_idx_episode = [int(float(s)) for s in stored_idx_episode]

        for k,v in stored_episodes_dict.items(): #v=observations,k=episode_idx
            counter_global.add(obs=np.asarray(v),episode_index=int(float(k)))
        print('Episodes (states) for W2 loaded!')

# ******************************************************************************
# Load algorithms
# ******************************************************************************
algo = PPOAlgo( env_type=env_type,
                envs=envs,
                envs_evaluation=envs_evaluation,
                acmodel=acmodel, device =device,
                num_frames_per_proc=args.nsteps,
                discount=args.discount, lr = args.lr,
                gae_lambda = args.gae_lambda,
                entropy_coef = args.entropy_coef,
                value_loss_coef = args.value_loss_coef,
                max_grad_norm = args.max_grad_norm,
                recurrence_actor = args.use_recurrence_actor,
                recurrence_critic = args.use_recurrence_critic,
                adam_eps = args.optim_eps,
                clip_eps = args.clip_eps,
                epochs = args.epochs,
                preprocess_obss = preprocess_obss,
                num_actions = ACTION_SPACE,
                separated_networks = args.separated_networks,
                env_name=env_dict,
                int_coef=args.intrinsic_motivation,
                normalize_int_rewards=args.normalize_intrinsic_bonus,
                im_type=args.im_type,
                use_episodic_counts = args.use_episodic_counts,
                use_only_not_visited = args.use_only_not_visited,
                total_num_frames = int(args.frames),
                sl_clipgrad = sl_clipgrad,
                nminibatch = args.nminibatch,
                num_train_seeds=args.num_train_seeds,
                init_level_seed = args.init_level_seed,
                )

# ******************************************************************************
# Load optimizer
# ******************************************************************************
if "optimizer_state" in status:
    if args.separated_networks:
        algo.optimizer[0].load_state_dict(status["optimizer_state"][0])
        algo.optimizer[1].load_state_dict(status["optimizer_state"][1])
        txt_logger.info("Optimizer loaded\n")
    else:
        algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")

# ******************************************************************************
# Train model
# ******************************************************************************
number_of_seeds_stored = {} # % of samples of each level/seed every update
levels_return_dict = {} # sampled seed and its return in each episode
levels_gae_dict = {} # sampled seed and its gae in each episode
levels_online_return_dict = {}
levels_online_gae_dict = {}
aux_ep_counter = 0
episode_idx_to_level_idx = {}

# criteria so store demos
baseline_rew = [0.9] #[0.1,0.6,0.9]
baseline_steps = [7500000,15000000,24900000] 

# only for early-stopping
max_score = 0
consecutive_updates_same_result = 0

current_buffer_size = 0
do_buffer = True if args.do_buffer or not disable_rapid else False # specify if we use experience replay
rewmean100 = 0 # avg ext return
episode_counter = 0
rapid_loss,grad_rapid = 0,0
num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

# ******************************************************************************
# Pre-training phase
# ******************************************************************************
for e in range(args.pre_train_epochs):

    # idx_buf ,sampled_obs, sampled_acs, sampled_dret, sampled_dones, sampled_nobs, influence_of_scores, seeds_dict
    idx_buf ,sampled_obs, sampled_acs, sampled_dret, sampled_dones, sampled_nobs, influence_of_scores, _ = ranking_buffer.sample_bc(sl_batch_size)
    rl, gr = algo.update_parameters_imitation(lr=args.lr, obs=sampled_obs, next_obs=sampled_nobs, acs=sampled_acs, dones=sampled_dones)

    if e % 100 == 0:
        # evaluate the agent randomly in 100 levels and get the return
        _reset_seed = False if env_type == 'procgen' else True
        sr,avgsteps,avgret = algo.evaluate_agent(num_episodes=args.num_test_seeds,
                                                 reset_seed=_reset_seed)

        # monitorization variables
        header = ["epochs","pre-trainig-rapidloss","pre-train_successratio","pre-train_avgstep","pre-train_avgret"]
        data = [e,rl,sr,avgsteps,avgret]

        txt_logger.info('Epochs:{} | Rapid_loss:{:.4f} | sr:{:.2f} | steps:{:.1f} | return:{:.3f}'.format(*data))

        # overwrite tensorboardX
        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, e)

    if e == args.pre_train_epochs -1:
        print('Pre-train finished!')


# 0. Main loop begins
while num_frames < args.frames:
    # print('Returns',levels_online_return_dict)
    # print('GAE',levels_online_gae_dict)
    update_start_time = time.time()

    # 1. collect a full rollout
    exps, logs1, episode_info = algo.collect_experiences()

    # 2. update PPO after rollout completion (explorative agent)
    if not args.disable_ppo:
        logs2 = algo.update_parameters_ppo(exps)
    else:
        logs2 = {"entropy": 0,
                "policy_loss": 0,
                "value": 0,
                "value_loss": 0,
                "grad_norm": 0,
                "grad_norm_critic": 0}
    # 3.0. extract and reset buffer information
    # calculate how many episodes have been completed in the rollout collection
    completed_episodes = len(episode_info)
    # reset the episode buffer information
    algo.reset_episode_info()
    # 3.1. update Buffer if episode has finished
    if do_buffer:

        # monitore current_rollout new levels
        current_levels = []
        current_episodes = []

        for e in range(completed_episodes):
            while True:
                aux_ep_counter += 1
                if aux_ep_counter not in stored_idx_episode:
                    break
            # get all the observations and actions of the episode and store them in the buffer
            obs_np = episode_info[e]['obs']
            acs_np = episode_info[e]['acs']
            rew_np = episode_info[e]['rew']
            nobs_np = episode_info[e]['nobs']
            dones_np = episode_info[e]['dones']
            ret_np = episode_info[e]['ret']
            lev_np = episode_info[e]['seed'] 
            steps_np = episode_info[e]['steps']
            gae_np = episode_info[e]['gae_score']
            # mem_np = episode_info[e]['mem'] if False else []
            mem_np=[]

            # update episodes monitorization dict - actual performance(return AKA steps)
            levels_return_dict[aux_ep_counter] = {lev_np:ret_np}
            levels_online_return_dict[lev_np] = ret_np
            # update episodes monitorization dict - gae(regret)
            levels_gae_dict[aux_ep_counter] = {lev_np:gae_np}
            levels_online_gae_dict[lev_np] = gae_np

            # update current levels (this rollout)
            current_levels.append(lev_np)
            current_episodes.append(aux_ep_counter)

            # update/train global bonus --> global scope
            if args.w2>0:
                reshaped_obs = obs_np.astype(float).reshape(obs_np.shape[0],-1)
                counter_global.add(reshaped_obs,aux_ep_counter)

            # insert values in buffer
            ranking_buffer.insert(obs=obs_np,acs=acs_np,rew=rew_np,nobs=nobs_np,dones=dones_np,
                                  ret=ret_np,level=lev_np,steps=steps_np,episode_idx=aux_ep_counter,
                                  mem=mem_np)



        if ranking_buffer.getIndex()>0: # if data is empty, there is nothing to sort
            # sort and drop
            ranking_buffer.sort_and_drop(current_episodes=current_episodes,
                                         current_levels=current_levels,
                                         counter_global=counter_global,
                                         online_return=levels_online_return_dict,
                                         online_gae=levels_online_gae_dict)
            # keep track just of stored tuples
            if args.w2 >0:
                used_ep_idx = ranking_buffer.getIdxEpisodesUsed()
                counter_global.updateEpisodeBuffers(used_ep_idx)

        # update buffer size for sampling
        current_buffer_size = ranking_buffer.getIndex()
    else: # do not do_buffer
        for e in range(completed_episodes):
            ret_np = episode_info[e]['ret']
            lev_np = episode_info[e]['seed']
            gae_np = episode_info[e]['gae_score'] 
            # update episodes monitorization dict - actual performance(return AKA steps)
            levels_return_dict[aux_ep_counter] = {lev_np:ret_np}
            levels_online_return_dict[lev_np] = ret_np
            # update episodes monitorization dict - gae(regret)
            levels_gae_dict[aux_ep_counter] = {lev_np:gae_np}
            levels_online_gae_dict[lev_np] = gae_np

    # 3.2. Imitation Learning with Behavioral Cloning - sample data from the buffer & train
    # if not disable_rapid and update % args.ratio_offpolicy_update == 0 and current_buffer_size > 0:
    if not disable_rapid and update % args.ratio_offpolicy_update == 0 and current_buffer_size > 0:

        rapid_loss, grad_rapid = [],[]
        w0_score,w1_score,w2_score=[],[],[]
        for ep in range(sl_num):
            
            idx_buf ,sampled_obs, sampled_acs, sampled_dret, sampled_dones, sampled_nobs, influence_of_scores, seeds_dict = ranking_buffer.sample_bc(batch_size=sl_batch_size,sample_full_traj=args.use_full_traj)
            number_of_seeds_stored[num_frames] = seeds_dict
            rl, gr = algo.update_parameters_imitation(lr=args.lr, obs=sampled_obs, next_obs=sampled_nobs, acs=sampled_acs, dones=sampled_dones, values_il_target=[])

            w0_score.append(influence_of_scores['w0'])
            w1_score.append(influence_of_scores['w1'])
            w2_score.append(influence_of_scores['w2'])
            rapid_loss.append(rl)
            grad_rapid.append(gr)

        w0_score=np.mean(w0_score)
        w1_score=np.mean(w1_score)
        w2_score=np.mean(w2_score)
        rapid_loss = np.mean(rapid_loss)
        grad_rapid = np.mean(grad_rapid)
    else:
        w0_score,w1_score,w2_score=0,0,0
        rapid_loss, grad_rapid = 0,0

    # 4. update logs
    additional_logs = {"w0_score":w0_score,
                       "w1_score":w1_score,
                       "w2_score":w2_score,
                       "rapid_loss":rapid_loss,
                       "grad_rapid":grad_rapid,
                       "current_buffer_size":current_buffer_size}
    logs = {**logs1, **logs2,**additional_logs}

    update_end_time = time.time()
    num_frames += logs["num_frames"]
    update += 1


    ##############################
    # Condition for storing DEMONSTRATIONS things!
    ##############################
    if env_type == 'procgen':
        br = baseline_steps[0] if len(baseline_steps)>0 else 100e7
            
        if args.store_demos and num_frames>br:
            # toggle (store just once)
            baseline_steps.pop(0)
            # 1. Store buffer demos
            ranking_buffer.saveDemonstrationBuffer(path = numpyfiles_dir + '_steps' + str(br) + '_episodes_ranking_buffer.npy')
            # 2. store episode states
            episodes_dict = counter_global.get_EpisodeBuffer()
            with open(numpyfiles_dir + '_steps' + str(br) + '_episodes_dict.json', 'w') as f:
                json.dump(episodes_dict, f)
            print('STORED BUFFERS')    
    else:
        br = baseline_rew[0] if len(baseline_rew)>0 else 1000

        if args.store_demos and rewmean100>br:
            # toggle (store just once)
            baseline_rew.pop(0)
            # 1. Store buffer demos
            ranking_buffer.saveDemonstrationBuffer(path = numpyfiles_dir + '_avgret' + str(br) + '_episodes_ranking_buffer.npy')
            # 2. store episode states
            episodes_dict = counter_global.get_EpisodeBuffer()
            with open(numpyfiles_dir + '_avgret' + str(br) + '_episodes_dict.json', 'w') as f:
                json.dump(episodes_dict, f)
            print('STORED BUFFERS')    

    # 4.1 Print logs
    if update % args.log_interval == 0:

        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        episodes = logs["episode_counter"]

        # intrinsic
        return_int_per_episode = utils.synthesize(logs["return_int_per_episode"])
        return_int__norm_per_episode = utils.synthesize(logs["return_int_per_episode_norm"])
        # extrinsic
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])


        # general values
        header = ["update", "frames", "FPS", "duration","episodes"]
        data = [update, num_frames, fps, duration, episodes]
        only_txt = [update, num_frames, fps, duration, episodes]

        # add beta coef
        header += ["buffer_size"]
        data += [logs["current_buffer_size"]]
        only_txt += [logs["current_buffer_size"]]

        # avg 100 episodes
        header += ["avg_success","avg_return","avg_return_int","avg_steps"]
        rewmean100 = logs["avg_return"] # used also at main loop!
        data += [logs["avg_success"],logs["avg_return"],logs["avg_return_int"],logs["avg_steps"]]
        only_txt += [logs["avg_success"],logs["avg_return"],logs["avg_return_int"],logs["avg_steps"]]

        # returns
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        only_txt += [rreturn_per_episode["mean"]]
        only_txt += [rreturn_per_episode["std"]]

        header += ["return_int_" + key for key in return_int_per_episode.keys()]
        data += return_int_per_episode.values()
        only_txt += [return_int_per_episode["mean"]]
        only_txt += [return_int_per_episode["std"]]


        # evaluate the agent randomly in 100 levels and get the return
        if env_type=='procgen':
            sr,avgsteps,avgret = algo.evaluate_agent(num_episodes=args.num_test_seeds,reset_seed=False)
        else:
            sr,avgsteps,avgret = 0,0,0 # in the case of minigrid, as we are using enough levels, we use the rewards from the environment as evaluation (generalization gap releated)
        header += ["ev_success","ev_return","ev_steps"]
        data += [sr,avgret,avgsteps]
        only_txt += [sr,avgret,avgsteps]

        header += ["entropy", "value", "policy_loss", "value_loss", "rapid_loss", "grad_norm", "grad_norm_critic","grad_rapid"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["rapid_loss"], logs["grad_norm"], logs["grad_norm_critic"],logs["grad_rapid"]]
        only_txt += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["rapid_loss"], logs["grad_norm"], logs["grad_norm_critic"],logs["grad_rapid"]]

        header+= ["w0","w1","w2","buffer_size","normalization_int_score","predominance_ext_over_int"]
        data += [logs["w0_score"],logs["w1_score"],logs["w2_score"],logs["current_buffer_size"],logs["normalization_int_score"],logs["predominance_ext_over_int"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | Eps {} | len(B):{} | SR: {:.2f} | RΩ: {:.3f} | RiΩ: {:.3f} | Steps: {:.1f} |rR:μσ {:.3f} {:.2f} | rRi:μσ {:.3f} {:.2f} | EvSR: {:.2f} | EvRΩ: {:.3f} | EvSteps: {:.1f} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | rL: {:.3f} | ∇p {:.3f} | ∇c {:.3f} | ∇r {:.3f}"
            .format(*only_txt))

        # overwrite csv file
        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        # overwrite tensorboardX
        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

        # for early stopping
        max_score = rewmean100 if max_score<rewmean100 else max_score
        consecutive_updates_same_result = consecutive_updates_same_result+1 if max_score==rewmean100 else 0

    # 4.2. Save status
    if (args.save_interval > 0) and (update % args.save_interval == 0):
        if args.separated_networks:
            acmodel_weights = (acmodel[0].state_dict(),
                                acmodel[1].state_dict())
            optimizer_state = (algo.optimizer[0].state_dict(),
                                algo.optimizer[1].state_dict())
        else:
            acmodel_weights = acmodel.state_dict()
            optimizer_state = algo.optimizer.state_dict()
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel_weights, "optimizer_state": optimizer_state}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")

        import pickle
        # store the seeds usage in the buffer
        with open(model_dir + '/number_of_seeds_stored.pkl', 'wb') as file:
            pickle.dump(number_of_seeds_stored, file)
        # store experienced seed and the return in each episode
        with open(model_dir + '/episodes_return.pkl', 'wb') as file:
            pickle.dump(levels_return_dict, file)
        with open(model_dir + '/episodes_gae.pkl', 'wb') as file:
            pickle.dump(levels_gae_dict, file)

    # Early stopping option
    if consecutive_updates_same_result>=5 and args.early_stopping:
        txt_logger.info('\nTrain finished')
        # file that I use as boolean to if finished
        fin_path = os.path.join(model_dir, "1.FINISHED.csv")
        utils.create_folders_if_necessary(fin_path)
        fin_file = open(fin_path, "a")
        break

#Close environments
for i in range(args.procs):
    envs[i].close()
