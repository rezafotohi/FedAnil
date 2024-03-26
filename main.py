import os
import sys
import argparse
import numpy as np
import random
import time
from datetime import datetime
import copy
from sys import getsizeof
import sqlite3
import pickle
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from Models import ConcatModel, CombinedModel
from Enterprise import Enterprise, EnterprisesInNetwork
# FedAnil: Consortium Blockchain
from Block import Block
# FedAnil: Consortium Blockchain
from Consortium_Blockchain import Consortium_Blockchain
import warnings
warnings.filterwarnings('ignore')

# set program execution time for logging purpose
date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_files_folder_path = f"logs/{date_time}"
NETWORK_SNAPSHOTS_BASE_FOLDER = "snapshots"
from Enterprise import flcnt, lastprc
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Block_FedAvg_Simulation")

# debug attributes
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-v', '--verbose', type=int, default=1, help='print verbose debug log')
parser.add_argument('-sn', '--save_network_snapshots', type=int, default=0, help='only save network_snapshots if this is set to 1; will create a folder with date in the snapshots folder')
parser.add_argument('-dtx', '--destroy_tx_in_block', type=int, default=0, help='currently transactions stored in the blocks are occupying GPU ram and have not figured out a way to move them to CPU ram or harddisk, so turn it on to save GPU ram in order for PoS to run 100+ rounds. NOT GOOD if there needs to perform chain resyncing.')
parser.add_argument('-rp', '--resume_path', type=str, default=None, help='resume from the path of saved network_snapshots; only provide the date')
parser.add_argument('-sf', '--save_freq', type=int, default=5, help='save frequency of the network_snapshot')
parser.add_argument('-sm', '--save_most_recent', type=int, default=2, help='in case of saving space, keep only the recent specified number of snapshops; 0 means keep all')

# FL attributes
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='OARF', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, use value from origin paper as default")
parser.add_argument('-op', '--optimizer', type=str, default="SGD", help='optimizer to be used, by default implementing stochastic gradient descent')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to enterprises')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=100, help='maximum number of communication rounds, may terminate early if converges')
parser.add_argument('-nd', '--num_enterprises', type=int, default=20, help='numer of the enterprises in the simulation network')
parser.add_argument('-st', '--shard_test_data', type=int, default=0, help='it is easy to see the global models are consistent across enterprises when the test dataset is NOT sharded')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help="number of malicious enterprises in the network. Malicious Enterprises data sets will be introduced Gaussian noise")
parser.add_argument('-nv', '--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
parser.add_argument('-le', '--default_local_epochs', type=int, default=5, help='local train epoch. Train local model by this same num of epochs for each local_enterprise, if -mt is not specified')
# FedAnil: Consortium_blockchain system consensus attributes
parser.add_argument('-ur', '--unit_reward', type=int, default=1, help='unit reward for providing data, verification of signature, validation and so forth')
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=6, help="a local_enterprise or validator enterprise is kicked out of the enterprise's peer list(put in black list) if it's identified as malicious for this number of rounds")
parser.add_argument('-lo', '--lazy_local_enterprise_knock_out_rounds', type=int, default=10, help="a local_enterprise enterprise is kicked out of the enterprise's peer list(put in black list) if it does not provide updates for this number of rounds, due to too slow or just lazy to do updates and only accept the model udpates.(do not care lazy validator or miner as they will just not receive rewards)")
parser.add_argument('-pow', '--pow_difficulty', type=int, default=0, help="if set to 0, meaning miners are using PoS")

# FedAnil: Consortium_blockchain FL validator/miner restriction tuning parameters
parser.add_argument('-mt', '--miner_acception_wait_time', type=float, default=0.0, help="default time window for miners to accept transactions, in seconds. 0 means no time limit, and each enterprise will just perform same amount(-le) of epochs per round like in FedAvg paper")
parser.add_argument('-ml', '--miner_accepted_transactions_size_limit', type=float, default=0.0, help="no further transactions will be accepted by miner after this limit. 0 means no size limit. either this or -mt has to be specified, or both. This param determines the final block_size")
parser.add_argument('-mp', '--miner_pos_propagated_block_wait_time', type=float, default=float("inf"), help="this wait time is counted from the beginning of the comm round, used to simulate forking events in PoS")
parser.add_argument('-vh', '--validator_threshold', type=float, default=1.0, help="a threshold value of accuracy difference to determine malicious local_enterprise")
parser.add_argument('-md', '--malicious_updates_discount', type=float, default=0.0, help="do not entirely drop the voted negative local_enterprise transaction because that risks the same local_enterprise dropping the entire transactions and repeat its accuracy again and again and will be kicked out. Apply a discount factor instead to the false negative local_enterprise's updates are by some rate applied so it won't repeat")
parser.add_argument('-mv', '--malicious_validator_on', type=int, default=0, help="let malicious validator flip voting result")


# distributed system attributes
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='the odds a enterprise is online')
parser.add_argument('-els', '--even_link_speed_strength', type=int, default=1, help="This variable is used to simulate transmission delay. Default value 1 means every enterprise is assigned to the same link speed strength -dts bytes/sec. If set to 0, link speed strength is randomly initiated between 0 and 1, meaning a enterprise will transmit  -els*-dts bytes/sec - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-dts', '--base_data_transmission_speed', type=float, default=70000.0, help="volume of data can be transmitted per second when -els == 1. set this variable to determine transmission speed (bandwidth), which further determines the transmission delay - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-ecp', '--even_computation_power', type=int, default=1, help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value 1 means evenly assign computation power to 1. If set to 0, power is randomly initiated as an int between 0 and 4, both included.")

# simulation attributes
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help="hard assign number of roles in the network, order by local_enterprise, validator and miner. e.g. 12,5,3 assign 12 local_enterprises, 5 validators and 3 miners. \"*,*,*\" means completely random role-assigning in each communication round ")
parser.add_argument('-aio', '--all_in_one', type=int, default=1, help='let all nodes be aware of each other in the network while registering')
parser.add_argument('-cs', '--check_signature', type=int, default=1, help='if set to 0, all signatures are assumed to be verified to save execution time')
parser.add_argument('-at', '--attack_type', type=str, default='', help='set the attack type used for attack simulation')
parser.add_argument('-ta', '--target_acc', type=float, default=0.9, help='set the target accuracy for end simulation')
# parser.add_argument('-la', '--least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')



if __name__=="__main__":
	# create logs/ if not exists
	if not os.path.exists('logs'):
		os.makedirs('logs')

	# get arguments
	args = parser.parse_args()
	args = args.__dict__
	
	# detect CUDA
	dev = torch.device("cpu") # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# pre-define system variables
	latest_round_num = 0

	''' If network_snapshot is specified, continue from left '''
	if args['resume_path']:
		if not args['save_network_snapshots']:
			print("NOTE: save_network_snapshots is set to 0. New network_snapshots won't be saved by conituing.")
		network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{args['resume_path']}"
		latest_network_snapshot_file_name = sorted([f for f in os.listdir(network_snapshot_save_path) if not f.startswith('.')], key = lambda fn: int(fn.split('_')[-1]) , reverse=True)[0]
		print(f"Loading network snapshot from {args['resume_path']}/{latest_network_snapshot_file_name}")
		print("BE CAREFUL - loaded dev env must be the same as the current dev env, namely, cpu, gpu or gpu parallel")
		latest_round_num = int(latest_network_snapshot_file_name.split('_')[-1])
		enterprises_in_network = pickle.load(open(f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", "rb"))
		enterprises_list = list(enterprises_in_network.enterprises_set.values())
		log_files_folder_path = f"logs/{args['resume_path']}"
		# for colab
		# log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{args['resume_path']}"
		# original arguments file
		args_used_file = f"{log_files_folder_path}/args_used.txt"
		file = open(args_used_file,"r") 
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			# abide by the original specified rewards
			if line.startswith('--unit_reward'):
				rewards = int(line.split(" ")[-1])
			# get number of roles
			if line.startswith('--hard_assign'):
				roles_requirement = line.split(" ")[-1].split(',')
			# get mining consensus
			if line.startswith('--pow_difficulty'):
				mining_consensus = 'PoW' if int(line.split(" ")[-1]) else 'PoS'
		# determine roles to assign
		try:
			local_enterprises_needed = int(roles_requirement[0])
		except:
			local_enterprises_needed = 1
		try:
			validators_needed = int(roles_requirement[1])
		except:
			validators_needed = 1
		try:
			miners_needed = int(roles_requirement[2])
		except:
			miners_needed = 1
	else:
		''' SETTING UP FROM SCRATCH'''
		
		# 0. create log_files_folder_path if not resume
		os.mkdir(log_files_folder_path)

		# 1. save arguments used
		with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
			f.write("Command line arguments used -\n")
			f.write(' '.join(sys.argv[1:]))
			f.write("\n\nAll arguments used -\n")
			for arg_name, arg in args.items():
				f.write(f'\n--{arg_name} {arg}')
				
		# 2. create network_snapshot folder
		if args['save_network_snapshots']:
			network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{date_time}"
			os.mkdir(network_snapshot_save_path)

		# 3. assign system variables
		# for demonstration purposes, this reward is for every rewarded action
		rewards = args["unit_reward"]
		
		# 4. get number of roles needed in the network
		roles_requirement = args['hard_assign'].split(',')
		# determine roles to assign
		try:
			local_enterprises_needed = int(roles_requirement[0])
		except:
			local_enterprises_needed = 1
		try:
			validators_needed = int(roles_requirement[1])
		except:
			validators_needed = 1
		try:
			miners_needed = int(roles_requirement[2])
		except:
			miners_needed = 1

		# 5. check arguments eligibility

		num_enterprises = args['num_enterprises']
		num_malicious = args['num_malicious']
		
		if num_enterprises < local_enterprises_needed + miners_needed + validators_needed:
			sys.exit("ERROR: Roles assigned to the enterprises exceed the maximum number of allowed enterprises in the network.")

		if num_enterprises < 3:
			sys.exit("ERROR: There are not enough enterprises in the network.\n The system needs at least one miner, one local_enterprise and/or one validator to start the operation.\nSystem aborted.")

		
		if num_malicious:
			if num_malicious > num_enterprises:
				sys.exit("ERROR: The number of malicious enterprises cannot exceed the total number of enterprises set in this network")
			else:
				print(f"Malicious enterprises vs total enterprises set to {num_malicious}/{num_enterprises} = {(num_malicious/num_enterprises)*100:.2f}%")

		# 6. create neural net based on the input model name
		#net = None
		#net = ConcatModel()
		net = CombinedModel()
		#if args['model_name'] == 'cnn':
		#	net = CNN()
		#elif args['model_name'] == 'OARF':
		#	net = ConcatModel()

		# 7. assign GPU(s) if available to the net, otherwise CPU
		# os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
		device = "mps" if torch.backends.mps.is_available() else "cpu"
		print(f"Using device: {device}")
		#if torch.cuda.device_count() > 1 :#or device:
		#	net = torch.nn.DataParallel(net)
		#print(f"{torch.cuda.device_count()} GPUs are available to use!")
		# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
		print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
		print(f"Is MPS available? {torch.backends.mps.is_available()}")
		net = net.to(dev)

		num_mps_devices = len(device)
		print("Number of MPS devices: ", num_mps_devices)
		print("\n")


		# 8. set loss_function
		loss_func = F.cross_entropy

		# 9. create enterprises in the network
		enterprises_in_network = EnterprisesInNetwork(data_set_name='femnist', is_iid=args['IID'], batch_size = args['batchsize'], learning_rate =  args['learning_rate'], loss_func = loss_func, opti = args['optimizer'], num_enterprises=num_enterprises, network_stability=args['network_stability'], net=net, dev=dev, knock_out_rounds=args['knock_out_rounds'], lazy_local_enterprise_knock_out_rounds=args['lazy_local_enterprise_knock_out_rounds'], shard_test_data=args['shard_test_data'], miner_acception_wait_time=args['miner_acception_wait_time'], miner_accepted_transactions_size_limit=args['miner_accepted_transactions_size_limit'], validator_threshold=args['validator_threshold'], pow_difficulty=args['pow_difficulty'], even_link_speed_strength=args['even_link_speed_strength'], base_data_transmission_speed=args['base_data_transmission_speed'], even_computation_power=args['even_computation_power'], malicious_updates_discount=args['malicious_updates_discount'], num_malicious=num_malicious, noise_variance=args['noise_variance'], check_signature=args['check_signature'], not_resync_chain=args['destroy_tx_in_block'])
		del net
		enterprises_list = list(enterprises_in_network.enterprises_set.values())

		# 10. register enterprises and initialize global parameterms
		for enterprise in enterprises_list:
			# set initial global weights
			enterprise.init_global_parameters()
			# helper function for registration simulation - set enterprises_list and aio
			enterprise.set_enterprises_dict_and_aio(enterprises_in_network.enterprises_set, args["all_in_one"])
			# simulate peer registration, with respect to enterprise idx order
			enterprise.register_in_the_network()
		# remove its own from peer list if there is
		for enterprise in enterprises_list:
			enterprise.remove_peers(enterprise)

		# 11. build logging files/database path
		# create log files
		open(f"{log_files_folder_path}/correctly_kicked_local_enterprises.txt", 'w').close()
		open(f"{log_files_folder_path}/mistakenly_kicked_local_enterprises.txt", 'w').close()
		open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'w').close()
		open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'w').close()
		# open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'w').close()
		# open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'w').close()
		open(f"{log_files_folder_path}/kicked_lazy_local_enterprises.txt", 'w').close()

		# 12. setup the mining consensus
		mining_consensus = 'PoW' if args['pow_difficulty'] else 'PoS'

	# create malicious local_enterprise identification database
	conn = sqlite3.connect(f'{log_files_folder_path}/malicious_enterprise_identifying_log.db')
	conn_cursor = conn.cursor()
	conn_cursor.execute("""CREATE TABLE if not exists  malicious_local_enterprises_log (
	enterprise_seq text,
	if_malicious integer,
	correctly_identified_by text,
	incorrectly_identified_by text,
	in_round integer,
	when_resyncing text
	)""")

	target_accuracy = args['target_acc']

	# FedAnil: Total Communication Cost (Bytes)
	communication_bytes_sum = 0
	# FedAnil: Total Computation Cost (Seconds)
	computation_sum = 0
	# FedAnil: Total Accuracy (%)
	total_accuracy = 0
	# FedAnil starts here
	for comm_round in range(latest_round_num + 1, args['max_num_comm']+1):
		communication_bytes_per_round = 0
		# create round specific log folder
		log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
		if os.path.exists(log_files_folder_path_comm_round):
			print(f"Deleting {log_files_folder_path_comm_round} and create a new one.")
			shutil.rmtree(log_files_folder_path_comm_round)
		os.mkdir(log_files_folder_path_comm_round)
		# free cuda memory
		if dev == torch.device("cuda"):
			with torch.cuda.device('cuda'):
				torch.cuda.empty_cache()
		print(f"\nCommunication round {comm_round}")
		# FedAnil: Total Computation Cost
		comm_round_start_time = time.time()
		# (RE)ASSIGN ROLES
		local_enterprises_to_assign = local_enterprises_needed
		miners_to_assign = miners_needed
		validators_to_assign = validators_needed
		local_enterprises_this_round = []
		miners_this_round = []
		validators_this_round = []
		random.shuffle(enterprises_list)
		enterprises_list.sort(key=lambda x: x.rewards, reverse=False)
		for enterprise in enterprises_list:
			enterprise.reset_last()
			if local_enterprises_to_assign:
				enterprise.assign_local_enterprise_role()
				local_enterprises_to_assign -= 1
			elif validators_to_assign:
				enterprise.assign_validator_role()
				validators_to_assign -= 1
			elif miners_to_assign:
				enterprise.assign_miner_role()
				miners_to_assign -= 1
			else:
				enterprise.assign_role()
			if enterprise.return_role() == 'local_enterprise':
				local_enterprises_this_round.append(enterprise)
			elif enterprise.return_role() == 'miner':
				miners_this_round.append(enterprise)
			else:
				validators_this_round.append(enterprise)
			# determine if online at the beginning (essential for step 1 when local_enterprise needs to associate with an online enterprise)
			enterprise.online_switcher()

		# re-init round vars - in real distributed system, they could still fall behind in comm round, but here we assume they will all go into the next round together, thought enterprise may go offline somewhere in the previous round and their variables were not therefore reset
		for miner in miners_this_round:
			miner.miner_reset_vars_for_new_round()
		for local_enterprise in local_enterprises_this_round:
			local_enterprise.local_enterprise_reset_vars_for_new_round()
		for validator in validators_this_round:
			validator.validator_reset_vars_for_new_round()

		# DOESN'T MATTER ANY MORE AFTER TRACKING TIME, but let's keep it - orginal purpose: shuffle the list(for local_enterprise, this will affect the order of dataset portions to be trained)
		random.shuffle(local_enterprises_this_round)
		random.shuffle(miners_this_round)
		random.shuffle(validators_this_round)

		# selected miners 
		selected_miners = miners_this_round
		stake_of_miners = []
		for it in range(len(selected_miners)):
			stake_of_miners.append(selected_miners[it].return_stake())
		index_max = stake_of_miners.index(max(stake_of_miners))
		leader_miner = selected_miners[index_max]
		
		random_selection_num = random.randrange(int(len(local_enterprises_this_round) * 0.8), int(len(local_enterprises_this_round) * 1.0))

		''' local_enterprises, validators and miners take turns to perform jobs '''
		selected_local_enterprises_this_round = local_enterprises_this_round[0:random_selection_num]
		print(f"SELECTION : {random_selection_num} of {local_enterprises_needed}")
		print(''' Step 1 - local_enterprises assign associated miner and validator (and do local updates, but it is implemented in code block of step 2) \n''')
		# FedAnil: Select Random numbers from enterprises
		
		''' DEBUGGING CODE '''
		if args['verbose']:

			# show enterprises initial chain length and if online
			for enterprise in selected_local_enterprises_this_round:
				if enterprise.is_online():
					print(f'{enterprise.return_idx()} {enterprise.return_role()} online - ', end='')
				else:
					print(f'{enterprise.return_idx()} {enterprise.return_role()} offline - ', end='')
				# debug chain length
				print(f"chain length {enterprise.return_consortium_blockchain_object().return_chain_length()}")
		
			# show enterprise roles
			print(f"\nThere are {len(selected_local_enterprises_this_round)} local_enterprises, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")
			print("\nLocal_enterprises this round are")
			for local_enterprise in selected_local_enterprises_this_round:
				print(f"e_{local_enterprise.return_idx().split('_')[-1]} online - {local_enterprise.is_online()} with chain len {local_enterprise.return_consortium_blockchain_object().return_chain_length()}")
			print("\nMiners this round are")
			for miner in miners_this_round:
				print(f"e_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_consortium_blockchain_object().return_chain_length()}")
			print("\nValidators this round are")
			for validator in validators_this_round:
				print(f"e_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_consortium_blockchain_object().return_chain_length()}")
			print()

			# show peers with round number
			print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
			for enterprise_seq, enterprise in enterprises_in_network.enterprises_set.items():
				peers = enterprise.return_peers()
				print(f"e_{enterprise_seq.split('_')[-1]} - {enterprise.return_role()[0]} has peer list ", end='')
				for peer in peers:
					print(f"e_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
				print()
			print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")

		''' DEBUGGING CODE ENDS '''

		#for local_enterprise_iter in range(len(local_enterprises_this_round)):
		for local_enterprise_iter in range(len(selected_local_enterprises_this_round)):
			#local_enterprise = local_enterprises_this_round[local_enterprise_iter]
			local_enterprise = selected_local_enterprises_this_round[local_enterprise_iter]
			# FedAnil: fetch global model from consortium blockchain
			local_enterprise.fetch_global_model(local_enterprise.return_consortium_blockchain_object())
			# resync chain(block could be dropped due to fork from last round)
			if local_enterprise.resync_chain(mining_consensus):
				local_enterprise.update_model_after_chain_resync(log_files_folder_path_comm_round, conn, conn_cursor)
			# FedAnil: Total Communication Cost (Bytes): Transfer of Global Model Bytes from Server to Clients
			communication_bytes_per_round += sys.getsizeof(local_enterprise.global_parameters)
			# local_enterprise (should) perform local update and associate
			#print(f"{local_enterprise.return_idx()} - local_enterprise {local_enterprise_iter+1}/{len(local_enterprises_this_round)} will associate with a validator and a miner, if online...")
			print(f"{local_enterprise.return_idx()} - local_enterprise {local_enterprise_iter+1}/{len(selected_local_enterprises_this_round)} will associate with a validator and a miner, if online...")
			# local_enterprise associates with a miner to accept finally mined block
			if local_enterprise.online_switcher():
				associated_miner = local_enterprise.associate_with_enterprise("miner")
				if associated_miner:
					associated_miner.add_enterprise_to_association(local_enterprise)
				else:
					print(f"Cannot find a qualified miner in {local_enterprise.return_idx()} peer list.")
			# local_enterprise associates with a validator to send local_enterprise transactions
			if local_enterprise.online_switcher():
				associated_validator = local_enterprise.associate_with_enterprise("validator")
				if associated_validator:
					associated_validator.add_enterprise_to_association(local_enterprise)
				else:
					print(f"Cannot find a qualified validator in {local_enterprise.return_idx()} peer list.")
		
		print(''' Step 2 - validators accept local updates and broadcast to other validators in their respective peer lists (local_enterprises local_updates() are called in this step.\n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			# resync chain
			if validator.resync_chain(mining_consensus):
				validator.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
			communication_bytes_per_round += sys.getsizeof(validator.global_parameters)
			# associate with a miner to send post validation transactions
			if validator.online_switcher():
				associated_miner = validator.associate_with_enterprise("miner")
				if associated_miner:
					associated_miner.add_enterprise_to_association(validator)
				else:
					print(f"Cannot find a qualified miner in validator {validator.return_idx()} peer list.")
			# validator accepts local updates from its local_enterprises association
			associated_local_enterprises = list(validator.return_associated_local_enterprises())
			if not associated_local_enterprises:
				print(f"No local_enterprises are associated with validator {validator.return_idx()} {validator_iter+1}/{len(validators_this_round)} for this communication round.")
				continue
			validator_link_speed = validator.return_link_speed()
			print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is accepting local_enterprises' updates with link speed {validator_link_speed} bytes/s, if online...")
			# records_dict used to record transmission delay for each epoch to determine the next epoch updates arrival time
			records_dict = dict.fromkeys(associated_local_enterprises, None)
			for local_enterprise, _ in records_dict.items():
				records_dict[local_enterprise] = {}
			# used for arrival time easy sorting for later validator broadcasting (and miners' acception order)
			transaction_arrival_queue = {}
			# local_enterprises local_updates() called here as their updates transmission may be restrained by miners' acception time and/or size
			if args['miner_acception_wait_time']:
				print(f"miner wait time is specified as {args['miner_acception_wait_time']} seconds. let each local_enterprise do local_updates till time limit")
				for local_enterprise_iter in range(len(associated_local_enterprises)):
					local_enterprise = associated_local_enterprises[local_enterprise_iter]
					if not local_enterprise.return_idx() in validator.return_black_list():
						# TODO here, also add print() for below miner's validators
						print(f'local_enterprise {local_enterprise_iter+1}/{len(associated_local_enterprises)} of validator {validator.return_idx()} is doing local updates')	 
						total_time_tracker = 0
						update_iter = 1
						local_enterprise_link_speed = local_enterprise.return_link_speed()
						lower_link_speed = validator_link_speed if validator_link_speed < local_enterprise_link_speed else local_enterprise_link_speed
						while total_time_tracker < validator.return_miner_acception_wait_time():
							# simulate the situation that local_enterprise may go offline during model updates transmission to the validator, based on per transaction
							if local_enterprise.online_switcher():
								# local_enterprise local update
								local_update_spent_time = local_enterprise.local_enterprise_local_update(rewards, log_files_folder_path_comm_round, comm_round)
								unverified_transaction = local_enterprise.return_local_updates_and_signature(comm_round)
								# size in bytes, usually around 35000 bytes per transaction
								communication_bytes_per_round += local_enterprise.size_of_encoded_data
								unverified_transactions_size = getsizeof(str(unverified_transaction))
								transmission_delay = unverified_transactions_size/lower_link_speed
								if local_update_spent_time + transmission_delay > validator.return_miner_acception_wait_time():
									# last transaction sent passes the acception time window
									break
								records_dict[local_enterprise][update_iter] = {}
								records_dict[local_enterprise][update_iter]['local_update_time'] = local_update_spent_time
								records_dict[local_enterprise][update_iter]['transmission_delay'] = transmission_delay
								records_dict[local_enterprise][update_iter]['local_update_unverified_transaction'] = unverified_transaction
								records_dict[local_enterprise][update_iter]['local_update_unverified_transaction_size'] = unverified_transactions_size
								if update_iter == 1:
									total_time_tracker = local_update_spent_time + transmission_delay
								else:
									total_time_tracker = total_time_tracker - records_dict[local_enterprise][update_iter - 1]['transmission_delay'] + local_update_spent_time + transmission_delay
								records_dict[local_enterprise][update_iter]['arrival_time'] = total_time_tracker
								if validator.online_switcher():
									# accept this transaction only if the validator is online
									print(f"validator {validator.return_idx()} has accepted this transaction.")
									transaction_arrival_queue[total_time_tracker] = unverified_transaction
								else:
									print(f"validator {validator.return_idx()} offline and unable to accept this transaction")
							else:
								# local_enterprise goes offline and skip updating for one transaction, wasted the time of one update and transmission
								wasted_update_time, wasted_update_params = local_enterprise.waste_one_epoch_local_update_time(args['optimizer'])
								wasted_update_params_size = getsizeof(str(wasted_update_params))
								wasted_transmission_delay = wasted_update_params_size/lower_link_speed
								if wasted_update_time + wasted_transmission_delay > validator.return_miner_acception_wait_time():
									# wasted transaction "arrival" passes the acception time window
									break
								records_dict[local_enterprise][update_iter] = {}
								records_dict[local_enterprise][update_iter]['transmission_delay'] = transmission_delay
								if update_iter == 1:
									total_time_tracker = wasted_update_time + wasted_transmission_delay
									print(f"local_enterprise goes offline and wasted {total_time_tracker} seconds for a transaction")
								else:
									total_time_tracker = total_time_tracker - records_dict[local_enterprise][update_iter - 1]['transmission_delay'] + wasted_update_time + wasted_transmission_delay
							update_iter += 1
			else:
				# did not specify wait time. every associated local_enterprise perform specified number of local epochs
				for local_enterprise_iter in range(len(associated_local_enterprises)):
					local_enterprise = associated_local_enterprises[local_enterprise_iter]
					if not local_enterprise.return_idx() in validator.return_black_list():
						print(f'local_enterprise {local_enterprise_iter+1}/{len(associated_local_enterprises)} of validator {validator.return_idx()} is doing local updates')	 
						if local_enterprise.online_switcher():
							local_update_spent_time = local_enterprise.local_enterprise_local_update(rewards, log_files_folder_path_comm_round, comm_round, local_epochs=args['default_local_epochs'])
							local_enterprise_link_speed = local_enterprise.return_link_speed()
							lower_link_speed = validator_link_speed if validator_link_speed < local_enterprise_link_speed else local_enterprise_link_speed
							unverified_transaction = local_enterprise.return_local_updates_and_signature(comm_round)
							unverified_transactions_size = getsizeof(str(unverified_transaction))
							transmission_delay = unverified_transactions_size/lower_link_speed
							if validator.online_switcher():
								transaction_arrival_queue[local_update_spent_time + transmission_delay] = unverified_transaction
								print(f"validator {validator.return_idx()} has accepted this transaction.")
							else:
								print(f"validator {validator.return_idx()} offline and unable to accept this transaction")
						else:
							print(f"local_enterprise {local_enterprise.return_idx()} offline and unable do local updates")
					else:
						print(f"local_enterprise {local_enterprise.return_idx()} in validator {validator.return_idx()}'s black list. This local_enterprise's transactions won't be accpeted.")
			validator.set_unordered_arrival_time_accepted_local_enterprise_transactions(transaction_arrival_queue)
			# in case validator off line for accepting broadcasted transactions but can later back online to validate the transactions itself receives
			validator.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))
			
			# broadcast to other validators
			if transaction_arrival_queue:
				validator.validator_broadcast_local_enterprise_transactions()
			else:
				print("No transactions have been received by this validator, probably due to local_enterprises and/or validators offline or timeout while doing local updates or transmitting updates, or all local_enterprises are in validator's black list.")


		print(''' Step 2.5 - with the broadcasted local_enterprises transactions, validators decide the final transaction arrival order \n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			accepted_broadcasted_validator_transactions = validator.return_accepted_broadcasted_local_enterprise_transactions()
			print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is calculating the final transactions arrival order by combining the direct local_enterprise transactions received and received broadcasted transactions...")
			accepted_broadcasted_transactions_arrival_queue = {}
			if accepted_broadcasted_validator_transactions:
				# calculate broadcasted transactions arrival time
				self_validator_link_speed = validator.return_link_speed()
				for broadcasting_validator_record in accepted_broadcasted_validator_transactions:
					broadcasting_validator_link_speed = broadcasting_validator_record['source_validator_link_speed']
					lower_link_speed = self_validator_link_speed if self_validator_link_speed < broadcasting_validator_link_speed else broadcasting_validator_link_speed
					for arrival_time_at_broadcasting_validator, broadcasted_transaction in broadcasting_validator_record['broadcasted_transactions'].items():
						transmission_delay = getsizeof(str(broadcasted_transaction))/lower_link_speed
						accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_validator] = broadcasted_transaction
			else:
				print(f"validator {validator.return_idx()} {validator_iter+1}/{len(validators_this_round)} did not receive any broadcasted local_enterprise transaction this round.")
			# mix the boardcasted transactions with the direct accepted transactions
			final_transactions_arrival_queue = sorted({**validator.return_unordered_arrival_time_accepted_local_enterprise_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
			validator.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
			print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")


		print(''' Step 3 - validators do self and cross-validation(validate local updates from local_enterprises) by the order of transaction arrival time.\n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			final_transactions_arrival_queue = validator.return_final_transactions_validating_queue()
			if final_transactions_arrival_queue:
				# validator asynchronously does one epoch of update and validate on its own test set
				local_validation_time = validator.validator_update_model_by_one_epoch_and_validate_local_accuracy(args['optimizer'])
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is validating received local_enterprise transactions...")
				for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
					if validator.online_switcher():
						# validation won't begin until validator locally done one epoch of update and validation(local_enterprise transactions will be queued)
						if arrival_time < local_validation_time:
							arrival_time = local_validation_time
						validation_time, post_validation_unconfirmmed_transaction = validator.validate_local_enterprise_transaction(unconfirmmed_transaction, rewards, log_files_folder_path, comm_round, args['malicious_validator_on'])
						if validation_time:
							validator.add_post_validation_transaction_to_queue((arrival_time + validation_time, validator.return_link_speed(), post_validation_unconfirmmed_transaction))
							print(f"A validation process has been done for the transaction from local_enterprise {post_validation_unconfirmmed_transaction['local_enterprise_enterprise_idx']} by validator {validator.return_idx()}")
					else:
						print(f"A validation process is skipped for the transaction from local_enterprise {post_validation_unconfirmmed_transaction['local_enterprise_enterprise_idx']} by validator {validator.return_idx()} due to validator offline.")
			else:
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} did not receive any transaction from local_enterprise or validator in this round.")

		print(''' Step 4 - validators send post validation transactions to associated miner and miner broadcasts these to other miners in their respecitve peer lists\n''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			# resync chain
			if miner.resync_chain(mining_consensus):
				miner.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
			
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting validators' post-validation transactions...")
			associated_validators = list(miner.return_associated_validators())
			if not associated_validators:
				print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
				continue
			self_miner_link_speed = miner.return_link_speed()
			validator_transactions_arrival_queue = {}
			for validator_iter in range(len(associated_validators)):
				validator = associated_validators[validator_iter]
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(associated_validators)} of miner {miner.return_idx()} is sending signature verified transaction...")
				post_validation_transactions_by_validator = validator.return_post_validation_transactions_queue()
				post_validation_unconfirmmed_transaction_iter = 1
				for (validator_sending_time, source_validator_link_spped, post_validation_unconfirmmed_transaction) in post_validation_transactions_by_validator:
					if validator.online_switcher() and miner.online_switcher():
						lower_link_speed = self_miner_link_speed if self_miner_link_speed < source_validator_link_spped else source_validator_link_spped
						transmission_delay = getsizeof(str(post_validation_unconfirmmed_transaction))/lower_link_speed
						validator_transactions_arrival_queue[validator_sending_time + transmission_delay] = post_validation_unconfirmmed_transaction
						print(f"miner {miner.return_idx()} has accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()}")
					else:
						print(f"miner {miner.return_idx()} has not accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()} due to one of enterprises or both offline.")
					post_validation_unconfirmmed_transaction_iter += 1
			miner.set_unordered_arrival_time_accepted_validator_transactions(validator_transactions_arrival_queue)
			miner.miner_broadcast_validator_transactions()

		print(''' Step 4.5 - with the broadcasted validator transactions, miners decide the final transaction arrival order\n ''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			accepted_broadcasted_validator_transactions = miner.return_accepted_broadcasted_validator_transactions()
			self_miner_link_speed = miner.return_link_speed()
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} calculating the final transactions arrival order by combining the direct local_enterprise transactions received and received broadcasted transactions...")
			accepted_broadcasted_transactions_arrival_queue = {}
			if accepted_broadcasted_validator_transactions:
				# calculate broadcasted transactions arrival time
				for broadcasting_miner_record in accepted_broadcasted_validator_transactions:
					broadcasting_miner_link_speed = broadcasting_miner_record['source_enterprise_link_speed']
					lower_link_speed = self_miner_link_speed if self_miner_link_speed < broadcasting_miner_link_speed else broadcasting_miner_link_speed
					for arrival_time_at_broadcasting_miner, broadcasted_transaction in broadcasting_miner_record['broadcasted_transactions'].items():
						transmission_delay = getsizeof(str(broadcasted_transaction))/lower_link_speed
						accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_miner] = broadcasted_transaction
			else:
				print(f"miner {miner.return_idx()} {miner_iter+1}/{len(miners_this_round)} did not receive any broadcasted validator transaction this round.")
			# mix the boardcasted transactions with the direct accepted transactions
			final_transactions_arrival_queue = sorted({**miner.return_unordered_arrival_time_accepted_validator_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
			miner.set_candidate_transactions_for_final_mining_queue(final_transactions_arrival_queue)
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")
		
		print(''' Step 5 - miners do self and cross-verification (verify validators' signature) by the order of transaction arrival time and record the transactions in the candidate block according to the limit size. Also mine and propagate the block.\n''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			final_transactions_arrival_queue = miner.return_final_candidate_transactions_mining_queue()
			valid_validator_sig_candidate_transacitons = []
			invalid_validator_sig_candidate_transacitons = []
			begin_mining_time = 0
			if final_transactions_arrival_queue:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is verifying received validator transactions...")
				time_limit = miner.return_miner_acception_wait_time()
				size_limit = miner.return_miner_accepted_transactions_size_limit()
				for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
					new_begining_mining_time = 0
					if miner.online_switcher():
						if time_limit:
							if arrival_time > time_limit:
								break
						if size_limit:
							if getsizeof(str(valid_validator_sig_candidate_transacitons+invalid_validator_sig_candidate_transacitons)) > size_limit:
								break
						# verify validator signature of this transaction
						verification_time, is_validator_sig_valid = miner.verify_validator_transaction(unconfirmmed_transaction)
						if verification_time:
							if is_validator_sig_valid:
								validator_info_this_tx = {
								'validator': unconfirmmed_transaction['validation_done_by'],
								'validation_rewards': unconfirmmed_transaction['validation_rewards'],
								'validation_time': unconfirmmed_transaction['validation_time'],
								'validator_rsa_pub_key': unconfirmmed_transaction['validator_rsa_pub_key'],
								'validator_signature': unconfirmmed_transaction['validator_signature'],
								'update_direction': unconfirmmed_transaction['update_direction'],
								'miner_enterprise_idx': miner.return_idx(),
								'miner_verification_time': verification_time,
								'miner_rewards_for_this_tx': rewards}
								# validator's transaction signature valid
								found_same_local_enterprise_transaction = False
								for valid_validator_sig_candidate_transaciton in valid_validator_sig_candidate_transacitons:
									if valid_validator_sig_candidate_transaciton['local_enterprise_signature'] == unconfirmmed_transaction['local_enterprise_signature']:
										found_same_local_enterprise_transaction = True
										break
								if not found_same_local_enterprise_transaction:
									valid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
									del valid_validator_sig_candidate_transaciton['validation_done_by']
									del valid_validator_sig_candidate_transaciton['validation_rewards']
									del valid_validator_sig_candidate_transaciton['update_direction']
									del valid_validator_sig_candidate_transaciton['validation_time']
									del valid_validator_sig_candidate_transaciton['validator_rsa_pub_key']
									del valid_validator_sig_candidate_transaciton['validator_signature']
									valid_validator_sig_candidate_transaciton['positive_direction_validators'] = []
									valid_validator_sig_candidate_transaciton['negative_direction_validators'] = []
									valid_validator_sig_candidate_transacitons.append(valid_validator_sig_candidate_transaciton)
								if unconfirmmed_transaction['update_direction']:
									valid_validator_sig_candidate_transaciton['positive_direction_validators'].append(validator_info_this_tx)
								else:
									valid_validator_sig_candidate_transaciton['negative_direction_validators'].append(validator_info_this_tx)
								transaction_to_sign = valid_validator_sig_candidate_transaciton
							else:
								# validator's transaction signature invalid
								invalid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
								invalid_validator_sig_candidate_transaciton['miner_verification_time'] = verification_time
								invalid_validator_sig_candidate_transaciton['miner_rewards_for_this_tx'] = rewards
								invalid_validator_sig_candidate_transacitons.append(invalid_validator_sig_candidate_transaciton)
								transaction_to_sign = invalid_validator_sig_candidate_transaciton
							# (re)sign this candidate transaction
							signing_time = miner.sign_candidate_transaction(transaction_to_sign)
							new_begining_mining_time = arrival_time + verification_time + signing_time
					else:
						print(f"A verification process is skipped for the transaction from validator {unconfirmmed_transaction['validation_done_by']} by miner {miner.return_idx()} due to miner offline.")
						new_begining_mining_time = arrival_time
					begin_mining_time = new_begining_mining_time if new_begining_mining_time > begin_mining_time else begin_mining_time
				# FedAnil: add global params to transaction of block
				transactions_to_record_in_block = {}
				transactions_to_record_in_block['valid_validator_sig_transacitons'] = valid_validator_sig_candidate_transacitons
				transactions_to_record_in_block['invalid_validator_sig_transacitons'] = invalid_validator_sig_candidate_transacitons
				transactions_to_record_in_block['global_update_params'] = miner.return_global_parametesrs()
				# put transactions into candidate block and begin mining
				# block index starts from 1
				start_time_point = time.time()
				candidate_block = Block(idx=miner.return_consortium_blockchain_object().return_chain_length()+1, transactions=transactions_to_record_in_block, miner_rsa_pub_key=miner.return_rsa_pub_key())
				# mine the block
				miner_computation_power = miner.return_computation_power()
				if not miner_computation_power:
					block_generation_time_spent = float('inf')
					miner.set_block_generation_time_point(float('inf'))
					print(f"{miner.return_idx()} - miner mines a block in INFINITE time...")
					continue
				recorded_transactions = candidate_block.return_transactions()
				if recorded_transactions['valid_validator_sig_transacitons'] or recorded_transactions['valid_validator_sig_transacitons']:
					print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} mining the block...")
					# return the last block and add previous hash
					last_block = miner.return_consortium_blockchain_object().return_last_block()
					if last_block is None:
						# will mine the genesis block
						candidate_block.set_previous_block_hash(None)
					else:
						candidate_block.set_previous_block_hash(last_block.compute_hash(hash_entire_block=True))
					# mine the candidate block by PoW, inside which the block_hash is also set
					mined_block = miner.mine_block(candidate_block, rewards)
				else:
					print("No transaction to mine for this block.")
					continue
				# unfortunately may go offline while propagating its block
				if miner.online_switcher():
					# sign the block
					miner.sign_block(mined_block)
					miner.set_mined_block(mined_block)
					# record mining time
					block_generation_time_spent = (time.time() - start_time_point)/miner_computation_power
					miner.set_block_generation_time_point(begin_mining_time + block_generation_time_spent)
					print(f"{miner.return_idx()} - miner mines a block in {block_generation_time_spent} seconds.")
					# immediately propagate the block
					miner.propagated_the_block(miner.return_block_generation_time_point(), mined_block)
				else:
					print(f"Unfortunately, {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} goes offline after, if successful, mining a block. This if-successful-mined block is not propagated.")
			else:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} did not receive any transaction from validator or miner in this round.")

		print(''' Step 6 - miners decide if adding a propagated block or its own mined block as the legitimate block, and request its associated enterprises to download this block''')
		forking_happened = False
		# comm_round_block_gen_time regarded as the time point when the winning miner mines its block, calculated from the beginning of the round. If there is forking in PoW or rewards info out of sync in PoS, this time is the avg time point of all the appended time by any enterprise
		comm_round_block_gen_time = []
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			unordered_propagated_block_processing_queue = miner.return_unordered_propagated_block_processing_queue()
			# add self mined block to the processing queue and sort by time
			this_miner_mined_block = miner.return_mined_block()
			leader_miner = miner
			if this_miner_mined_block:
				unordered_propagated_block_processing_queue[miner.return_block_generation_time_point()] = this_miner_mined_block
			ordered_all_blocks_processing_queue = sorted(unordered_propagated_block_processing_queue.items())
			if ordered_all_blocks_processing_queue:
				if mining_consensus == 'PoW':
					print("\nselect winning block based on PoW")
					
					# abort mining if propagated block is received
					print(f"{leader_miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is deciding if a valid propagated block arrived before it successfully mines its own block...")
					for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
						verified_block, verification_time = leader_miner.verify_block(block_to_verify, block_to_verify.return_mined_by())
						if verified_block:
							block_mined_by = verified_block.return_mined_by()
							if block_mined_by == leader_miner.return_idx():
								print(f"Miner {leader_miner.return_idx()} is adding its own mined block.")
							else:
								print(f"Miner {leader_miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()}.")
							if leader_miner.online_switcher():
								# FedAnil: upload global model to consortium blockchain
								leader_miner.add_block(verified_block)
							else:
								print(f"Unfortunately, miner {leader_miner.return_idx()} goes offline while adding this block to its chain.")
							if leader_miner.return_the_added_block():
								# requesting enterprises in its associations to download this block
								leader_miner.request_to_download(verified_block, block_arrival_time + verification_time)
								break								
				else:
					# PoS
					candidate_PoS_blocks = {}
					print("select winning block based on PoS")
					# filter the ordered_all_blocks_processing_queue to contain only the blocks within time limit
					for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
						if block_arrival_time < args['miner_pos_propagated_block_wait_time']:
							candidate_PoS_blocks[enterprises_in_network.enterprises_set[block_to_verify.return_mined_by()].return_stake()] = block_to_verify
					high_to_low_stake_ordered_blocks = sorted(candidate_PoS_blocks.items(), reverse=True)
					# for PoS, requests every enterprise in the network to add a valid block that has the most miner stake in the PoS candidate blocks list, which can be verified through chain
					for (stake, PoS_candidate_block) in high_to_low_stake_ordered_blocks:
						verified_block, verification_time = miner.verify_block(PoS_candidate_block, PoS_candidate_block.return_mined_by())
						if verified_block:
							block_mined_by = verified_block.return_mined_by()
							if block_mined_by == leader_miner.return_idx():
								print(f"Miner {leader_miner.return_idx()} with stake {stake} is adding its own mined block.")
							else:
								print(f"Miner {leader_miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()} with stake {stake}.")
							if leader_miner.online_switcher():
								# FedAnil: upload global model to consortium blockchain
								leader_miner.add_block(verified_block)
							else:
								print(f"Unfortunately, miner {leader_miner.return_idx()} goes offline while adding this block to its chain.")
							if leader_miner.return_the_added_block():
								# requesting enterprises in its associations to download this block
								leader_miner.request_to_download(verified_block, block_arrival_time + verification_time)
								break
				leader_miner.add_to_round_end_time(block_arrival_time + verification_time)
			else:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} does not receive a propagated block and has not mined its own block yet.")
		# CHECK FOR FORKING
		added_blocks_miner_set = set()
		for enterprise in enterprises_list:
			the_added_block = enterprise.return_the_added_block()
			if the_added_block:
				print(f"{enterprise.return_role()} {enterprise.return_idx()} has added a block mined by {the_added_block.return_mined_by()}")
				added_blocks_miner_set.add(the_added_block.return_mined_by())
				block_generation_time_point = enterprises_in_network.enterprises_set[the_added_block.return_mined_by()].return_block_generation_time_point()
				# commented, as we just want to plot the legitimate block gen time, and the wait time is to avoid forking. Also the logic is wrong. Should track the time to the slowest local_enterprise after its global model update
				# if mining_consensus == 'PoS':
				# 	if args['miner_pos_propagated_block_wait_time'] != float("inf"):
				# 		block_generation_time_point += args['miner_pos_propagated_block_wait_time']
				comm_round_block_gen_time.append(block_generation_time_point)
		if len(added_blocks_miner_set) > 1:
			print("WARNING: a forking event just happened!")
			forking_happened = True
			with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file:
				file.write(f"Forking in round {comm_round}\n")
		else:
			print("No forking event happened.")
			

		print(''' Step 6 last step - process the added block - 1.collect usable updated params\n 2.malicious enterprises identification\n 3.get rewards\n 4.do local udpates\n This code block is skipped if no valid block was generated in this round''')
		all_enterprises_round_ends_time = []
		for enterprise in enterprises_list:
			if enterprise.return_the_added_block() and enterprise.online_switcher():
				# collect usable updated params, malicious enterprises identification, get rewards and do local udpates
				processing_time = enterprise.process_block(enterprise.return_the_added_block(), log_files_folder_path, conn, conn_cursor)
				enterprise.other_tasks_at_the_end_of_comm_round(comm_round, log_files_folder_path)
				enterprise.add_to_round_end_time(processing_time)
				all_enterprises_round_ends_time.append(enterprise.return_round_end_time())
		# FedAnil: Total Accuracy (%)
		print(''' Logging Accuracies by Enterprises ''')
		for enterprise in enterprises_list:
			accuracy_this_round = enterprise.validate_model_weights()
			if total_accuracy < accuracy_this_round:
				total_accuracy = accuracy_this_round
			with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
				is_malicious_node = "M" if enterprise.return_is_malicious() else "B"
				file.write(f"{enterprise.return_idx()} {enterprise.return_role()} {is_malicious_node}: {accuracy_this_round}\n")
         
		# FedAnil: Total Computation Cost (Seconds)
		communication_bytes_sum += communication_bytes_per_round
		# logging time, mining_consensus and forking
		# get the slowest enterprise end time
		# # FedAnil: Total Computation Cost (Seconds)
		comm_round_spent_time = time.time() - comm_round_start_time
		with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
			# corner case when all miners in this round are malicious enterprises so their blocks are rejected
			try:
				comm_round_block_gen_time = max(comm_round_block_gen_time)
				file.write(f"comm_round_block_gen_time: {comm_round_block_gen_time}\n")
				file.write(f"communication overhead: {communication_bytes_per_round} bytes\n")
			except:
				no_block_msg = "No valid block has been generated this round."
				print(no_block_msg)
				file.write(f"comm_round_block_gen_time: {no_block_msg}\n")
				with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a+') as file2:
					# TODO this may be caused by "no transaction to mine" for the miner. Forgot to check for block miner's maliciousness in request_to_downlaod()
					file2.write(f"No valid block in round {comm_round}\n")
			try:
				slowest_round_ends_time = max(all_enterprises_round_ends_time)
				file.write(f"slowest_enterprise_round_ends_time: {slowest_round_ends_time}\n")
			except:
				# corner case when all transactions are rejected by miners
				file.write("slowest_enterprise_round_ends_time: No valid block has been generated this round.\n")
				with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a+') as file2:
					no_valid_block_msg = f"No valid block in round {comm_round}\n"
					if len(file2.readlines()) > 0:
						if file2.readlines()[-1] != no_valid_block_msg:
							file2.write(no_valid_block_msg)
			file.write(f"mining_consensus: {mining_consensus} {args['pow_difficulty']}\n")
			file.write(f"forking_happened: {forking_happened}\n")
			file.write(f"comm_round_spent_time_on_this_machine: {comm_round_spent_time}\n")
	        # FedAnil: Total Computation Cost (Second)
			computation_sum += comm_round_spent_time
		conn.commit()

		# if no forking, log the block miner
		if not forking_happened:
			legitimate_block = None
			for enterprise in enterprises_list:
				legitimate_block = enterprise.return_the_added_block()
				if legitimate_block is not None:
					# skip the enterprise who's been identified malicious and cannot get a block from miners
					break
			with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
				if legitimate_block is None:
					file.write("block_mined_by: no valid block generated this round\n")
				else:
					block_mined_by = legitimate_block.return_mined_by()
					is_malicious_node = "M" if enterprises_in_network.enterprises_set[block_mined_by].return_is_malicious() else "B"
					file.write(f"block_mined_by: {block_mined_by} {is_malicious_node}\n")
		else:
			with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
				file.write(f"block_mined_by: Forking happened\n")

		with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
			file.write(f"Malicious Enterprises:\n")
			for enterprise in enterprises_list:
				if enterprise.return_is_malicious():
					file.write(f"{enterprise.return_idx()}, ")
		print(''' Logging Stake by Enterprises ''')
		for enterprise in enterprises_list:
			accuracy_this_round = enterprise.validate_model_weights()
			with open(f"{log_files_folder_path_comm_round}/stake_comm_{comm_round}.txt", "a") as file:
				is_malicious_node = "M" if enterprise.return_is_malicious() else "B"
				file.write(f"{enterprise.return_idx()} {enterprise.return_role()} {is_malicious_node}: {enterprise.return_stake()}\n")

		# a temporary workaround to free GPU mem by delete txs stored in the blocks. Not good when need to resync chain
		if args['destroy_tx_in_block']:
			for enterprise in enterprises_list:
				last_block = enterprise.return_consortium_blockchain_object().return_last_block()
				if last_block:
					last_block.free_tx()

		# save network_snapshot if reaches save frequency
		if args['save_network_snapshots'] and (comm_round == 1 or comm_round % args['save_freq'] == 0):
			if args['save_most_recent']:
				paths = sorted(Path(network_snapshot_save_path).iterdir(), key=os.path.getmtime)
				if len(paths) > args['save_most_recent']:
					for _ in range(len(paths) - args['save_most_recent']):
						open(paths[_], 'w').close() 
						os.remove(paths[_])
			snapshot_file_path = f"{network_snapshot_save_path}/snapshot_r_{comm_round}"
			print(f"Saving network snapshot to {snapshot_file_path}")
			pickle.dump(enterprises_in_network, open(snapshot_file_path, "wb"))
		# FedAnil: if accuracy reach more than target accuracy the iteration finished
		if total_accuracy >= target_accuracy:
			break
	
	with open(f'{log_files_folder_path}/Output.txt', 'w') as f:
		# FedAnil: Total Computation Cost (Seconds)
		f.write(f"Total Computation Cost (Seconds): {computation_sum} \n")
		# FedAnil: Total Communication Cost (Bytes)
		f.write(f"Total Communication Cost (Bytes): {communication_bytes_sum} \n")
		# FedAnil: Total Accuracy (%)
		f.write(f"Total Accuracy (%): {total_accuracy * 100} \n")