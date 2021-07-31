import time
import os
import parameters as pm
import multiprocessing
import agent 
from torch.utils.tensorboard import SummaryWriter

def log_config():
	# log all configurations in parameters and backup py
	global LOG_DIR
	if pm.EXPERIMENT_NAME is None:
		LOG_DIR = "./backup/"
	else:
		LOG_DIR = "./" + pm.EXPERIMENT_NAME + "/"

	os.system("mkdir -p " + LOG_DIR + "; cp *.py *.txt " + LOG_DIR)

	pm_md = globals().get('pm', None) # Return the dictionary containing the current scope's global variables.
	train_config = dict()
	if pm_md:
		train_config = {key: value for key, value in pm_md.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
	train_config_str = ""
	for key, value in train_config.items():
		if key != None and value != None:
			train_config_str += "{:<30}{:<100}".format(key, value) + "\n\n" # It ususally don't have any 

	# tb_logger.add_text(tag="Config", value=train_config_str, step=0)
	# tb_logger.flush()
    # the codes above are original codes, we do not write metadata and summary in trainconfig. there is a problem

	if pm.TRAINING_MODE == "SL":
		f = open(pm.MODEL_DIR + "sl_model.config", "w")
	else:
		f = open(pm.MODEL_DIR + "rl_model.config", "w")
	f.write(train_config_str)
	f.close()

	f = open(LOG_DIR + "config.md", 'w')
	f.write(train_config_str)
	f.close()





def main():
	os.system("rm -f *.log")
	os.system("sudo pkill -9 tensorboard; sleep 3")
    
	net_weights_qs = [multiprocessing.Queue(1) for i in range(pm.NUM_AGENTS)]
	net_gradients_qs = [multiprocessing.Queue(1) for i in range(pm.NUM_AGENTS)]
	stats_qs = [multiprocessing.Queue() for i in range(pm.NUM_AGENTS)]

	os.system("mkdir -p " + pm.MODEL_DIR + "; mkdir -p " + pm.SUMMARY_DIR)
	if pm.EXPERIMENT_NAME is None:
		cmd = "cd " + pm.SUMMARY_DIR + " && rm -rf *; tensorboard --logdir=./"
		board = multiprocessing.Process(target=lambda: os.system(cmd), args=())
		board.start()
		time.sleep(3) # let tensorboard start first since it will clear the dir


	agent.central_agent(net_weights_qs, net_gradients_qs, stats_qs)
	log_config() # It used to place in central agent
	# master = multiprocessing.Process(target=central_agent, args=(net_weights_qs, net_gradients_qs, stats_qs,))
	# master.start()
	#agent(net_weights_qs[0], net_gradients_qs[0], stats_qs[0], 0)
	#exit()



if __name__ == "__main__":
	main()