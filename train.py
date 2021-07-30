import time
import os
import parameters as pm
import multiprocessing
import agent 
from torch.utils.tensorboard import SummaryWriter

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
	# master = multiprocessing.Process(target=central_agent, args=(net_weights_qs, net_gradients_qs, stats_qs,))
	# master.start()
	#agent(net_weights_qs[0], net_gradients_qs[0], stats_qs[0], 0)
	#exit()



if __name__ == "__main__":
	main()