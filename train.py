import os
import parameters as pm
import multiprocessing


def main():
	os.system("rm -f *.log")
	os.system("sudo pkill -9 tensorboard; sleep 3")
    
	net_weights_qs = [multiprocessing.Queue(1) for i in range(pm.NUM_AGENTS)]
	net_gradients_qs = [multiprocessing.Queue(1) for i in range(pm.NUM_AGENTS)]
	stats_qs = [multiprocessing.Queue() for i in range(pm.NUM_AGENTS)]

    




if __name__ == "__main__":
	main()