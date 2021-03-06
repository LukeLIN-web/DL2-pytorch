import numpy as np
import parameters as pm
import math
class Trace:
	def __init__(self, logger=None):
		self.logger = logger
		# job arrival patterns
		self.arrv_pattern_1 = [1, 22, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
								 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
								 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1]
		self.arrv_pattern_2 = [2, 40, 2, 2, 2, 3, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1,
								 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
								 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1]
		self.arrv_pattern_3 = [2, 57, 3, 3, 3, 4, 3, 3, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 4, 3, 2, 2, 2,
								 1, 2, 2, 2, 2, 2, 2, 3, 3, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3,
								 2, 2, 2, 2, 3, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 3, 2]
		self.arrv_pattern_4 = [3, 74, 4, 4, 4, 6, 5, 4, 3, 3, 3, 3, 4, 4, 5, 3, 4, 4, 4, 4, 4, 2, 2, 4, 5, 4, 2, 3, 2,
								 2, 2, 3, 2, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 2, 2, 2, 3, 4, 3, 4, 4, 4,
								 3, 3, 3, 3, 4, 3, 2, 3, 2, 2, 2, 3, 3, 3, 2, 4, 3]
		self.arrv_pattern_5 = [4, 10, 5, 4, 5, 6, 5, 4, 3, 3, 4, 3, 4, 5, 5, 4, 5, 5, 5, 5, 4, 3, 3, 4, 6, 4, 3, 3, 3,
								 2, 3, 3, 3, 4, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 4, 4, 4, 4, 4, 4,
								 4, 4, 3, 4, 4, 3, 2, 4, 2, 3, 3, 4, 3, 3, 3, 4, 3]
		self.arrv_pattern_10 = [6, 134, 8, 7, 8, 10, 9, 7, 6, 5, 6, 5, 7, 8, 9, 6, 8, 8, 8, 8, 7, 5, 5, 7, 9, 7, 5, 5, 5,
							   3, 5, 6, 4, 6, 6, 6, 7, 7, 4, 5, 5, 3, 5, 5, 4, 6, 5, 6, 5, 3, 4, 4, 6, 7, 7, 7, 7, 7, 6,
							   6, 6, 6, 7, 6, 4, 6, 4, 5, 4, 6, 5, 6, 5, 7, 6]
		self.arrv_pattern_15 = [11, 224, 13, 12, 14, 18, 15, 12, 10, 9, 11, 9, 12, 14, 15, 10, 14, 14, 13, 14, 12, 8,
								  8, 12, 16, 12, 8, 9, 8, 6, 8, 10, 8, 11, 10, 10, 12, 12, 7, 9, 9, 6, 9, 9, 8, 10, 9,
								  10, 9, 6, 7, 7, 11, 12, 11, 13, 12, 12, 11, 11, 10, 11, 12, 10, 7, 11, 7, 8, 8, 11, 9,
								  10, 8, 12, 10]
		self.arrv_pattern_20 = [14, 288, 17, 16, 18, 23, 19, 16, 13, 11, 14, 11, 15, 18, 19, 13, 18, 18, 17, 18, 16, 11,
							   10, 15, 21, 15, 10, 11, 10, 8, 11, 12, 10, 14, 12, 13, 16, 16, 9, 12, 12, 7, 11, 12, 10,
							   13, 12, 13, 12, 8, 10, 9, 14, 15, 15, 16, 15, 16, 14, 14, 13, 14, 15, 13, 9, 14, 9, 11,
							   10, 14, 12, 13, 11, 15, 13]
		self.arrv_pattern_30 = [20, 403, 24, 23, 26, 32, 27, 22, 18, 16, 19, 16, 22, 26, 27, 19, 25, 26, 24, 25, 22, 15,
							   15, 21, 29, 21, 15, 16, 15, 11, 15, 18, 14, 20, 18, 19, 22, 23, 13, 17, 17, 11, 16, 16,
							   14, 18, 17, 18, 17, 11, 14, 13, 20, 21, 21, 23, 22, 23, 20, 20, 18, 20, 21, 19, 12, 20,
							   12, 16, 14, 20, 17, 18, 15, 21, 19]
		self.arrv_pattern_40 = [25, 504, 30, 29, 32, 40, 34, 28, 23, 20, 24, 20, 27, 32, 34, 24, 32, 33, 31, 32, 28, 19,
							   18, 27, 37, 27, 18, 20, 19, 14, 19, 22, 18, 25, 22, 23, 28, 28, 17, 22, 21, 13, 20, 21,
							   18, 23, 22, 23, 21, 14, 17, 17, 25, 27, 26, 29, 27, 28, 25, 25, 22, 26, 27, 23, 16, 26,
							   15, 20, 18, 25, 22, 23, 19, 27, 23]
		self.arrv_pattern_50 = [33, 672, 40, 38, 43, 54, 45, 38, 31, 27, 33, 27, 36, 43, 46, 32, 43, 44, 41, 42, 37, 26,
							   25, 36, 49, 36, 25, 27, 25, 19, 25, 30, 24, 34, 30, 31, 37, 38, 23, 29, 28, 18, 27, 28,
							   24, 30, 29, 30, 28, 18, 23, 22, 34, 36, 35, 39, 37, 38, 34, 34, 30, 34, 36, 31, 21, 34,
							   21, 26, 24, 34, 29, 30, 26, 36, 31]
		# ali trace, JCT 147 minutes on average
		self.ali_trace_arrv_pattern = []

	def _get_pattern(self, max_arrvs_per_ts):
		if pm.JOB_ARRIVAL_PATTERN == "Uniform":
			return [max_arrvs_per_ts for _ in range(100)]
		elif pm.JOB_ARRIVAL_PATTERN == "Poisson":
			return np.random.poisson(max_arrvs_per_ts, 100)
		elif pm.JOB_ARRIVAL_PATTERN == "Google_Trace":
			if max_arrvs_per_ts == 1:
				return self.arrv_pattern_1
			elif max_arrvs_per_ts == 2:
				return self.arrv_pattern_2
			elif max_arrvs_per_ts == 3:
				return self.arrv_pattern_3
			elif max_arrvs_per_ts == 4:
				return self.arrv_pattern_4
			elif max_arrvs_per_ts == 5:
				return self.arrv_pattern_5
			elif max_arrvs_per_ts == 10:
				return self.arrv_pattern_10
			elif max_arrvs_per_ts == 15:
				return self.arrv_pattern_15
			elif max_arrvs_per_ts == 20:
				return self.arrv_pattern_20
			elif max_arrvs_per_ts == 30:
				return self.arrv_pattern_30
			elif max_arrvs_per_ts == 40:
				return self.arrv_pattern_40
			elif max_arrvs_per_ts == 50:
				return self.arrv_pattern_50
			else:
				self.logger.error("unrecognizable arrival pattern!")
				exit(-1)
		elif pm.JOB_ARRIVAL_PATTERN == "Ali_Trace":
			ratio = max(self.ali_trace_arrv_pattern)/float(max_arrvs_per_ts) # the codes have a problem
			trace = []
			for arrv in self.ali_trace_arrv_pattern:
				trace.append(int(math.ceil(arrv/ratio)))
			return trace

	def get_trace(self, num_type=8):
		# google trace
		trace = dict()
		id = 1
		count_num_jobs = 0
		done = False
		arrv_pattern = self._get_pattern(pm.MAX_ARRVS_PER_TS)

		offset = np.random.randint(5)

