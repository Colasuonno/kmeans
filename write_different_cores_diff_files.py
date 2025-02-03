import csv
import datetime
import multiprocessing
import subprocess
import os
import difflib

TIMEOUT = 60 * 5 # 5 minutes

INPUT_AMOUNTS = {
	"2D": 5000,
	"20D": 10000,
	"100D2": 100000,
}

INPUT_FILE = "test_files/input"
ITERATIONS = 1000
THESHOLD = 1 # 2 max dist
CHANGES = 0.1

CSV_HEADER = ["# Cores", "# Cluster", "# Iterations", "% Changes", "Threshold", "END Before " + str(TIMEOUT), "Time"]

TESTING_DATA = {
	"clusters": [
		10,
		100,
		1000,
		10000,
		100000
	]
}

def _run_test(cores, cluster, input_file, iterations, changes, threshold, output_file):
	start = datetime.datetime.now()
	print("Running on " + str(cores) + " cores")
	command = ["mpirun", "-np", cores, "./KMEANS_omp", str(input_file), str(cluster), str(iterations), str(changes), str(threshold), output_file]
	prc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	try:
		prc.wait(timeout=TIMEOUT)
	except subprocess.TimeoutExpired:
		prc.kill()  # Kill the process if it exceeds the timeout
		return False, -1  # Return failure status and time as -1
    
	if prc.returncode != 0:
		return False, -1

	tot_time = datetime.datetime.now() - start

	return True, tot_time

def diff_between_files(filename1, filename2):
	with open(filename1, 'r', encoding='utf-8') as f1, open(filename2, 'r', encoding='utf-8') as f2:
		f1_lines = f1.readlines()
		f2_lines = f2.readlines()
		return len(list(difflib.unified_diff(f1_lines, f2_lines, fromfile=filename1, tofile=filename2, lineterm=""))) > 0

def run_test(output_writer):
	
	for cluster_amnt in TESTING_DATA["clusters"]:
		
		reports = []


		for inpt_amnt in INPUT_AMOUNTS:

			inpt_file = INPUT_FILE + inpt_amnt + ".inp"

			for i in range(1,5):
				# I have 4 core :()
				omp_i_end, omp_time = _run_test(str(i), cluster_amnt, inpt_file, ITERATIONS, THESHOLD, CHANGES, "useless.txt")
				reports.append([
				"OMP/MPI (" + str(i) + " Cores) (File: " + str(inpt_file) + ")",
				cluster_amnt,
				ITERATIONS,
				CHANGES,
				THESHOLD,
				omp_i_end,
				omp_time
				])

		output_writer.writerows(reports)



		

if __name__ == "__main__":
	# Run test and compare
	
	# Recompile everythig
	os.system("make all")
	with open("omp_cores_diff_files_output.csv", mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(CSV_HEADER)
		run_test(writer)