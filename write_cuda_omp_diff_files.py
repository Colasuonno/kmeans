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
THESHOLD = 2 # 2 max dist
CHANGES = 0.1

CSV_HEADER = ["Type", "# Cluster", "# Iterations", "% Changes", "Threshold", "END Before " + str(TIMEOUT), "Time"]

TESTING_DATA = {
	"clusters": [
		10,
		100,
		1000,
		10000,
		100000,
		1000000
	]
}

def _run_test(ex_type, cluster, input_file, iterations, changes, threshold, output_file):
	start = datetime.datetime.now()
	print("Running on " + str(ex_type) +  " " + str(input_file))
	if ex_type == "omp":
		command = ["mpirun", "-np", "4", "./KMEANS_omp", str(input_file), str(cluster), str(iterations), str(changes), str(threshold), output_file]
	else:
		command = ["./KMEANS_cuda", str(input_file), str(cluster), str(iterations), str(changes), str(threshold), output_file]
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
			# I have 4 core :()
			omp_end, omp_time = _run_test("omp", cluster_amnt, inpt_file, ITERATIONS, THESHOLD, CHANGES, "useless.txt")
			cuda_end, cuda_time = _run_test("cuda", cluster_amnt, inpt_file, ITERATIONS, THESHOLD, CHANGES, "useless.txt")

			reports.append([
			"OMP/MPI (File: " + str(inpt_file) + ")",
			cluster_amnt,
			ITERATIONS,
			CHANGES,
			THESHOLD,
			omp_end,
			omp_time
			]
			)
			reports.append([
			"CUDA (File: " + str(inpt_file) + ")",
			cluster_amnt,
			ITERATIONS,
			CHANGES,
			THESHOLD,
			cuda_end,
			cuda_time
			])

		output_writer.writerows(reports)



		

if __name__ == "__main__":
	# Run test and compare
	
	# Recompile everythig
	os.system("make all")
	with open("omp_cuda_omp_diff_files_output.csv", mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(CSV_HEADER)
		run_test(writer)