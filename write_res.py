import csv
import datetime
import multiprocessing
import subprocess
import os
import difflib

TIMEOUT = 60 * 5 # 5 minutes

INPUT_NAME = "input100D2"
INPUT_FILE = "test_files/" + str(INPUT_NAME) + ".inp"
ITERATIONS = 1000
THESHOLD = 1 # 2 max dist
CHANGES = 0.1

CSV_HEADER = ["Type", "# Cluster", "# Iterations", "% Changes", "Threshold", "END Before " + str(TIMEOUT), "Time"]

TESTING_DATA = {
	"clusters": [
		10,
		100,
		1000,
		10000,
		100000
	]
}

def _run_test(initial_command, cluster, input_file, iterations, changes, threshold, output_file):
	start = datetime.datetime.now()
	print("Running " + str(initial_command))
	if "omp" in initial_command:
		command = ["mpirun", "-np", "4", "./KMEANS_omp", str(input_file), str(cluster), str(iterations), str(changes), str(threshold), output_file]
	else:
		command = [initial_command, str(input_file), str(cluster), str(iterations), str(changes), str(threshold), output_file]
	prc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	try:
		prc.wait(timeout=TIMEOUT)
	except subprocess.TimeoutExpired:
		print(f"{initial_command} for {cluster} clusters REACHED TIMEOUT... ending")
		prc.kill()  # Kill the process if it exceeds the timeout
		return False, -1  # Return failure status and time as -1
    
	if prc.returncode != 0:
		print(f"Error occurred while running {initial_command} for {cluster} clusters. Return code: {prc.returncode}")
		return False, -1

	tot_time = datetime.datetime.now() - start

	print(initial_command +" with # " + str(cluster) + " elapsed " + str(tot_time))
	return True, tot_time

def diff_between_files(filename1, filename2):
	with open(filename1, 'r', encoding='utf-8') as f1, open(filename2, 'r', encoding='utf-8') as f2:
		f1_lines = f1.readlines()
		f2_lines = f2.readlines()
		return len(list(difflib.unified_diff(f1_lines, f2_lines, fromfile=filename1, tofile=filename2, lineterm=""))) > 0

def run_test(output_writer):
	
	for cluster_amnt in TESTING_DATA["clusters"]:
		seq_end, seq_time = _run_test("./KMEANS_seq", cluster_amnt, INPUT_FILE, ITERATIONS,THESHOLD, CHANGES, "outputs_" + str(INPUT_NAME) +  "/" + str(cluster_amnt) + "_out_seq.txt")
		omp_end, omp_time = _run_test("omp", cluster_amnt, INPUT_FILE, ITERATIONS, THESHOLD, CHANGES, "outputs_" + str(INPUT_NAME) +  "/" + str(cluster_amnt) + "_out_omp.txt")
		cuda_end, cuda_time = _run_test("./KMEANS_cuda", cluster_amnt, INPUT_FILE, ITERATIONS, THESHOLD, CHANGES, "outputs_" + str(INPUT_NAME) +  "/" + str(cluster_amnt) + "_out_cuda.txt")

		output_writer.writerows([[
				"SEQ",
				cluster_amnt,
				ITERATIONS,
				CHANGES,
				THESHOLD,
				seq_end,
				seq_time
			],
			[
				"OMP/MPI",
				cluster_amnt,
				ITERATIONS,
				CHANGES,
				THESHOLD,
				omp_end,
				omp_time
			],
			[
				"CUDA",
				cluster_amnt,
				ITERATIONS,
				CHANGES,
				THESHOLD,
				cuda_end,
				cuda_time
			]
			])


if __name__ == "__main__":
	# Run test and compare
	
	# Recompile everythig
	os.system("make all")
	os.makedirs("outputs_" + INPUT_NAME, exist_ok=True)
	with open("output_" + str(INPUT_NAME) + ".csv", mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(CSV_HEADER)
		run_test(writer)