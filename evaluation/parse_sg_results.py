import os

results_dir = "sg_results"

def main():
	results = {}
	def foo(task):
		outfile = os.path.join(results_dir, f"{task}.txt")
		if not os.path.exists(outfile):
			print(f"File {outfile} does not exist, skipping")
			return
		with open(outfile, "r") as f:
			lines = f.readlines()
		prefix = "Stratum set: 1..2\t Time in run(): "

		lines = [line[len(prefix):].strip().replace("us","") for line in lines if line.startswith(prefix)]
		lines = [int(line) for line in lines]
		if len(lines) == 0:
			print(f"No valid lines in {outfile}, skipping")
			return
		average = sum(lines) / len(lines)
		results[task] = average


	tasks = ["fe-sphere",
	"CA-HepTH",
	"ego-Facebook",
	"Gnutella31",
	"fe_body",
	"loc-Brightkite",
	"SF.cedge",
	"com-dblp",
	"usroad",
	"fc_ocean",
	"vsp_finan"]
	for task in tasks:
		foo(task)

	for task in tasks:
		print(task, end=',')
		if task in results:
			print(results[task])
		else:
			print()

main()
