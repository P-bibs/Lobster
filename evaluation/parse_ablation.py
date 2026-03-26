import os
import sys
from collections import defaultdict

results_dir = "ablation_results"

def main(task):
	table = defaultdict(dict)
	count = defaultdict(dict)
	for i in range(5,36):
		suffixes = ["no_frog_no_absorption", "yes_frog_no_absorption", "no_frog_yes_absorption", "yes_frog_yes_absorption", "scallop"]
		for suffix in suffixes:
			outfile = os.path.join(results_dir, f"{task}{i}_{suffix}.txt")
			if not os.path.exists(outfile):
				print(f"File {outfile} does not exist, skipping")
				continue
			with open(outfile, "r") as f:
				lines = f.readlines()

			#if False:
			#	prefix = "Timer [main]: "
			#else:
			#	if "yes_absorption" in suffix:
			#		prefix = "Stratum set: 16..18\t Time in run(): "
			#	else:
			#		prefix = "Stratum set: 16..17\t Time in run(): "
			prefix = "Total sample time: "

			lines = [line[len(prefix):].strip().replace("us","") for line in lines if line.startswith(prefix)]
			lines = [int(line) for line in lines]
			lines = lines[2:]
			if len(lines) == 0:
				print(f"No valid lines in {outfile}, skipping")
				continue
			average = sum(lines) / len(lines)
			table[i][suffix] = average
			count[i][suffix] = len(lines)

	#try:
	#	padding = 7
	#	# pad each to 7 chars
	#	print("nono".ljust(padding), "yesno".ljust(padding), "noyes".ljust(padding), "yesyes".ljust(padding))
	#	for key in sorted(table.keys()):
	#		print(f"{key:<{2}}", end=': ')
	#		def do_print(s):
	#			print(f"{s:<{padding}}", end=' ')
	#		do_print(round(table[key]["no_frog_no_absorption"]))
	#		do_print(round(table[key]["yes_frog_no_absorption"]))
	#		do_print(round(table[key]["no_frog_yes_absorption"]))
	#		do_print(round(table[key]["yes_frog_yes_absorption"]))
	#		print()
	#except Exception as e:
	#	print(f"Error while printing table: {e}")
	#	print("Falling back to comma-separated values")

	def foo(t):
		print("size,nono,yesno,noyes,yesyes,scallop,")
		for key in sorted(t.keys()):
			print(key, end=',')
			def do_print(d, k):
				if k in d:
					print(d[k], end=',')
				else:
					print("", end=',')
			do_print(t[key],("no_frog_no_absorption")  )
			do_print(t[key],("yes_frog_no_absorption") )
			do_print(t[key],("no_frog_yes_absorption") )
			do_print(t[key],("yes_frog_yes_absorption"))
			do_print(t[key],("scallop"))
			print()
	foo(table)
	foo(count)


main("pacman")
main("pathfinder")
