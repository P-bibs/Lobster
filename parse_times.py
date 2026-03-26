import sys

def chunks(xs, n):
    n = max(1, n)
    return list(xs[i:i+n] for i in range(0, len(xs), n))

def main():
	# read from stdin until EOF
	data = sys.stdin.read()
	lines = data.split('\n')

	lines = [line for line in lines if "CPU" in line or "GPU" in line]
	
	table = [line.split() for line in lines]

	times = [row[1] for row in table]

	times = chunks(times, 2)

	for time in times:
		print(",".join(time))

main()
