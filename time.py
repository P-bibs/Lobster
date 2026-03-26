import re
import sys
import statistics
import signal
import time
import os

pattern1 = re.compile(r"Stratum set: (\d+)\.\.(\d+)\s*Time in run\(\): (\d+)us")
pattern2 = re.compile(r"Stratum: (\d+)\s+Time in run\(\): (\d+)us")

def average(l):
    return (sum(l) - l[0]) / (len(l) - 1)


gpu_times = []
cpu_times = []

def signal_handler(sig, frame):
    # calculate mean and stddev
    cpu_mean = average(cpu_times)
    gpu_mean = average(gpu_times)

    cpu_stddev = statistics.stdev(cpu_times)
    gpu_stddev = statistics.stdev(gpu_times)

    print("Num samples: ", len(cpu_times)-1)
    print("CPU: ", cpu_mean, " +/- ", cpu_stddev, "\nGPU: ", gpu_mean, " +/- ", gpu_stddev)
    sys.exit()

def main():

    if len(sys.argv) > 1:
        timeout = float(sys.argv[1])
    else:
        timeout = float('inf')

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGSEGV, signal_handler)

    stratum_set = None
    current_gpu_time = None
    current_cpu_time = 0

    found_first_line = False

    start = time.time()
    for line in sys.stdin:
        if time.time() - start > timeout:
            os.kill(os.getpid(), signal.SIGINT)
        if line.strip() == "=" * 40:
            if not found_first_line:
                found_first_line = True
                continue
            
            if current_cpu_time == 0 and current_gpu_time is None:
                print("No CPU or GPU time, skipping")
                continue

            #if current_cpu_time == 0:
            #    raise Exception("No CPU time")
            #if current_gpu_time is None:
            #    raise Exception("No GPU time")
            cpu_times.append(current_cpu_time)
            gpu_times.append(current_gpu_time)
            current_cpu_time = 0
            current_gpu_time = None
            if len(gpu_times) > 3:
                # print last sample
                print("CPU: ", cpu_times[-1], "\nGPU: ", gpu_times[-1])
                # print average
                print("CPU: ", average(cpu_times[2:]), "\nGPU: ", average(gpu_times[2:]), end='\n\n')
        if "Stratum set" in line:
            line = line.strip()
            if stratum_set is None:
                result = re.search(pattern1, line)
                stratum_set_start = result.group(1)
                stratum_set_end = result.group(2)
                stratum_set = []
                for i in range(int(stratum_set_start), int(stratum_set_end)):
                    stratum_set.append(str(i))
            duration = int(line.split(": ")[-1].replace("us", ""))
            if current_gpu_time is not None:
                raise Exception("GPU time already set")
            current_gpu_time = duration
        elif "run()" in line:
            result = re.search(pattern2, line)
            stratum = result.group(1)
            duration = int(result.group(2))
            if stratum_set is not None and stratum in stratum_set:
                current_cpu_time += duration

main()
