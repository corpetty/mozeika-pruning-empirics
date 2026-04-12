#!/usr/bin/env python3
import argparse
import subprocess
import time
import statistics
import re
from pathlib import Path

def run_once(cmd):
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                t1 = time.perf_counter()
                return t1 - t0, proc.returncode, proc.stdout

def extract_round_times(output: str):
    # Optional future extension; currently just counts rounds seen
                rounds = len(re.findall(r"Round\s+\d+/\d+", output))
                return rounds

def main():
                parser = argparse.ArgumentParser(description="Benchmark pruning scripts by wall-clock time.")
                parser.add_argument("--python", default="python", help="Python executable to use")
                parser.add_argument("--repeats", type=int, default=3, help="How many repeats per command")
                parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per command")
                parser.add_argument("--outfile", default="benchmark_results.txt", help="Where to save results")
                parser.add_argument("commands", nargs="+", help="Commands to benchmark, each wrapped in quotes and excluding leading python")
                args = parser.parse_args()

                lines = []
                for cmd_str in args.commands:
                                cmd = [args.python] + cmd_str.split()
                                lines.append("=" * 80)
                                lines.append("COMMAND: " + " ".join(cmd))

                                for i in range(args.warmup):
                                                dt, code, out = run_once(cmd)
                                                lines.append(f"warmup {i+1}: {dt:.3f}s returncode={code}")

                                times = []
                                rounds_seen = []
                                for i in range(args.repeats):
                                                dt, code, out = run_once(cmd)
                                                times.append(dt)
                                                rounds_seen.append(extract_round_times(out))
                                                lines.append(f"run {i+1}: {dt:.3f}s returncode={code} rounds_seen={rounds_seen[-1]}")
                                                if code != 0:
                                                                lines.append("---- stdout/stderr ----")
                                                                lines.append(out)
                                                                lines.append("-----------------------")

                                mean_t = statistics.mean(times)
                                med_t = statistics.median(times)
                                std_t = statistics.pstdev(times) if len(times) > 1 else 0.0
                                lines.append(f"mean={mean_t:.3f}s median={med_t:.3f}s std={std_t:.3f}s")
                                if rounds_seen and all(r > 0 for r in rounds_seen):
                                                mean_per_round = mean_t / statistics.mean(rounds_seen)
                                                lines.append(f"approx mean seconds per round={mean_per_round:.3f}s")

                outpath = Path(args.outfile)
                outpath.write_text("\n".join(lines))
                print(outpath)

if __name__ == "__main__":
                main()
