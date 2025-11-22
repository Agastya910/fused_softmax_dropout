#!/usr/bin/env python3
import subprocess, sys, json

def run_pytest():
    p = subprocess.run(["PYTHONPATH=.", "pytest", "-q"], shell=False, capture_output=True, text=True)
    out = p.stdout + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out

if __name__ == "__main__":
    code, out = run_pytest()
    print(out)
    # crude summary line parser (pytests prints a summary)
    for line in out.splitlines()[::-1]:
        if "passed" in line or "failed" in line:
            print("SUMMARY:", line)
            break
    sys.exit(code)
