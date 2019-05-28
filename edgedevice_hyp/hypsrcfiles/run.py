import sys
import time
import subprocess
# import os
# import signal


def run_cmd(cmd, arguments):
    cmd = cmd+ " "
    argstr  = ""
    argstr = cmd+ " ".join([str(i) for i in arguments[2:]])
    pr = subprocess.Popen("exec "+argstr, shell=True)
    print("test:",argstr)
    return pr


def kill_cmd(pr, cmd):
    print('killing ', cmd)
    #pr.terminate()
    pr.kill()
    pr.wait()


if __name__ == '__main__':
    print('~~~~~ start')
    proc_name = sys.argv[1] # python3
    arguments = sys.argv[2] #
    print("number of arguements", len(sys.argv)-1)
    sleep_interval = 190

    pr = run_cmd(proc_name, sys.argv)
    time.sleep(sleep_interval)
    kill_cmd(pr, proc_name)
