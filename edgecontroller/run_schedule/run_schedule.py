import sys
import subprocess 
import time
GPU_BOOT_TIME = 15
GPU_PROC_RUN_LIMIT = 180 


def run_cmd(cmd):
    print('[SCRIPT] running ', cmd)
    pr = subprocess.Popen("exec "+cmd, shell=True)
    return pr

def kill_cmd(pr, cmd):
    print ('[SCRIPT] killed ', cmd)
    pr.terminate()
    pr.wait()

if __name__ == '__main__':
    # loading all run command for gpu
    '''    
    with open('gpu_rc.txt') as f1:
        gpucmd  = f1.readlines()
    with open('cam1_rc.txt') as f2:
        cam1cmd  = f2.readlines()
    with open('cam2_rc.txt') as f3:
        cam2cmd  = f3.readlines()
    count1 = len(open('gpu_rc.txt').readlines(  )) 
    '''
    with open('gpu_rc_full.txt') as f1:
        gpucmd  = f1.readlines()
    with open('cam1_rc_full.txt') as f2:
        cam1cmd  = f2.readlines()
    with open('cam2_rc_full.txt') as f3:
        cam2cmd  = f3.readlines()
    count1 = len(open('gpu_rc_full.txt').readlines(  )) 
    for i in range (count1):
        # run for loop for each scheme.
        print('running... gpu command')
        pr1 = run_cmd(gpucmd[i])
        time.sleep(GPU_BOOT_TIME)
        print('running... cam1')
        pr2 = run_cmd (cam1cmd[i])
        print('running... cam2')
        pr3 = run_cmd (cam2cmd[i])
        time.sleep(GPU_PROC_RUN_LIMIT)
        kill_cmd(pr1, gpucmd[i])
        time.sleep(5)

