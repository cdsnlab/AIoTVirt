import sys
import subprocess 
import time
import argparse
GPU_BOOT_TIME = 15
GPU_PROC_RUN_LIMIT = 200


def run_cmd(cmd):
    print('[SCRIPT] running ', cmd)
    pr = subprocess.Popen("exec "+cmd, shell=True)
    return pr

def kill_cmd(pr, cmd):
    print ('[SCRIPT] killed ', cmd)
    pr.terminate()
    pr.wait()

def create_cam1_e1 (scheme):
    line = "ssh -p 40000 -t cdsn@143.248.55.237 "
    cdline = "cd /home/cdsn/spencer/chameleon/edgedevice_hyp/hypsrcfiles && python /home/cdsn/spencer/chameleon/edgedevice_hyp/hypsrcfiles/run.py python3 device_hypervisor.py -vf /home/cdsn/samplevideos/edge-node01_2046_480_60.mp4 -tr "
    cdline = cdline + str(scheme) +" -ln "+ scheme +"_cam1_2046_60.txt"
    final = line + "'" + cdline +"'"
    return final

def create_cam2_e1 (scheme):
    line = "ssh -p 50000 -t cdsn@143.248.55.237 "
    cdline = "cd /home/cdsn/spencer/chameleon/edgedevice_hyp/hypsrcfiles && python /home/cdsn/spencer/chameleon/edgedevice_hyp/hypsrcfiles/run.py python3 device_hypervisor.py -vf /home/cdsn/samplevideos/edge-node02_2046_480_60.mp4 -tr "
    cdline = cdline + str(scheme) +" -ln "+ scheme +"_cam2_2046_60.txt"
    final = line + "'" + cdline +"'"
    return final

def create_gpu_e1(scheme):
    line = "python3 gpu_controller.py -tr "
    line = line + scheme

    cdline = line + " -ln1 "+ scheme +"_cam1_2046_60.txt" + " -ln2 " + scheme + "_cam2_2046_60.txt" + " -ln4 " + scheme + "_gpu_2046_60.txt"
#    line = "python3 gpu_controller.py -tr e1g -ln1 e1g_cam1_2046_60.txt -ln2 e1g_cam2_2046_60.txt -ln4 e1g_gpu_2046_60.txt"
    return cdline

def create_cam1_logname(sk, fs):
    line = "ssh -p 40000 -t cdsn@143.248.55.237 "
    cdline = "cd /home/cdsn/spencer/chameleon/edgedevice_hyp/hypsrcfiles && python /home/cdsn/spencer/chameleon/edgedevice_hyp/hypsrcfiles/run.py python3 device_hypervisor.py -vf /home/cdsn/samplevideos/edge-node01_2046_480_60.mp4 -tr p "
    cdline = cdline + (" -ln " + "p_tsdr_sk"+str(sk)+"_fs"+str(fs)+"_cam1_2046_60.txt")
    final = line+ "'" + cdline + "'"
    return final
    
def create_cam2_logname(sk, fs):
    line = "ssh -p 50000 -t cdsn@143.248.55.237 "
    cdline = "cd /home/cdsn/spencer/chameleon/edgedevice_hyp/hypsrcfiles && python /home/cdsn/spencer/chameleon/edgedevice_hyp/hypsrcfiles/run.py python3 device_hypervisor.py -vf /home/cdsn/samplevideos/edge-node02_2046_480_60.mp4 -tr p "
    cdline = cdline + (" -ln " + "p_tsdr_sk"+str(sk)+"_fs"+str(fs)+"_cam2_2046_60.txt")
    final = line + "'" + cdline + "'"
    return final

def create_gpu_logname(sk, fs):
    line = "python3 gpu_controller.py -tr p -ts dr -ln1 p_tsdr_sk" +str(sk)+"_fs"+str(fs)+"_cam1_2046_60.txt" + " -ln2 p_tsdr_sk"+str(sk)+"_fs"+str(fs)+"_cam2_2046_60.txt" + " -ln4 p_tsdr_sk" +str(sk)+"_fs"+str(fs)+"_gpu_2046_60.txt"
#    line = "python3 gpu_controller.py -tr e1g -ln1 e1g_cam1_2046_60.txt -ln2 e1g_cam2_2046_60.txt -ln4 e1g_gpu_2046_60.txt"
    return line



    
    
if __name__ == '__main__':
    # loading all run command for gpu

    parser = argparse.ArgumentParser (description ="run in batch")
    parser.add_argument('-msk', '--maxsk', type=int, default = 100, help = "maximum sk value")
    parser.add_argument('-jsk', '--jumpsk', type=int, default = 5, help = "jump btw sk value")

    parser.add_argument('-mfs', '--maxfs', type=int, default = 20, help = "maximum fs value")
    parser.add_argument('-jfs', '--jumpfs', type=int, default = 5, help = "jump btw fs value")
    ARGS = parser.parse_args()

    print ("go through e1g")
    gcmd = create_gpu_e1("e1g")
    pr1 = run_cmd(gcmd)
    time.sleep(GPU_BOOT_TIME)
    c1cmd = create_cam1_e1("e1-1")
    pr2 = run_cmd(c1cmd)
    c2cmd = create_cam2_e1("e1-1")
    pr3 = run_cmd(c2cmd)
    time.sleep(GPU_PROC_RUN_LIMIT)
    kill_cmd(pr1, gcmd)
    time.sleep(10)
    
    print ("go through e1-1")
    gcmd = create_gpu_e1("e1-1")
    pr1 = run_cmd(gcmd)
    time.sleep(GPU_BOOT_TIME)
    c1cmd = create_cam1_e1("e1-1")
    pr2 = run_cmd(c1cmd)
    c2cmd = create_cam2_e1("e1-1")
    pr3 = run_cmd(c2cmd)
    time.sleep(GPU_PROC_RUN_LIMIT)
    kill_cmd(pr1, gcmd)
    time.sleep(10)
    
    print ("go through e1-2")
    gcmd = create_gpu_e1("e1-2")
    pr1 = run_cmd(gcmd)
    time.sleep(GPU_BOOT_TIME)
    c1cmd = create_cam1_e1("e1-2")
    pr2 = run_cmd(c1cmd)
    c2cmd = create_cam2_e1("e1-2")
    pr3 = run_cmd(c2cmd)
    time.sleep(GPU_PROC_RUN_LIMIT)
    kill_cmd(pr1, gcmd)
    time.sleep(10)
    print("go through p")    
    for i in range (2, ARGS.maxsk, ARGS.jumpsk):
        # run for loop for each scheme.
        print('running... gpu command')
        gcmd = create_gpu_logname(i, 20)
        pr1 = run_cmd(gcmd)
        time.sleep(GPU_BOOT_TIME)
        print('running... cam1')
        c1cmd = create_cam1_logname(i, 20)
        pr2 = run_cmd (c1cmd)
        print('running... cam2')
        c2cmd = create_cam2_logname(i, 20)
        pr3 = run_cmd (c2cmd)
        time.sleep(GPU_PROC_RUN_LIMIT)
        kill_cmd(pr1, gcmd)
        time.sleep(5)
