import os
import random
import sys
import time
import datetime
import threading
import torch

EXP_BASE_PATH = "/raid_sdb/home/tsy/KG-Agent"

def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    used_memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    all_memory = int(gpu_status[2].split('/')[-1].strip()[:-3])
    free_memory = all_memory - used_memory
    return power, free_memory


def waiting_gpu(interval=1, need_memory=100):
    # need_memory_lst = [28000]  # [12000, 20000]
    need_memory_lst = [need_memory]
    num_lst = [1]
    gid = [x for x in range(torch.cuda.device_count())]
    while True:
        for need_memory, num in zip(need_memory_lst, num_lst):
            candid_gid_lst = []
            for gpu_id in gid:
                gpu_power, free_gpu_memory = gpu_info(gpu_id)
                if free_gpu_memory >= need_memory and check_used_gpu_num():
                    if gpu_id not in candid_gid_lst:
                        candid_gid_lst.append((free_gpu_memory, gpu_id))

                if free_gpu_memory < need_memory and gpu_id in candid_gid_lst:
                    candid_gid_lst.remove((free_gpu_memory, gpu_id))

                gpu = 'gpu id:%d' % gpu_id
                gpu_power_str = 'gpu power:%d W |' % gpu_power
                gpu_memory_str = 'free memory:%d MiB |' % free_gpu_memory
                gpu_select_rule = 'memory=%d ; num=%d |' % (need_memory, num)
                sys.stdout.write(
                    '\r' + gpu + ' ' + gpu_memory_str + ' ' + gpu_power_str + gpu_select_rule + " candid_gid_lst:" + str(
                        candid_gid_lst) + " | " +
                    "waiting cmd ids:" + str(set(range(len(cmds))) - set(running_id) - set(finished_id)) + " | "
                                                                                                           "running cmd ids:" + str(
                        running_id)
                )
                sys.stdout.flush()

            if len(candid_gid_lst) >= num:
                if len(running_id) == 0:
                    candid_gid_lst.sort(reverse=True)
                else:
                    candid_gid_lst.sort()
                candid_gid_lst = [x[1] for x in candid_gid_lst[:num]]
                candid_gid_lst.sort()
                return candid_gid_lst

            time.sleep(interval)


def check_used_gpu_num():
    global running_id
    now_running_num = len(running_id)

    if (datetime.datetime.now().hour >= 23 or datetime.datetime.now().hour <= 8) and now_running_num <= 4:
        return True
    if now_running_num <= 5:
        return True
    else:
        return False


def ExecCmd(idx, cmd, gpu_ids):
    try:
        print("-------------------------------")
        cnt = random.randint(0, 1000)
        print(f"start running: {cmd})")
        print(f"time: {datetime.datetime.now()} ||| screen_id: {cmd_screen[idx]}_{cnt}")
        os.system(f"screen -dmS {cmd_screen[idx]}_{cnt}")
        os.system(f'screen -x -S {cmd_screen[idx]}_{cnt} -p 0 -X stuff "conda activate LLM\n" ')
        if "devices" not in cmd:
            cmd = f"CUDA_VISIBLE_DEVICES={str(gpu_ids)[1:-1].replace(' ', '')} " + cmd
            cmd += " --devices '%s'" % str(gpu_ids)

        # split to avoid "too long remote command" error
        cmd1 = " ".join(cmd.split(" ")[:len(cmd.split(" ")) // 2]) + " "
        cmd2 = " ".join(cmd.split(" ")[len(cmd.split(" ")) // 2:])
        assert cmd1 + cmd2 == cmd

        os.system(f'screen -x -S {cmd_screen[idx]}_{cnt} -p 0 -X stuff "{cmd1}"')
        os.system(f'screen -x -S {cmd_screen[idx]}_{cnt} -p 0 -X stuff "{cmd2}\n"')

        # running_id.append(idx)
        finished_id.append(idx)
        running_id.append(idx)

        finished = False
        error = False
        now_task_cmd = cmd

        time.sleep(10)
    except:
        return

    #     while (not finished) and (not error):
    #         with open("auto_run_log.txt", 'r') as f_log:
    #             finished_task = f_log.read()
    #             if now_task_cmd in finished_task:
    #                 finished = True

    #         with open("auto_error_log.txt", 'r+') as e_log:
    #             error_task = e_log.read()
    #             if now_task_cmd in error_task:
    #                 error = True
    #                 content = error_task.replace(now_task_cmd + "\n", "").replace(now_task_cmd, "")
    #                 with open("auto_error_log.txt", 'w') as tmp:
    #                     tmp.write(content)

    #         time.sleep(60 * 10)

    #     if finished:
    #         print("cmd %s is finished at %s" % (cmd, datetime.datetime.now()), "\n")
    #         running_id.remove(idx)
    #         finished_id.append(idx)
    #         with open("finished_cmd_log.txt", 'a+') as fc_log:
    #             fc_log.write(f"{datetime.datetime.now()}---{repr(cmd)}")
    #             fc_log.write('\n')
    #             fc_log.flush()
    #     elif error:
    #         running_id.remove(idx)
    # except:
    #     print('fail at:', idx)
    #     running_id.remove(idx)


global running_id
global finished_id
global cmd_screen
global cmds

if __name__ == '__main__':
    running_id = []
    finished_id = []
    cmds = [
        #---------------------------- run base ---------------------------
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+forced_filter' --max_step 10 "
        #     "--sample_data_num 100 "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'wqsp_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+forced_filter' --max_step 10 "
        #     "--sample_data_num 100 "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'cwq_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+forced_filter' --max_step 10 "
        #     "--sample_data_num 100 "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+forced_filter+long_mem' --max_step 10 "
        #     "--sample_data_num 100 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_kqa_v1+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+prompt_ToT+max_trips_10000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'wqsp_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+forced_filter+long_mem' --max_step 10 "
        #     "--sample_data_num 100 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_wqsp_v4+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs+forced_filter+long_mem' --max_step 10 "
        #     "--sample_data_num 100 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_kqa_v1+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+prompt_ToT+max_trips_10000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_kqa_v1+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+long_mem+prompt_ToT+max_trips_10000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'wqsp_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs+forced_filter+long_mem' --max_step 10 "
        #     "--sample_data_num 100 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_wqsp_v4+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_wqsp_v4+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+long_mem+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),


        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'cwq_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+forced_filter+long_mem' --max_step 10 "
        #     "--sample_data_num 100 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_cwq_v4+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'cwq_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs+forced_filter+long_mem' --max_step 10 "
        #     "--sample_data_num 100 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_cwq_v4+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_Llama-3.2-3B-instruct+data_cwq_v4+sample_100.0+task_kg_agent_QA_V1+exp_set_base+expand_attrs+forced_filter+long_mem+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),

        # after rebuttle run at 4.17
        (
            "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'kqa_v1' "
            "--task 'kg_agent_QA_V2' --exp_set 'base+expand_attrs' --max_step 10 --repeat_turn 3 "
        ),
        (
            "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'wqsp_v4' --max_trips 1000  "
            "--task 'kg_agent_QA_V2' --exp_set 'base+expand_attrs' --max_step 10 --repeat_turn 3 "
        ),
        (
            "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'cwq_v4' --max_trips 1000  "
            "--task 'kg_agent_QA_V2' --exp_set 'base+expand_attrs' --max_step 10 --repeat_turn 3 "
        ),
        
        (
            "python main.py --model_name 'Meta-Llama-3-8B-Instruct' --data_name 'kqa_v1' "
            "--task 'kg_agent_QA_V2' --exp_set 'base+expand_attrs' --max_step 10 --repeat_turn 3 "
        ),
        (
            "python main.py --model_name 'Meta-Llama-3-8B-Instruct' --data_name 'wqsp_v4' --max_trips 1000  "
            "--task 'kg_agent_QA_V2' --exp_set 'base+expand_attrs' --max_step 10 --repeat_turn 3 "
        ),
        (
            "python main.py --model_name 'Meta-Llama-3-8B-Instruct' --data_name 'cwq_v4' --max_trips 1000  "
            "--task 'kg_agent_QA_V2' --exp_set 'base+expand_attrs' --max_step 10 --repeat_turn 3 "
        ),
        #


        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs' --max_step 20 "
        # ),
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'wqsp_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs' --max_step 10 "
        # ),
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'cwq_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs' --max_step 15 "
        # ),

        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs' --max_step 20 "
        # ),
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'wqsp_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs' --max_step 10 "
        # ),
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'cwq_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs' --max_step 15 "
        # ),

        # long-mem
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+long_mem-top3-step3' --max_step 20 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_kqa_v1+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+long_mem-top3-step3' --max_step 10 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+long_mem-top3-step3' --max_step 15 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_cwq_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_15+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),


        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+long_mem' --max_step 20 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_kqa_v1+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+long_mem' --max_step 10 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1' --exp_set 'base+expand_attrs+long_mem' --max_step 15 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_cwq_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_15+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        # ),

        # -------------------------------------------------------------------------------
        
        # ----------------- re-run base TODO------------------------
        # done(
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs' --max_step 20 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_kqa_v1+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # no(
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs' --max_step 10 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # done(
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs' --max_step 15 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_cwq_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_15+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),

        # done(
        #     "python main.py --model_name 'gpt-4o' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs' --max_step 20 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_kqa_v1+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # done(
        #     "python main.py --model_name 'gpt-4o' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs' --max_step 10 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # done(
        #     "python main.py --model_name 'gpt-4o' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1_r1' --exp_set 'base+expand_attrs' --max_step 15 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_cwq_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_15+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),


        # long-mem
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1_r2' --exp_set 'base+expand_attrs+long_mem' --max_step 20 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_kqa_v1+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_kqa_v1+sample_-1+task_kg_agent_QA_V1_r1+exp_set_base+expand_attrs+long_mem+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1_r2' --exp_set 'base+expand_attrs+long_mem' --max_step 10 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1_r1+exp_set_base+expand_attrs+long_mem+prompt_ToT+max_trips_3000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1_r2' --exp_set 'base+expand_attrs+long_mem' --max_step 15 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_cwq_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_15+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_cwq_v4+sample_-1+task_kg_agent_QA_V1_r1+exp_set_base+expand_attrs+long_mem+prompt_ToT+max_trips_3000+max_step_15+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),

        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1_r2' --exp_set 'base+expand_attrs+long_mem' --max_step 20 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_kqa_v1+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_kqa_v1+sample_-1+task_kg_agent_QA_V1_r1+exp_set_base+expand_attrs+long_mem+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1_r2' --exp_set 'base+expand_attrs+long_mem' --max_step 10 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1_r1+exp_set_base+expand_attrs+long_mem+prompt_ToT+max_trips_3000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1_r2' --exp_set 'base+expand_attrs+long_mem' --max_step 15 "
        #     "--long_mem_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_cwq_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_15+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_cwq_v4+sample_-1+task_kg_agent_QA_V1_r1+exp_set_base+expand_attrs+long_mem+prompt_ToT+max_trips_3000+max_step_15+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'rerun' "
        # ),

        # -------------------- baselines ---------------------------------
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'kqa_v1' "
        #     "--task 'ToG_agent_QA' --exp_set 'base' "
        # ),
        #         (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'kqa_v1' "
        #     "--task 'GoG_agent_QA' --exp_set 'base' "
        # ),

        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'kqa_v1' "
        #     "--task 'ToG_agent_QA' --exp_set 'base' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'kqa_v1' "
        #     "--task 'GoG_agent_QA' --exp_set 'base' "
        # ),

        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'cwq_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1_baseqa' --exp_set 'qa' --max_step 10 "
        #     "--sample_data_num 100 "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'wqsp_v4' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1_baseqa' --exp_set 'qa' --max_step 10 "
        #     "--sample_data_num 100 "
        # ),
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'kqa_v1' --max_trips 1000  "
        #     "--task 'kg_agent_QA_V1_baseqa' --exp_set 'qa' --max_step 10 "
        #     "--sample_data_num 100 "
        # ),

        # ---------------------- ablation study --------------------------------------------
        # w/ short-mem vs. w/o short-mem
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1+no_short_mem' --exp_set 'base+expand_attrs' --max_step 20 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_kqa_v1+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'no_short_mem' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+no_short_mem' --exp_set 'base+expand_attrs' --max_step 10 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'no_short_mem' "
        # ),

        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'kqa_v1' "
        #     "--task 'kg_agent_QA_V1+no_short_mem --exp_set 'base+expand_attrs+long_mem' --max_step 20 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_kqa_v1+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_10000+max_step_20+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'no_short_mem' "
        # ),
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+no_short_mem' --exp_set 'base+expand_attrs+long_mem' --max_step 10 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_wqsp_v4+sample_-1+task_kg_agent_QA_V1+exp_set_base+expand_attrs+prompt_ToT+max_trips_1000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.9' "
        #     "--ablation_set 'no_short_mem' "
        # ),


        #----------------------------- run w/o short-mem REKGA ------------------
        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        # ),

        # (
        #     "python main.py --model_name 'Llama-3.2-3B-instruct' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        # ),

        # ----------------- w/o DKA ----------------
        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        # ),

        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        # ),

        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        # ),

        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        # ),
        
        # ------------- w/ DKA ------------------------------

        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA+DKA2_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_wqsp_v4+sample_0.1+task_kg_agent_QA_V1+REKGA+DKA_top10+exp_set_expand_attrs+prompt_ToT+max_trips_3000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.0' "
        #     "--ablation_set 'gen_only+rerun' "
        # ),

        # (
        #     "python main.py --model_name 'gpt-4o' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA+DKA2_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-4o+data_cwq_v4+sample_0.1+task_kg_agent_QA_V1+REKGA+DKA_top10+exp_set_expand_attrs+prompt_ToT+max_trips_3000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.0' "
        #     "--ablation_set 'gen_only+rerun' "
        # ),

        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'wqsp_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA+DKA2_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_wqsp_v4+sample_0.1+task_kg_agent_QA_V1+REKGA+DKA_top10+exp_set_expand_attrs+prompt_ToT+max_trips_3000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.0' "
        #     "--ablation_set 'gen_only+rerun' "
        # ),

        # (
        #     "python main.py --model_name 'gpt-3.5-turbo' --data_name 'cwq_v4' --max_trips 3000  "
        #     "--task 'kg_agent_QA_V1+REKGA+DKA2_top10' --exp_set 'expand_attrs' --max_step 10 --max_show_num 10 --answer_T 0.0 "
        #     "--sample_data_num 0.1 "
        #     "--ablation_from '/raid_sdb/tsy/KG-Agent/outputs/model_gpt-3.5-turbo+data_cwq_v4+sample_0.1+task_kg_agent_QA_V1+REKGA_top10+exp_set_expand_attrs+prompt_ToT+max_trips_3000+max_step_10+roll_back_max_score+roll_back_T_0.3+answer_T_0.0' "
        #     "--ablation_set 'gen_only+rerun' "
        # ),


    ]    
    GPU_USAGE_FOR_DATASET = {'3B':20000, '8B':35000}
    MAX_RUN_NUM = 3
    need_gpu_memory = 15000
    threads = []
    cmd_screen = [f"exp_{x}" for x in range(len(cmds))]
    print("begin run all cmds %s" % datetime.datetime.now())
    while (len(finished_id) != len(cmds)):
        for idx, cmd in enumerate(cmds):
            if idx in finished_id or idx in running_id:
                continue
            print("waiting gpus for cmd: ", idx, "||| ", cmd)
            print("now running cmd idx: ", running_id, "|||", "now waiting cmd num",
                  len(cmds) - len(finished_id) - len(running_id))
            for k, v in GPU_USAGE_FOR_DATASET.items():
                if k in cmd:
                    need_gpu_memory = v
                    break
            gpu_ids = waiting_gpu(need_memory=need_gpu_memory)
            th = threading.Thread(target=ExecCmd, args=(idx, cmd, gpu_ids,))
            th.start()
            threads.append(th)
            time.sleep(60*8)
            # while len(running_id) >= MAX_RUN_NUM:
            #     print(f"now running {len(running_id)} cmds, waiting 4 hour...")
            #     time.sleep(60*60*4)

            os.system('curl -d "user=2022010360&pass=Drgtklij12136819" "http://10.3.8.211/login"\n')
    # waiting all threads over
    for th in threads:
        th.join()

    print("all cmds over %s" % datetime.datetime.now())
    
    while(1):
        os.system('curl -d "user=2022010360&pass=Drgtklij12136819" "http://10.3.8.211/login"\n')
        time.sleep(60*60*1)

    
