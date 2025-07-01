import os, sys
from importlib import import_module
import pickle, json
import pandas as pd
from argparse import ArgumentParser
import lightning as L
import networkx as nx
from tqdm import trange, tqdm
import time
from typing import Annotated
from typing_extensions import TypedDict, Literal

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain.globals import set_debug, set_verbose
from loguru import logger

from utils import print_error, save_results, cal_sample_size, get_trips_from_graph, check_filename_available, LoggingHandler
from eval import BM25Retrieve, eval_metrics
from prompts import prompt_dict, output_format_dict, load_result
from agent import Env



def run(args, df, agent, output_path):
    results = ["None" for _ in range(len(df))]
    state_log = None
    error_lst = set()
    done_lst = set()

    # continue from previous 
    if os.path.isfile(f"{output_path}/results.csv"):
        args.res_log = pd.read_csv(f"{output_path}/results.csv", index_col=0)
        for i in range(len(args.res_log)):
            idx = eval(args.res_log.iloc[i]['input'])['idx']
            output = args.res_log.iloc[i]['output']
            if output != 'None' and output != "ERROR":
                done_lst.add(idx)
                results[idx] = output
            else:
                error_lst.add(idx)
    if "rerun" in args.ablation_set:
        args.res_log = args.ab_res
        for i in range(len(args.ab_res)):
            idx = eval(args.ab_res.iloc[i]['input'])['idx']
            pre_h1 = args.ab_res.iloc[i]['h1']
            output = args.ab_res.iloc[i]['output']
            if pre_h1 == 1:
                done_lst.add(idx)
                results[idx] = output
            else:
                if idx in done_lst:
                    done_lst.remove(idx)
                if idx in error_lst:
                    error_lst.remove(idx)

    def _parser_sample(df, i, turn=1):
        question = df.iloc[i]['question']
        inputs = {'idx':i,
            'query': question,
            'topic_entities': eval(df.iloc[i]['entities']),
            }
        output_lst = []
        log_lst = []
        while turn:
            try:
                outputs = agent.run(inputs)
                state_log = {"actions": agent.state.actions, "history": agent.state.history, "score": [(x['step'], x['score']) for x in agent.memory.short_mem],
                             "kg":agent.state.get_edges(), "action_chain": agent.memory.action_chain, "ent_set": agent.state.set_to_ent_map, "trip_set": agent.state.trip_set,
                             "answers":agent.candidate_answers}
            except Exception as e:
                print_error(e)
                print(f"-----------------ERROR at sample {i}, skip it-----------------")
                error_lst.add(i)
                outputs = "ERROR"
                state_log = "NONE"
            output_lst.append(outputs)
            log_lst.append(state_log)
            turn -= 1
        outputs = f"[ANS]".join(output_lst)
        results[i] = outputs

        # save to log
        args.res_log.loc[i, 'question'] = question
        args.res_log.loc[i, 'answers'] = str(df.iloc[i]['answers'])
        args.res_log.loc[i, 'input'] = str(inputs)
        args.res_log.loc[i, 'output'] = str(outputs)
        args.res_log.loc[i, 'agent_log'] = str(state_log)

        # save long-term memory
        if agent.memory.long_mem['docs']:
            args.long_term_memory = agent.memory.long_mem['docs']

    print(f"\tfinished sample num: {len(done_lst)}\n\terror sample num:{len(error_lst)}\n")
    sleep_cnt = 0
    for i in trange(len(df)):
        if i in done_lst or i in error_lst:
            continue

        question = df.iloc[i]['question']
        if 'e2e' in args.task:
            if args.context == 'gold_query_graph':
                G = nx.DiGraph(eval(df.iloc[i]['query_graphs']))
                context = get_trips_from_graph(G)
                inputs = {"role":"user", 
                    "question": question,
                    "triples": context}
            if args.context == 'none':
                inputs = {"role":"user", 
                    "question": question}
            try:
                outputs = agent.invoke(inputs, config={'callbacks': [LoggingHandler(logger=logger)]})
                outputs = outputs['parsed'].dict()[args.output_format_key]
            except:
                print(f"find unformatted response at {i} sample")
                outputs = "NONE"
        if 'agent' in args.task:
            _parser_sample(df,i, turn=args.repeat_turn)
            
        # if (len(error_lst)-sleep_cnt+1) % 5 == 0:
        #     print(f"current error sample num: {len(error_lst)}, sleep 10 minutes...")
        #     sleep_cnt += 1
        #     time.sleep(60*10)

        if args.save_per_sample and (i+1) % args.save_per_sample == 0:
            print(f"-----------save results of {i}/{len(df)}------------")
            save_results(args, output_path=output_path, cover=True)
        if args.eval_per_sample and (i+1) % args.eval_per_sample == 0:
            print(f"-----------eval results of {i}/{len(df)}------------")
            metrics = eval_metrics(results, df, args)
            args.metrics = metrics

    print(f"\tfinished sample num: {len(done_lst)}\n\terror sample num:{len(error_lst)}\nNow retry error sample again...\n")
    # retry error sample again
    for k in tqdm(error_lst):
        _parser_sample(df,k)

    return results

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # Basic Setting
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--devices', default='[2]')
    parser.add_argument('--exp_name', default='TEST', type=str)
    parser.add_argument('--base_path', default="/raid_sdb/tsy/KG-Agent", type=str)
    parser.add_argument('--entity_linker_path', default="/raid_sdb/tsy/KG-Agent/models/entity_linker", type=str)
    parser.add_argument('--model_name', default='gpt-3.5-turbo', type=str) # llama3-70b-8192
    parser.add_argument('--data_name', default='wqsp_v4', type=str)
    parser.add_argument('--task', default='kg_agent_QA', type=str)
    parser.add_argument('--prompt', default='ToT', type=str)
    parser.add_argument('--context', default='none', type=str)
    parser.add_argument('--output_format', default='none', type=str)
    parser.add_argument('--repeat_turn', default=3, type=int)            # run n times for each sample
    parser.add_argument('--save_per_sample', default=50, type=int)      # save per x sample
    parser.add_argument('--eval_per_sample', default=50, type=int)      # eval per x sample
    
    # Agent Setting
    parser.add_argument('--exp_set', default='', type=str)          # short_mem(base), long_mem, plan, plan_sbs, reflect, forced_filter, expand_attrs
    parser.add_argument('--max_step', default=10, type=int)
    parser.add_argument('--max_trips', default=10000, type=int)      # max trips num save in ent_set    10000 for wikidata ; 3000 for freebase
    parser.add_argument('--max_get_trips', default=20, type=int)
    parser.add_argument('--sample_data_num', default=-1, type=float)
    parser.add_argument('--max_show_num', default=5, type=int)      # 5
    parser.add_argument('--roll_back', default='max_score', type=str)
    parser.add_argument('--roll_back_T', default=0.3, type=float)
    parser.add_argument('--answer_T', default=0.9, type=float)
    parser.add_argument('--long_mem_from', default=None, type=str)
    parser.add_argument('--trigger_tokens', default=None, type=str)

    parser.add_argument('--ablation_from', default=None, type=str)
    parser.add_argument('--ablation_set', default='', type=str)     # x_hop, no_qg,  rerun_error

    args = parser.parse_args()
    args.res_log = pd.DataFrame()
    args.param_log = {}
    #args.sample_data_num = int(args.sample_data_num)
    args.kg = 'freebase' if 'wqsp' in args.data_name or 'cwq' in args.data_name else "wikidata"

    if 'agent' in args.task:
        args.exp_name = f"model_{args.model_name}+data_{args.data_name}+sample_{args.sample_data_num}+task_{args.task}+exp_set_{args.exp_set}+prompt_{args.prompt}+max_trips_{args.max_trips}+max_step_{args.max_step}+roll_back_{args.roll_back}+roll_back_T_{args.roll_back_T}+answer_T_{args.answer_T}"
    else:   # single LLM
        args.exp_name = f"model_{args.model_name}+data_{args.data_name}+task_{args.task}+prompt_{args.prompt}+outformat_{args.output_format}"

    # Load Data
    df = pd.read_csv(f"{args.base_path}/data/{args.data_name}.csv")
    if args.sample_data_num != -1:
        if args.sample_data_num <= 1 and args.sample_data_num > 0:
            df = df.sample(int( len(df)*args.sample_data_num) )
        else:
            df = df.sample(int(args.sample_data_num))
    else:
        best_sample_size = cal_sample_size(len(df))     # 95% ± 3%
        df = df.sample(best_sample_size)
        
    if args.debug:
        print("run in debug mode...")
        args.devices = eval(args.devices)[0]
        df = df.sample(5)

        args.exp_name = "TEST"
    else:
        logger.remove(handler_id=None)  # not print

    output_path = f'{args.base_path}/outputs/{args.exp_name}'
    args.output_path = output_path
    # if not args.debug and os.path.exists(f"{output_path}/results.csv") and os.path.exists(f"{output_path}/logs.json"):
    #     print("This experiment already done , skip...")
    #     sys.exit()
    
    os.makedirs(output_path, exist_ok=True)
    logger.add(sink=f"{output_path}/run_log.log", colorize=True, enqueue=True)
    args.logger = logger
    # Load Agent
    if "e2e" in args.task:
        prompt = prompt_dict[args.prompt]
        llm = load_llm(model_name=args.model_name)
        output_format = output_format_dict[args.output_format]
        args.output_format_key = list(output_format.model_fields.keys())[-1]

        agent = prompt | llm.with_structured_output(output_format, include_raw=True, strict=False)

        print(agent)
        args.param_log['agent'] = str(agent)
    elif "GoG" in args.task:
        from other.GoG_api import GoG_agent
        agent = GoG_agent(args)
    elif "ToG" in args.task:
        from other.ToG_api import ToG_agent
        agent = ToG_agent(args)
    elif 'agent' in args.task:
        agent = Env(args)


    if os.path.exists(f"{args.ablation_from}/results.csv"):
        from utils import load_result
        print("start ablation ...")
        args.ab_res = load_result(args.ablation_from)
        
    # not args.debug and 
    if os.path.exists(f"{output_path}/results.csv"):
        print("Find generated results...")
        res = pd.read_csv(f"{output_path}/results.csv")
        # if len(res) == len(df):
        #     print("Already have full generated results, only evaluating...")
        #     results = load_result(res)
        # else:
        print("continue running...")
        results = run(args, df, agent, output_path)
        save_results(args, output_path, cover=True)
    else:
        # Run
        results = run(args, df, agent, output_path)
        save_results(args, output_path, cover=True)
        
    # Evaluate
    metrics = eval_metrics(results, df, args)
    args.param_log['metrics'] = metrics

    # Save Logs
    with open(check_filename_available(f"{output_path}/logs.json"), "w") as f:
        json.dump(args.param_log, f, indent=4)
