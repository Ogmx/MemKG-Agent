import networkx as nx
import os, sys
import pickle
import pandas as pd
import json
from tqdm import tqdm, trange
import traceback
from typing_extensions import TypedDict, Literal, Dict, Any, List
import torch
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks.tracers import FunctionCallbackHandler
from loguru import logger

def get_trips_from_graph(G):
    trips = []
    for edge in G.edges:
        h,t = edge[0], edge[1]
        r = G[h][t]['rel']
        h = h[1] if h[1] != "" else h[0]
        t = t[1] if t[1] != "" else t[0]
    
        trips.append([h,r,t])
    return trips


def check_filename_available(filename):
    n = [0]
    def check_meta(file_name):
        file_name_new = file_name
        if os.path.isfile(file_name):
            file_name_new = file_name[:file_name.rfind('.')] + '_' + str(n[0]) + file_name[file_name.rfind('.'):]
            n[0] += 1
        if os.path.isfile(file_name_new):
            file_name_new = check_meta(file_name)
        return file_name_new

    return_name = check_meta(filename)
    return return_name


# Agent CallBacks

class LoggingHandler(FunctionCallbackHandler):
    """Tracer that logger to the file."""

    name: str = "logger_callback_handler"

    def __init__(self, logger, **kwargs: Any) -> None:
        super().__init__(function=logger.info, **kwargs)


def print_error(e):
    print(e)
    print(sys.exc_info()) 
    print(traceback.print_exc())
    print('\n','>>>' * 20)
    print(traceback.format_exc())

def save_results(args, output_path, cover=False):
    if cover:
        args.res_log.to_csv(f"{output_path}/results.csv")
    else:
        args.res_log.to_csv(check_filename_available(f"{output_path}/results.csv"))
    if 'long_term_memory' in args and args.long_term_memory:
        with open(f'{output_path}/long_mem.pkl', 'wb') as file:
            pickle.dump(args.long_term_memory, file)


def load_result(PATH):
    df = pd.read_csv(f"{PATH}/results.csv")
    with open(f"{PATH}/logs.json", "r") as f:
        logs = json.load(f)
    if 'metrics' in logs:
        logs = logs['metrics']
    if len(logs['h1']) == len(df):
        df['h1'] = logs['h1']
    else:
        idx = 0
        for i in range(len(df)):
            output = str(df.iloc[i]['output']).strip().lower()
            if output == 'error' or output == 'none':
                df.loc[i, 'h1'] = -1
                continue
            df.loc[i, 'h1'] = logs['h1'][idx]
            idx += 1
        assert idx == len(logs['h1'])
    return df


def cal_sim_score(query, cands, model=None, topk=20, return_idx=False, rerank=True):
    from langchain_openai import OpenAIEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import SentenceTransformer

    # if len(cands) < topk or model is None:
    #     if return_idx:
    #         return cands[:topk], list(range(len(cands[:topk])))
    #     else:
    #         return cands[:topk]
    topk_cands = [] 
    index = []
    if model is None:
         model = HuggingFaceEmbeddings(model_name="/raid_sdb/tsy/KG-Agent/models/stella_en_400M_v5", model_kwargs={"trust_remote_code":True, 
                                                                                                    "prompts":{"s2s_query": "Instruct: Retrieve semantically similar text.\nQuery: "}, "default_prompt_name":"s2s_query"})
    if len(cands) < topk:
        topk = len(cands)
        if not rerank:
            return cands
    if len(cands) > 0 and len(query) > 0:
        if not isinstance(query, list):
            query = [query]
        if isinstance(model, OpenAIEmbeddings) or isinstance(model, HuggingFaceEmbeddings):
            vectors = model.embed_documents(query + cands)
            q, rels = torch.tensor(vectors[0]), torch.tensor(vectors[1:])
            scores = q @ rels.T
            index = scores.topk(topk)[-1].tolist()
            topk_cands = [cands[k] for k in index]

    if return_idx:
        return topk_cands, index
    else:
        return topk_cands



def cal_sample_size(data_size):
    # based on "The RCSI Sample size handbook"
    # https://www.surveysystem.com/sscalc.htm

    z = 1.96    # Confidence Level: 95%
    p = 0.5     # percentage picking a choice
    c = 0.03    # Confidence Interval: ±3%

    ss = z**2 * p * (1-p) / c**2
    sample_num = ss / (1+((ss-1)/data_size))

    return int(sample_num)


#---------------------------- ablation study ------------------
def rebuild_retrieved_trips(df, i, env, topk=10):
    all_ents = set()
    try:
        logs = eval(df.iloc[i]['agent_log'])
    except:
        return [],[]
    
    query = df.iloc[i]['question']
    trips = list(logs['trip_set'])
    tmp_trips = [f"{x[0][-1], x[1], x[2][-1]}" for x in trips]
    _, index = cal_sim_score(query, tmp_trips, model=env.agent.emb, topk=topk, return_idx=True)
    ranked_trips = set([trips[idx] for idx in index])
    for trip in ranked_trips:
        h,r,t = trip
        if isinstance(h, tuple):
            all_ents.add(h)
        if isinstance(t, tuple):
            all_ents.add(t)
    
    return all_ents, ranked_trips

def expand_trip_sets(df, i, env):
    try:
        logs = eval(df.iloc[i]['agent_log'])
    except:
        return [],[]
    trip_set = logs['trip_set']
    ent_set = logs['ent_set']

    all_trips = set()
    all_ents = set()
    for trip in trip_set:
        h,r,t = trip
        h_ents = []
        t_ents = []
        try:
            if 'ent_set' in h:
                h = h.split(": ")[0]
                h_ents = ent_set[h]
            else:
                h_ents = [tuple(h.split(": "))]
            if 'ent_set' in t:
                t = t.split(": ")[0]
                t_ents = ent_set[t]
            else:
                t_ents = [tuple(t.split(": "))]

            for he in h_ents:
                for te in t_ents:
                    all_trips.add((he, r, te))
        except:
            continue

        if len(all_trips) > 100:        # retain top 100 trips
            tmp_trips = [f"{x[0][-1], x[1], x[2][-1]}" for x in all_trips][:1000]
            _, ids = cal_sim_score(env.state.cur_query, tmp_trips, env.agent.emb, topk=100, return_idx=True)
            all_trips = set([list(all_trips)[idx] for idx in ids])

    for trip in all_trips:
        h,r,t = trip
        if isinstance(h, tuple):
            all_ents.add(h)
        if isinstance(t, tuple):
            all_ents.add(t)
    return all_ents, all_trips