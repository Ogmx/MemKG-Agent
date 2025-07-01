import os, sys, time
import importlib
from queue import PriorityQueue
from copy import copy, deepcopy
import pickle
import re
import random
from tqdm import trange, tqdm
from typing import List, Set
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
import networkx as nx
import lightning as L
import timeout_decorator
from retry import retry
import openai
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from utils import print_error, cal_sim_score, rebuild_retrieved_trips, get_trips_from_graph, check_filename_available, LoggingHandler
# from interface.freebase_func import get_all_relations, get_all_entities, convert_id_to_name, get_types, process_middle_node, get_name_by_id
# from interface.wikidata_func import get_all_relations, get_all_entities, get_types, convert_id_to_name, convert_name_to_id, get_name_by_id


API_KEY = ""
BASE_URL = ""

MODEL_PATH = "/raid_sdb/tsy/LLMs"

class EntityLinker:
    def __init__(self, args):
        sys.path.append(args.entity_linker_path)
        from bert_entity_linker import BertEntityLinker
        import surface_index_memory
        print("loading entity linker...")
        self.surface_index = surface_index_memory.EntitySurfaceIndexMemory(
                f"{args.entity_linker_path}/data/entity_list_file_freebase_complete_all_mention", 
                f"{args.entity_linker_path}/data/surface_map_file_freebase_complete_all_mention",
                f"{args.entity_linker_path}/freebase_complete_all_mention")
        
        self.entity_linker = BertEntityLinker(self.surface_index, model_path=f"{args.entity_linker_path}/BERT_NER/trained_ner_model/", 
                                    device="cuda:0")

    def extract_entities(self, text) -> Set[Set]:
        if text.startswith("m.") or text.startswith("g."):
            return [(text, text, 1)]
        
        identified_entities = self.entity_linker.identify_entities(text)

        entities = set()
        for res in identified_entities:
            entities.add((res.entity.id, res.entity.name, res.entity.score))

        return entities


class Agent:

    def __init__(self, args, logger):
        from prompts import prompt_dict, output_format_dict
        self.logger = logger
        self.args = args
        self.llm, self.emb = self.load_llm()
        
        self.prompt_dict = prompt_dict
        self.format_dict = output_format_dict
        self.callbacks = [LoggingHandler(logger=self.logger)] if self.logger else []

    def forward(self, inputs: str, state, action):
        prompt, pattern = self.get_prompt(inputs, state, action)
        agent = self.llm
        cnt = 0
        try:
            inputs = [{"role":"user", "content": prompt}]
            res = self._call(agent, inputs)
            next_action, output = self.parse_res(res, pattern)
        except Exception as e:
            # print_error(e)
            cnt += 1
            print(f"API ERROR {cnt}")
            output = f"API ERROR: {e}"
            self.logger.info(f"\n-------------------------- API BUG --------------------\n{e}")
            next_action = "ERROR"


        self.logger.info("\n-------------------------- input --------------------\n" + prompt)
        self.logger.info("\n-------------------------- output --------------------\n" + output)

        return next_action, output
    
    
    OPENAI_ERRORS = (openai.Timeout, openai.InternalServerError, openai.APIError, openai.APIConnectionError)
    @retry(openai.RateLimitError, tries=2, jitter=30, delay=60*3, max_delay=60*5)
    @retry(OPENAI_ERRORS, tries=2, delay=60, max_delay=60*3)
    def _call(self, llm, input):
        res = llm.invoke(input, config={'callbacks': self.callbacks})
        return res
    
    def get_prompt(self, inputs:dict, state, cur_action):
        action2prompt_map = {"DEFAULT":'ToT_base_prompt' if 'base' in self.args.exp_set or 'short-mem' in self.args.exp_set else 'ToT_BASE_ACTION_SELECTION_PROMPT_NO_FILTER',
                     "THINK": 'ToT_base_prompt',
                     "PLAN": "ToT_DECOMPOSE_PROMPT",
                     "EXPAND_ENT": "ToT_expand_ent_prompt", 
                     "EXPAND_REL": "ToT_expand_rel_prompt",
                     "FILTER": "ToT_FILTER_ENTITY_PROMPT",
                     "EXTRACT_ANS": "ToT_EXTRACT_PROMPT",
                     "ANSWER": "ToT_answer_prompt",
                     "EVALUATE_STATE": "ToT_EVALUATE_STATE_PROMPT",
                     "EVALUATE_ANS": "ToT_EVALUATE_ANSWER_PROMPT",
                     "QA": "SIMPE_QA_PROPMT",
                    }
        cur_prompt = action2prompt_map[cur_action] if cur_action in action2prompt_map else None
        pattern = None
        try:
            current_prompt = self.prompt_dict[cur_prompt].format(**inputs)
        except:
            current_prompt = self.prompt_dict[cur_prompt]
        if 'qa' in self.args.exp_set:
            current_prompt = self.prompt_dict["SIMPE_QA_PROPMT"]
            full_prompt = current_prompt.format(
            question=state.query.replace("Original Query:",""))
        else:
            full_prompt = self.prompt_dict['ToT_full_prompt'].format(
                    query=state.query,
                    planning=state.planning,
                    kg_state=state.kg_to_text(self.emb, inputs),
                    trajectory=state.history_to_text() if "ANS" not in cur_action else "",
                    current_prompt=current_prompt)
        pattern = None
        if cur_action in ["DEFAULT", "EXPAND_ENT", "EXPAND_REL", "FILTER"] :
            cands = set()
            #sorted(, key=len, reverse=True)
            for k, v in inputs.items():  
                if k in ['options', 'entities', 'ent_sets','relations']:
                    for x in v:
                        cands.add(x)
            cands = sorted(list(cands), key=len, reverse=True)
            pattern = "(" + "|".join(cands).replace("(","\(").replace(")","\)") + ")"
        return full_prompt, pattern
    
    def parse_res(self, res, pattern) -> str:
        # with structured output
        if isinstance(res, dict):
            output = []
            for k,v in res['parsed'].dict().items():
                if isinstance(v, list):
                    for x in v:
                        output.append(x)
                else:
                    output.append(v)
            if pattern:
                match_results = "| ".join((pattern, str(output)))
            else:
                match_results = output
            output = '| '.join(output)
        # sentence output
        else:
            output = res.content
            if "assistant\n\n" in output:       # when use llama
                output = output.split("assistant\n\n")[0]

            if pattern:
                match_results = "| ".join(re.findall(pattern, str(output)))
            else:
                match_results = res

        return match_results, output

    def load_llm(self):
        if 'gpt-3.5' in self.args.model_name or 'gpt-4' in self.args.model_name:
            llm = BaseChatOpenAI(model_name=self.args.model_name,
                    temperature=0,
                    top_p=0.2,
                    api_key = API_KEY,
                    base_url = BASE_URL,
                    max_tokens=256,
                    seed=42,
                    timeout=60*2,
                    max_retries=0,
            )
        else:
            llm = HuggingFacePipeline.from_model_id(
                model_id=f"{MODEL_PATH}/{self.args.model_name}",
                task="text-generation",
                device=0 if not self.args.debug else self.args.devices,
                batch_size=1,
                pipeline_kwargs={"max_new_tokens": 256, 'return_full_text': False},
                model_kwargs={'temperature':0, }
            )
            llm = ChatHuggingFace(llm=llm)

        embeddings = HuggingFaceEmbeddings(model_name=f"{self.args.base_path}/models/stella_en_400M_v5", model_kwargs={"trust_remote_code":True, 
                                                                                                                "device":self.args.devices if self.args.debug else 0,
                                                                                                                "prompts":{"s2s_query": "Instruct: Retrieve semantically similar text.\nQuery: "}, "default_prompt_name":"s2s_query"})

        # embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
        #                               api_key = API_KEY,
        #                               base_url = BASE_URL)
        return llm, embeddings

    
class State:
    def __init__(self, data=None, args=None):
        self.query = data['query'] if data else None
        self.cur_query = self.query
        self.topic_entities : List = data['topic_entities'] if data else []
        self.max_show_num = args.max_show_num
        self.exp_set = args.exp_set
        self.trigger_tokens = args.trigger_tokens
        self.history = []
        self.actions = []
        self.planning = ""
        self.used_edge = set()
        self.used_node = set()
        self.used_edge_score = {}
        self.unavailable_node = set()
        self.topk_ents = set()

        self.trip_set: List[dict] = []      # whole edge list without filter
        self.set_to_ent_map = {}
        self.ent_to_set_map = {}
        self.sub_kg = nx.DiGraph()
        for x in self.topic_entities:
            self.sub_kg.add_node(x)

        # self._update_trip_set(ents=self.topic_entities)
    
    def _update_trip_set(self, ents, trips=[], h=None, r=None):
        tmp = {"head":h,
               "rel":r,
               "ents":set(),
                "trips":set()}
        
        pervious_ents = set()
        pervious_trips = set()
        for dic in self.trip_set:
            pervious_ents.update(dic['ents'])
            pervious_trips.update(dic['trips'])

        for ent in ents:
            # if ent not in self.used_node and ent not in pervious_ents:
            tmp['ents'].add(ent)

        for trip in trips:
            h,r,t = trip
            if h in tmp['ents'] or t in tmp['ents']: # and trip not in pervious_trips:
                tmp['trips'].add(trip)

        tmp['ent_num'] = len(tmp['ents'])
        tmp['trip_num'] = len(tmp['trips'])

        self.trip_set.append(tmp)

    def update(self, cur_action, obs=None, output=None):
        self.actions.append(cur_action)
        if cur_action == "PLAN":
            self.planning = obs
        elif obs:
            self.history.append(obs)
        if output:
            if 'source_ents' in output:
                for ent in output['source_ents']:         # update used node & edge
                    self.used_node.add(ent)
                    for rel in output['source_rels']:
                        self.used_edge.add((ent, rel))
            if 'used_func' in output:
                self.used_edge.add(output['used_func'])
    
    def update_kg(self, new_trips, ents, rels=None, use_attr_trips=True, state=None, env=None):
        # ent_set_trips = set()
        # source_to_set_map = {}
        # tail_to_set_map = {}
        # set_to_ent_map = self.set_to_ent_map.copy() if not state else state.set_to_ent_map.copy()
        # ent_to_set_map = self.ent_to_set_map.copy() if not state else state.ent_to_set_map.copy()
        # query_graph_to_ent_set_map = {}


        # extract all trips and attributes
        def extract_trips_and_attrs(new_trips, ents, rels, env):
            forward_trips = set()
            backward_trips = set()
            forward_attrs = set()
            backward_attrs = set()
            all_rels = set()
            for trip in new_trips:
                h,r,t = trip
                r = rels if rels else r
                all_rels.add(r)
                if '[' in h[0] and '| ' in h[0] and t[0] in ents:
                    qh, qr, qt = h[0].replace("[","").replace("]","").split("| ")
                    backward_attrs.add(((qh, api.get_name_by_id(qh, env)), qr, (qt, api.get_name_by_id(qt, env))))
                    all_rels.add(qr)

                elif '[' in t[0] and '| ' in t[0] and h[0] in ents:
                    qh, qr, qt = t[0].replace("[","").replace("]","").split("| ")
                    forward_attrs.add(( (qh, api.get_name_by_id(qh, env)), qr, (qt, api.get_name_by_id(qt, env)) ))
                    all_rels.add(qr)

                elif h[0] in ents:
                    forward_trips.add((h,r,t))
                elif t[0] in ents:
                    backward_trips.add((t,r,h))

            top_rels = cal_sim_score(self.cur_query, list(all_rels), model=env.agent.emb, topk=10, rerank=False)

            return forward_trips, backward_trips, forward_attrs, backward_attrs, top_rels

        def build_ent_set_trips(all_trips, ents, di='forward', top_rels=[]):
            trip_info = {'h->id':{}, 't->id':{}, 'qg->hid':{}, 'qg->tid':{}, 'id->ent':{}, 'ent->id':{}, 'trip->dir':{}}
            # build ent to id maps
            for trip in all_trips:
                h, r, t = trip
                if h[0] not in ents or r not in top_rels:
                    continue
                if (h,r) not in trip_info['h->id']:
                    trip_info['h->id'][(h,r)] = max([(len(self.set_to_ent_map[idx]), idx) for idx in self.ent_to_set_map[h]])[-1] if h in self.ent_to_set_map else h
                qg = (trip_info['h->id'][(h,r)], r)
                trip_info['qg->hid'][qg] = trip_info['qg->hid'][qg] if qg in trip_info['qg->hid'] else len(trip_info['qg->hid'])
                trip_info['qg->tid'][qg] = trip_info['qg->tid'][qg] if qg in trip_info['qg->tid'] else f"ent_set_{len(trip_info['qg->tid'])+len(self.set_to_ent_map)}"
                trip_info['t->id'][(qg,t)] = trip_info['qg->tid'][(qg)]
                trip_info["id->ent"].setdefault(trip_info['h->id'][(h,r)], set()).add(h)
                trip_info["id->ent"].setdefault(trip_info['t->id'][(qg,t)], set()).add(t)
                trip_info['trip->dir'][trip] = di


            # union same and build sub_set with exist ent_set
            exist_ents_to_idx_map = {}
            for k,v in self.set_to_ent_map.items():
                v = str(sorted(list(v)))
                if v not in exist_ents_to_idx_map:                      # first appear
                    exist_ents_to_idx_map[v] =  k

            changed_map = {}
            for cur_id, cur_ent_set in trip_info['id->ent'].items():
                cur_ent_set = str(sorted(list(cur_ent_set)))
                if cur_ent_set in exist_ents_to_idx_map:                # same with exist ent_set
                    exist_idx = exist_ents_to_idx_map[cur_ent_set]
                    for h_ent, h_id in trip_info['h->id'].items():
                        if h_id == cur_id and h_id != exist_idx:
                            trip_info['h->id'][h_ent] = exist_idx
                            changed_map[h_id] =exist_idx
                    for t_ent, t_id in trip_info['t->id'].items():
                        if t_id == cur_id and t_id != exist_idx:
                            trip_info['t->id'][t_ent] = exist_idx
                            changed_map[t_id] = exist_idx
                elif cur_id not in self.set_to_ent_map:      # is a new ent_set
                    continue
                else:                                   # is a sub_set of  exist ent_set
                    for cnt in range(100):
                        new_idx = f"{cur_id}-{cnt}"
                        if new_idx not in self.set_to_ent_map:
                            break
                    for h_ent, h_id in trip_info['h->id'].items():
                        if h_id == cur_id:
                            trip_info['h->id'][h_ent] = new_idx
                            changed_map[h_id] = new_idx
                    for t_ent, t_id in trip_info['t->id'].items():
                        if t_id == cur_id:
                            trip_info['t->id'][t_ent] = new_idx
                            changed_map[t_id] = new_idx

            # rebuild t->id map
            tmp_dic = {}
            drop_lst = []
            for k, v in trip_info['t->id'].items():
                qg, t = k
                h,r = qg
                if qg[0] in changed_map.keys():
                    new_tid = ((changed_map[qg[0]], qg[1]), t)
                    tmp_dic[new_tid] = trip_info['t->id'][k]
                    drop_lst.append(k)
                elif qg[1] in changed_map.keys():
                    new_tid = ((qg[0], changed_map[qg[1]]), t)
                    tmp_dic[new_tid] = trip_info['t->id'][k]
                    drop_lst.append(k)
            trip_info['t->id'].update(tmp_dic)
            for k in drop_lst:
                trip_info['t->id'].pop(k)

            # rebuild
            trip_info['qg->tid'] = {}
            trip_info['id->ent'] = {}
            for trip in all_trips:
                h, r, t = trip
                if h[0] not in ents or r not in top_rels:
                    continue
                qg = (trip_info['h->id'][(h,r)], r)
                if qg in trip_info['qg->tid']:
                    trip_info['qg->tid'][qg] = trip_info['qg->tid'][qg]  
                else:
                    new_idx = f"ent_set_{len(trip_info['qg->tid'])+len(self.set_to_ent_map)}"
                    trip_info['qg->tid'][qg] = changed_map[new_idx] if new_idx in changed_map else new_idx
                trip_info['id->ent'].setdefault(trip_info['h->id'][(h,r)], set()).add(h)
                trip_info['id->ent'].setdefault(trip_info['t->id'][(qg,t)], set()).add(t)

            # build ent set trips
            def get_representation(trip_info, id_dict, key, ent):
                if len(trip_info['id->ent'][id_dict[key]]) > 1:
                    return id_dict[key]
                elif len(trip_info['id->ent'][id_dict[key]]) == 1 and 'ent_set' in str(id_dict[key]):
                    return f"{id_dict[key]}: {ent[0]}: {ent[1]}" if ent[0] != ent[1] else ent[1]
                else:
                    return f"{ent[0]}: {ent[1]}" if ent[0] != ent[1] else ent[1]
                
            ent_set_trips = set()
            for trip in all_trips:
                h,r,t = trip
                if h[0] not in ents or r not in top_rels:
                    continue
                qg = (trip_info['h->id'][(h,r)], r)
                hw = get_representation(trip_info, trip_info['h->id'], (h,r), h)
                tw = get_representation(trip_info, trip_info['t->id'], (qg,t), t)
                if di == 'forward':
                    ent_set_trips.add((hw, r, tw))
                else:
                    ent_set_trips.add((tw, r, hw))

            # update global info
            for k,v in trip_info["id->ent"].items():
                if 'ent_set' in k:
                    if k not in self.set_to_ent_map:
                        self.set_to_ent_map[k] = v
                    for e in v:
                        if e not in self.ent_to_set_map:
                            self.ent_to_set_map[e] = [k]
                        else:
                            if k not in self.ent_to_set_map[e]:
                                self.ent_to_set_map[e].append(k)

            return ent_set_trips, trip_info
        
        forward_trips, backward_trips, forward_attrs, backward_attrs, top_rels = extract_trips_and_attrs(new_trips, ents, rels, env=env)

        if 'base' in self.exp_set or "short-mem" in self.exp_set:       # using short-mem to compressed the explored triples
            # process forward trips and attrs
            all_ent_set_trips = set()
            ent_set_trips, trip_info = build_ent_set_trips(forward_trips, ents, di='forward', top_rels=top_rels)
            all_ent_set_trips.update(ent_set_trips)
            if use_attr_trips:
                tail_ents = list(set([x[-1][0] for x in trip_info['t->id']]))
                attr_set_trips,_ = build_ent_set_trips(forward_attrs, list(tail_ents), top_rels=top_rels)
                all_ent_set_trips.update(attr_set_trips)
            else:
                q_rels = set([x[1] for x in forward_attrs])
                tail_ent_set = list(trip_info['t->id'].values())
                if q_rels and tail_ent_set:
                    tail_ent_set = list(trip_info['t->id'].values())[0]
                    all_ent_set_trips.add((tail_ent_set, "Filterable by", f"{q_rels}"))

            # process backward trips and attrs
            ent_set_trips, trip_info = build_ent_set_trips(backward_trips, ents, di='backward', top_rels=top_rels)
            all_ent_set_trips.update(ent_set_trips)
            if use_attr_trips:
                tail_ents = list(set([x[-1][0] for x in trip_info['t->id']]))
                attr_set_trips,_ = build_ent_set_trips(backward_attrs, list(tail_ents), di='backward', top_rels=top_rels)
                all_ent_set_trips.update(attr_set_trips)
            else:
                q_rels = set([x[1] for x in backward_attrs])
                tail_ent_set = list(trip_info['t->id'].values())
                if q_rels and tail_ent_set: 
                    tail_ent_set = list(trip_info['t->id'].values())[0]
                    all_ent_set_trips.add((tail_ent_set, "Filterable by", f"{q_rels}"))


            for trip in sorted(list(all_ent_set_trips)):
                if trip not in self.trip_set:
                    self.trip_set.append(trip)

        else:   # not using short-mem
            tail_ents = set()
            all_new_trips = set()
            all_new_ents = set()
            for trips in [forward_trips, backward_trips, forward_attrs, backward_attrs]:
                for trip in trips:
                    h,r,t = trip
                    all_new_trips.add((h,r,t))
                    all_new_ents.add(h)
                    all_new_ents.add(t)

            for trip in all_new_trips:
                if trip not in self.trip_set:
                    self.trip_set.append(trip)
            if 'explored triples' not in self.set_to_ent_map:
                self.set_to_ent_map['explored triples'] = []

            for ent in all_new_ents:
                if ent not in self.set_to_ent_map['explored triples']:
                    self.set_to_ent_map['explored triples'].append(ent)


    def replace_sub_KG(self, trips=None, ents=None, kg=None, trip_set=None):
        self.sub_kg = nx.DiGraph()
        if isinstance(kg, nx.DiGraph):
            self.sub_kg = kg
        else:
            for x in self.topic_entities:
                self.sub_kg.add_node(x)
            for node in ents:
                self.sub_kg.add_node(node)
            for trip in trips:
                h,r,t = trip
                self.sub_kg.add_node(h)
                self.sub_kg.add_node(t)
                self.sub_kg.add_edge(h, t, rel=r)
        if trip_set:
            for t in trip_set:
                self.trip_set.append(t)
        else:
            self._update_trip_set(ents=ents, trips=trips)
    
    def get_entities(self, drop_used=False, only_entity=False):
        ents = set()
        for node in self.topic_entities:
            if only_entity and not node[0].startswith("m.") and not node[0].startswith("Q"):
                continue
            if node[0] in self.unavailable_node:
                continue
            ents.add(node)
        ent_sets = set()
        for ent_set in self.set_to_ent_map:
            if 'ent_set' in ent_set:
                ent_sets.add(ent_set)
        for ent in self.topk_ents:
            if ent[0].startswith("Q") or ent[0].startswith("m."):
                ents.add(ent)
        # for node in self.sub_kg.nodes():
        #     if drop_used and node[0] in self.used_node:
        #         continue
        #     if only_entity and not node[0].startswith("m.") and not node[0].startswith("Q"):
        #         continue
        #     ents.add(node)
        return list(ents), list(ent_sets)
    
    def get_edges(self):
        trips = []
        tmp = set()
        for edge in self.sub_kg.edges():
            h,t = edge
            r = self.sub_kg[h][t]['rel']
            if (h,r,t) in tmp or (t,r,h) in tmp:
                continue
            tmp.add((h,r,t))
            tmp.add((t,r,h))
            trips.append([h,r,t])

        return trips
    
    def kg_to_text(self, emb=None, inputs=[]):
        #trips = self.get_edges()
        #ents, ent_sets = self.get_entities(only_entity=True)
        max_show_num = self.max_show_num

        expand_ent_set_ids = []
        expand_trip_set = []
        if 'ent_set' in inputs:
            expand_ent_set_ids, expand_ent_sets = inputs['ent_set']
        elif 'trip_set' in inputs:
            expand_trip_set = inputs['trip_set']

        ents_str = "Knowledge Graph Entities:\n"
        if len(self.topic_entities):
            ents_str += f"\ttopic entities:\n"
            for ent in self.topic_entities:
                ents_str += f"\t\t{ent[0]}: {ent[1]}\n"

        if len(self.set_to_ent_map):
            # set_to_ent_lst = sorted(self.set_to_ent_map)
            for k in self.set_to_ent_map:
                v = self.set_to_ent_map[k]
                if len(v) == 1:             # not show ent_set with only one entity
                    continue
                if "-" in k:
                    ents_str += f"\t{k}: contain {len(v)} entities; is a subset of {'-'.join(k.split('-')[:-1])}\n"
                else:
                    ents_str += f"\t{k}: contain {len(v)} entities\n"
                if k == expand_ent_set_ids:
                    for ent in expand_ent_sets:
                        ents_str += f"\t\t{ent[0]}: {ent[1]}\n"
                    if len(v)-len(expand_ent_sets) > 0:
                        ents_str += f"\t\t... and other {len(v)-len(expand_ent_sets)} entities\n"
                else:
                    _, top_k_ents = self.rank_show_ents(emb=emb, ents=list(v), topk=max_show_num)
                    for ent in top_k_ents:
                        ents_str += f"\t\t{ent[0]}: {ent[1]}\n" if len(ent)>=2 else f"\t\t{ent[0]}\n"
                        self.topk_ents.add(ent)
                    if len(v)-max_show_num > 0:
                        ents_str += f"\t\t... and other {len(v)-max_show_num} entities\n"


        trip_str = "Knowledge Graph Edges:\n"
        if len(self.trip_set):
            if 'base' not in self.exp_set and 'short-mem' not in self.exp_set:
                top_k_trips, _ = self.rank_show_ents(emb=emb, trips=list(self.trip_set), topk=max_show_num)
                show_trip_sets = top_k_trips
            else:
                show_trip_sets = self.trip_set
            for trip in show_trip_sets:
                h,r,t = trip
                if self.trigger_tokens:
                    T = eval(self.trigger_tokens)
                    cur_str = f"\t{T[0]}{h[-1] if isinstance(h, tuple) else h}{T[1]}{r}{T[2]}{t[-1] if isinstance(t, tuple) else t}{T[3]}\n"
                else:
                    cur_str = f"\t({h[-1] if isinstance(h, tuple) else h}, {r}, {t[-1] if isinstance(t, tuple) else t})\n"
                if cur_str == expand_trip_set:
                    trip_str += f"{cur_str}; This could be helpful!\n"
                else:
                    trip_str += cur_str

        # if len(self.trip_set):
        #     ent_set_str = "Entity Sets:\n"
        #     for idx, dic in enumerate(self.trip_set):
        #         desc = f" about the {dic['rel']} of {dic['head']} or whose {dic['rel']} is {dic['head']}" if dic['rel'] and dic['head'] else ""
        #         ent_set_str = ent_set_str + f"ent_set_{idx}: contain {dic['ent_num']} entities" + desc + '\n'
        # else:
        #     ent_set_str = "Entity Sets:\nEMPTY"
        # if len(ents):
        #     ents_str = "Knowledge Graph Entities:\n{}".format('\n'.join([f": ".join(ent) for ent in ents]))
        # else:
        #     ents_str = "Knowledge Graph Entities:\nEMPTY"

        # if len(trips):
        #     edges_str = "Knowledge Graph Edges:\n{}".format("\n".join([f"({trip[0][-1]}, {trip[1]}, {trip[2][-1]})" for trip in trips]))
        # else:
        #     edges_str =  "Knowledge Graph Edges:\nEMPTY"

        # ents_str = "Knowledge Graph Entities:\n"
        # trip_str = "Knowledge Graph Edges:\n"

        # for idx, dic in enumerate(self.trip_set):
        #     ents_str += f"ent_set_{idx}: have {dic['ent_num']} entities; {dic['from']}\n"
        #     trip_str += f"trip_set_{idx}: have {dic['trip_num']} triples; {dic['from']}\n"
        #     ent_lst = list(dic['ents'])
        #     trip_lst = list(dic['trips'])
        #     if len(trip_lst) > max_show_num:
        #         trip_lst, ent_lst = self.get_topk(emb=emb, trips=trip_lst, ents=ent_lst, topk=max_show_num)

        #     for ent in ent_lst:
        #         ents_str += f"  {ent[0]}: {ent[-1]}\n"
        #     for trip in trip_lst:
        #         trip_str += f"  ({trip[0][-1]}, {trip[1]}, {trip[2][-1]})\n"

        #     if len(list(dic['ents']))-max_show_num > 0:
        #         ents_str += f"  ... and other {len(list(dic['ents']))-max_show_num} entities\n"
        #     if len(list(dic['trips']))-max_show_num > 0:
        #         trip_str += f"  ... and other {len(list(dic['trips']))-max_show_num} triples\n"

        return ents_str + '\n\n' + trip_str
    
    def rank_show_ents(self, emb=None, trips=[], ents=[], topk=10):
        if trips:   # get topk trips
            tmp_trips = [f"{x[0][-1], x[1], x[2][-1]}" for x in trips]
            _, index = cal_sim_score(self.query, tmp_trips, model=emb, topk=topk, return_idx=True)
            trips = [trips[idx] for idx in index]
        
        if ents:
            tmp_ents = [f"{x[-1]}" for x in ents]
            _, ids = cal_sim_score(self.query, tmp_ents, model=emb, topk=topk, return_idx=True)
            ents = [ents[idx] for idx in ids]

        return trips, ents

    def history_to_text(self):
        his_str = ""
        for his in self.history:
            his = his.replace('\n','\n\t')
            his_str += f"\t{his}\n"
        return his_str

class Memory:
    def __init__(self, args, emb):
        self.args = args
        self.emb = emb
        self.long_mem = self.load_long_memory()                # for whole dataset

        self.short_mem : List[dict] = []                       # for a question
        self.action_chain = []
        self.sub_questions = []

        self.trip_set: List[dict] = []   
        self.set_to_ent_map = {}
        self.ent_to_set_map = {}

    def init_memory(self):          # init short memory at each sample
        self.short_mem = []
        self.action_chain = []
        self.sub_questions = []
        self.trip_set: List[dict] = []   
        self.set_to_ent_map = {}
        self.ent_to_set_map = {}

    def load_long_memory(self):
        embeddings = self.emb
        if os.path.exists(f"{self.args.long_mem_from}/chroma.sqlite3"):
            print("loading processed vector store...")
            with open(f"{self.args.long_mem_from}/long_mem.pkl", 'rb') as file:
                documents = pickle.load(file)
            vector_store = Chroma(
                collection_name="long_term_memory",
                embedding_function=embeddings,
                persist_directory=self.args.long_mem_from
            )

        elif os.path.exists(f"{self.args.long_mem_from}/long_mem.pkl"):
            print("building vector store from docs...")
            with open(f"{self.args.long_mem_from}/long_mem.pkl", 'rb') as file:
                documents = pickle.load(file)
            vector_store = Chroma(
                collection_name="long_term_memory",
                embedding_function=embeddings,
                persist_directory=self.args.long_mem_from,
            )
            vector_store.add_documents(ids=[str(i) for i in range(len(documents))],
                                                   documents=documents)
        else:
            if os.path.exists(f"{self.args.output_path}/long_mem.pkl"):
                print("loading exist long-mem to continue...")
                with open(f"{self.args.output_path}/long_mem.pkl", 'rb') as file:
                    documents = pickle.load(file)
                vector_store = None
            else:
                documents = []
                vector_store = None

        return {"emb":embeddings, "vector_store":vector_store, "docs":documents}

    def update_global_kg(self, state):
        self.ent_to_set_map = state.ent_to_set_map
        self.set_to_ent_map = state.set_to_ent_map

    def update_short_mem(self, step, state, score):
        if 'rollback' in self.args.exp_set:
            self.short_mem.append({"step": step,
                               "state": deepcopy(state),
                               "score": score})
    
    def convert_short_to_long_mem(self, state, env=None):
        query = state.cur_query
        schema = self.get_schema(state, query, env)
        doc = Document(page_content=query,
                       metadata={'schema':str(schema),
                                 'pre_ans':str(env.candidate_answers)})
        self.long_mem['docs'].append(doc)

    def recall_from_long_mem(self, query, k):
        res = self.long_mem['vector_store'].similarity_search(query, 
                                                              k = k * self.args.repeat_turn)

        recall_schemas = []
        for i in range(len(res)):
            tmp = {'query':res[i].page_content,
                   'schemas':[]}
            schemas = eval(res[i].metadata['schema'])
            ans_score = eval(res[i].metadata['pre_ans'])[-1][0] * -1
            if ans_score < 0.5:     # skip low confidence / score resutls
                continue
            for step, schema in enumerate(schemas):
                h,r,t = schema
                tmp['schemas'].append({
                    "query": res[i].page_content,
                    "head_ents": h[0],
                    "head_ent_typs":h[1],
                    "rel":r,
                    "tail_ents":t[0],
                    "tail_ent_typs":t[1],
                    "step":step,
                })
            recall_schemas.append(tmp)
        return recall_schemas
    
    def get_schema(self, state, query, env=None):
        schema = []
        trips = state.trip_set

        # convert path to schema
        def get_typ_lst(lst, field=None):
            if isinstance(lst, str):
                lst = [lst]
            all_typs = set()
            for x in lst:
                if "ent_set" in x:
                    ents = list(state.set_to_ent_map[x])
                    tmp = [ent[1] for ent in ents]
                    _, ids = cal_sim_score(query, tmp, env.agent.emb, topk=5, return_idx=True)
                    ents = [ents[id] for id in ids]
                    for ent in ents:
                        typs = api.get_types(ent[0])
                        for typ in typs:
                            all_typs.add(typ)
                elif x.startswith("Q") or x.startswith("m."):
                    typs = api.get_types(x)
                    for typ in typs:
                        if field:
                            for t in typ.split("."):
                                if t in field:          # if any domain match
                                    all_typs.add(typ)
                        else:
                            all_typs.add(typ)
            return all_typs
        
        for trip in trips:
            h,r,t = trip
            hid = h.split(": ")[0]
            tid = t.split(": ")[0]
            try:
                h_typs = get_typ_lst(hid)
                t_typs = get_typ_lst(tid)
                schema.append([[hid, h_typs], r, [tid, t_typs]])
            except:
                continue

        return schema

class Env:
    def __init__(self, args) -> None:
        self.args = args
        if self.args.kg == 'freebase':
            with open(f"{args.base_path}/data/{args.data_name.split('_')[0]}_mid2name.pkl", 'rb') as f:
                id2name = pickle.load(f)
            self.id2name = id2name
        else:
            self.id2name = {}

        global api                      # import KG interface
        if args.kg == 'freebase':
            api = importlib.import_module('interface.freebase_func')
        elif args.kg == 'wikidata':
            api = importlib.import_module('interface.wikidata_kqa_func')

        self.logger = args.logger
        self.entity_linker = None

        self.agent : Agent = Agent(args, logger=self.logger)
        self.memory : Memory = Memory(args, emb=self.agent.emb)
        self.state : State = None
        self.candidate_answers = None
        self.cur_step = 0

        self.max_step = args.max_step
        self.roll_back_T = args.roll_back_T
        self.answer_T = args.answer_T
        self.max_trips = args.max_trips
        self.max_get_trips = args.max_get_trips
        

    def _init_state(self, data, file_log):
        self.cur_step = 0
        self.idx = data['idx']
        if not data['topic_entities']:
            data['topic_entities'] = self.entity_linker.extract_entities(self.state.query)
        
        # process topic_entities, get id or name
        if not isinstance(data['topic_entities'][0], tuple):
            if self.args.kg == "freebase":
                data['topic_entities'] = [(x, api.get_name_by_id(x)) for x in data['topic_entities']]
            elif self.args.kg == 'wikidata':
                tmp = []
                for ent in data['topic_entities']:
                    ids = api.convert_name_to_id(ent)
                    if isinstance(ids, list):
                        for id in ids:
                            tmp.append((id[0], id[-1]))
                    else:
                        tmp.append((ids, ent))
                data['topic_entities'] = tmp

        self.state = State(data, self.args)
        self.expand_init_state()
        self.memory.short_mem = []
        self.memory.update_short_mem(self.cur_step, self.state, self.roll_back_T)
        self.candidate_answers = PriorityQueue()

    def expand_init_state(self):
        for ent in self.state.topic_entities:
            rels = api.get_all_relations(ent[0])
            if len(rels) == 1:          # If a topic ent only have one relation, it must be explored
                new_trips, new_ents = api.get_all_entities(ent, rels[0], expand_attr=False)
                new_trips, new_ents = self.format_trips(new_trips)
                self.state.update_kg(new_trips=new_trips, ents=ent[0], env=self)

    @timeout_decorator.timeout(60*20)
    def run(self, data, file_log=None):
        self.logger.info(f"\n======================== Now processing sample-{data['idx']} =========================\n")
        self.memory.init_memory()
        self._init_state(data, file_log)
        # load ablation set
        if "qa" in self.args.exp_set:
            self.max_step = -1

        if "no_short_mem" in self.args.ablation_set:
            from utils import expand_trip_sets
            ents, trips = expand_trip_sets(self.args.ab_res, self.idx, self)
            self.max_step = -1
            self.state.max_show_num = self.args.max_show_num
            self.state.set_to_ent_map = {"entities": ents}
            self.state.trip_set = trips

        if "gen_only" in self.args.ablation_set:
            ents, trips = rebuild_retrieved_trips(self.args.ab_res, self.idx, self, self.args.max_show_num)
            self.state.max_show_num = self.args.max_show_num
            if "2" in self.args.ablation_set:
                self.state.trigger_tokens = "['<triple>', ', ',', ','</triple>']"
            else:
                self.state.trigger_tokens = "['[/head]', '[/relation]', '[/tail]', '[/sep]']"
            self.state.set_to_ent_map = {"entities": ents}
            self.state.trip_set = trips
            self.max_step = -1

        step_by_step = True if 'sbs' in self.args.exp_set and 'plan' in self.args.exp_set else False
        if 'plan' in self.args.exp_set:
            sub_questions = self.planning(step_by_step=step_by_step)        # decompose original question into sub-questions
            global_query = self.state.query
        # # TODO:check
        if step_by_step and len(sub_questions) > 0:  # plan_step_by_step
            for idx, cur_question in enumerate(sub_questions):
                self.logger.info(f"\n======================== Process sub-question_{idx}/{len(sub_questions)} =========================\n")
                self.state.query = f"Global Query: {global_query}\n\n"
                if len(self.memory.sub_questions) > 0:
                    self.state.query += "Already processed sub-questions of Global Query:\n"
                    for q,a in self.memory.sub_questions:
                        self.state.query += f"\t{q}\n\t\tANSWER: {a}\n"
                    self.state.query += "\n"
                self.state.query += "Current sub-question of the Global Query to be addressed:\n"
                self.state.query += f"Original Query: {cur_question}\n\n"
                self.state.cur_query = cur_question

                self.max_step = self.args.max_step // 2                 # use half step to process a sub-question
                ans = self.solve_simple_query(global_query=False)
                self.memory.sub_questions.append([cur_question, ans])
                #self.memory.update_global_kg(self.state)
                self.cur_step = 0
                self.state.history = []
                self.state.actions = []
                self.memory.short_mem = []
                self.memory.update_short_mem(self.cur_step, self.state, self.roll_back_T)
                # self._init_state(data, file_log)

            # solve original question finally
            self.logger.info(f"\n======================== Process final question =========================\n")
            #self.state.replace_sub_KG(kg=self.memory.global_sub_kg, trip_set=self.memory.global_trip_set)
            self.state.query = f"Original Query: {global_query}\n\n"
            self.state.cur_query = global_query
            if len(self.memory.sub_questions) > 0:
                self.state.query += "Sub-questions of the original query and their answers:\n"
                for q,a in self.memory.sub_questions:
                    self.state.query += f"\t{q}\n\t\tANSWER: {a}\n"
            self.max_step = self.args.max_step
            ans = self.solve_simple_query(global_query=True)
        else:
            self.state.query = f"Original Query: {self.state.query}"
            ans = self.solve_simple_query(global_query=True)
            
        return ans

    def solve_simple_query(self, global_query=True):
        if "long_mem" in self.args.exp_set:
            self.recall_long_mem() # use long-term memory to build a good init state if possible
        cur_action = "DEFAULT"
        pre_output = None

        while cur_action != "FINISH" and self.cur_step <= self.max_step:
            self.logger.info(f"\n======================== Execute action for step {self.cur_step} =========================\n")
            next_action, output = self.execute(cur_action, pre_output)

            if 'forced_filter' in self.args.exp_set:
                if cur_action == 'DEFAULT' and len(self.state.actions) > 4 and "EXPAND_KG" not in self.state.actions[-2*2:]:        # avoid repeat same actions more than 3 times
                    next_action = 'EXPAND_KG'
                    output = "EXPAND_KG"
                if cur_action == 'DEFAULT' and len(self.state.actions) > 4 and "FILTER" not in self.state.actions[-2*2:]:
                    next_action = 'FILTER'
                    output = "FILTER"

            self.step(cur_action, next_action, output)
            if 'reflect' in self.args.exp_set and cur_action == "EXPAND_KG" and self.rollback():
                next_action = "DEFAULT"     # redo from pervious best state

            next_action = "DEFAULT" if next_action == "ERROR" else next_action
            cur_action = next_action
            pre_output = str(output)

        if self.cur_step >= self.max_step:
            self.logger.info(f'\n================= Hit max expansions {self.max_step}, stopping ==================\n')
            # Try answer based on current have information
            cur_action = "ANSWER"
            next_action, output = self.execute(cur_action)
            self.step(cur_action, next_action, output)

        answers_lst = [item for item in self.candidate_answers.queue]
        output = self.candidate_answers.get()[-1] if not self.candidate_answers.empty() else "No Answer"
        if global_query:
            self.candidate_answers = answers_lst

        if self.args.ablation_set == "" and 'base' in self.args.exp_set:
            self.memory.convert_short_to_long_mem(self.state, env=self)

        return output
    
    # update state and memory
    def step(self, cur_action, next_action, output):
        if next_action == "ERROR":
            return
        
        if cur_action == "DEFAULT":
            if output:
                next_action = next_action.split("| ")[0]
                # content = output.split("\n")[0].split(next_action)[-1].replace(":","").strip()
                self.state.update(cur_action, f"{next_action}:")
        
        elif cur_action == "FILTER":
            obs = output['obs']
            if obs not in self.state.history:
                self.state.update(cur_action, obs, output=output)
                self.state.update_kg(new_trips=output['new_trips'], ents=output['source_ents'], rels=output['source_rels'], env=self)
                #self.memory.update_short_mem(self.cur_step, self.state, self.memory.short_mem[-1]['score'] - 0.01)

            if next_action == "EXPAND_KG":
                self.state.update("EXPAND_KG", "EXPAND_KG:")
            
            #self.state.replace_sub_KG(new_trips, new_ents)
            

        elif cur_action == "PLAN":
            # content = "".join(re.findall(r"(?<=sub-questions:).*", str(output), re.S|re.I)).strip()
            # if not content:
            #     content = output
            self.state.update(cur_action, obs=output)
            # self.memory.update_short_mem(self.cur_step, self.state, self.roll_back_T)

        elif cur_action == "EXPAND_KG":
            obs=output['obs']
            if obs not in self.state.history:
                self.state.update(cur_action, obs=obs, output=output)
                self.state.update_kg(new_trips=output['new_trips'], ents=output['source_ents'], env=self, 
                                 use_attr_trips=True if 'expand_attrs' in self.args.exp_set else False)
                self.memory.update_short_mem(self.cur_step, self.state, output['score'])
    
        # elif cur_action == "EVALUATE_STATE":
        #     if isinstance(output, float):       # update score
        #         self.state.update(cur_action)
        #         self.memory.update_short_mem(self.cur_step, self.state, output)
        #     self.cur_step -= 1                  # must be executed follow by EXPAND_KG for now setting
        
        elif cur_action == "ANSWER":
            if isinstance(output, tuple):
                ans_score, ans = output
                raw_ans = ans
                ans = ans.split("ANSWER:")[-1].strip()
                ans = ' '.join(ans.split('\n'))
                if 'NO ANS' not in ans:
                    self.state.update(cur_action, f"  ANSWER: {ans}\n  ANSWER SCORE: {ans_score}")
                    self.candidate_answers.put((-ans_score, self.cur_step, raw_ans, ans))
                if next_action == "EXPAND_KG":
                    self.state.update("ANSWER->EXPAND_KG", "EXPAND_KG:")

        elif cur_action == "FINISH":
            ...

        self.memory.action_chain.append(cur_action)
        self.cur_step += 1


    def execute(self, cur_action, pre_output=None):
        available_actions = [#'THINK', 
                             "ANSWER", "FILTER", "EXPAND_KG"]
        if 'base' not in self.args.exp_set and 'short-mem' not in self.args.exp_set:
            available_actions.remove("FILTER")
        # The init step, select an action
        if cur_action == 'DEFAULT':
            inputs = {'options': available_actions}
            actions, output = self.agent.forward(inputs=inputs, state=self.state, action=cur_action)
            actions = actions.split("| ")[0]
        # Do planning, decompose the original query
        elif cur_action == 'PLAN':
            inputs = {}
            actions, output = self.agent.forward(inputs=inputs, state=self.state, action=cur_action)
        # Do expand kg
        elif cur_action == 'EXPAND_KG':
            relations = []
            selected_entities = []
            selected_relations = []
            new_trips = []
            new_ents = []
            source_ents = []
            source_rels = []
            obs = None

            reason = "".join(re.findall(r"(?<=THINK:).*(?=[\n.])", str(pre_output), re.S|re.I)).replace("\n"," ").strip() 
            if not reason:
                reason = "".join(re.findall(r".*?(?=SELECT ACTION)", str(pre_output), re.S|re.I)).replace("\n"," ").strip()
            rel_reason = reason
            # sub-step 1: expand entities, h->r
            ents, ent_sets = self.state.get_entities(only_entity=True)      # Allow select same entity
            if len(ents) > 0:
                inputs = {'think': reason, 'entities': [x[0] for x in ents], 'ent_sets': ent_sets} #sorted([x[0] if isinstance(x, tuple) else x for x in ents], key=len, reverse=True)}
                actions, output = self.agent.forward(inputs=inputs, state=self.state, action="EXPAND_ENT")
                #selected_entities = re.findall(sorted(, key=len, reverse=True)})
                # cal interface to get relations
                selected_entities = actions.split("| ")[:10]         # Max select ten entities
                expand_ent_set = set()
                select_ent_set_ids = set(re.findall(r"ent_set_.+?", " ".join(re.findall(r"ent_set_.+", str(selected_entities))).replace(",","")))
                for idx in select_ent_set_ids:
                        if idx not in self.state.set_to_ent_map:
                            continue
                        for ent in self.state.set_to_ent_map[f"{idx}"]:
                            expand_ent_set.add(ent[0])
                for ent in selected_entities:
                    if ent.startswith("Q") or ent.startswith("m."):
                        expand_ent_set.add(ent)
                expand_ent_set = list(expand_ent_set)
                if len(expand_ent_set) > 0:
                    relations = api.get_all_relations(expand_ent_set)
                else:
                    obs = f"  SELECT ENTITIES: {output if 'SELECT ENTITIES: ' not in output else output.split('SELECT ENTITIES: ')[-1]}\n  Observation: Error, invalid entities selected."
                    relations = []
            else:
                obs = f"  SELECT ENTITIES: NONE\n  Observation: Error: No available entities."
                relations = []
            # sub-step 2: expand relations, r->t
            relations = self.filter_relations(relations, ents=selected_entities)
            if len(relations) > 0:
                inputs = {'think': reason, 'selected_entities': selected_entities, 'relations': sorted(relations, key=len, reverse=True)}
                actions, output = self.agent.forward(inputs=inputs, state=self.state, action="EXPAND_REL")
                rel_reason = "".join(re.findall(r"(?<=THINK:).*?(?=[\n])", str(output), re.S|re.I)).replace("\n"," ").strip()
                if not rel_reason:
                    rel_reason = "".join(re.findall(r".*?(?=SELECT)", str(output), re.S|re.I)).replace("\n"," ").strip()
                constrain = None
                # if 'CONSTRAIN' in output:
                #     constrain = "".join(re.findall(r"\[.*\]", output)).strip()
                #     operator = "".join(re.findall(r"(?<=\[).*?(?=,)", constrain)).replace('"',"").replace("'","").strip()
                #     string = "".join(re.findall(r"(?<=,).*?(?=\])", constrain)).replace('"',"").replace("'","").strip()
                #     if operator and string:
                #         constrain = [operator, string]
                #     else:
                #         constrain = None
                selected_relations = [actions.split("| ")[-1]] if actions else []
                if len(selected_relations) > 0 and f"({selected_entities}, {source_rels})" not in self.state.used_edge:     # select available relation
                    # cal interface to get corresponding entity
                    new_trips, new_ents = api.get_all_entities(expand_ent_set, selected_relations, constrain, expand_attr=True)
                    # if self.args.kg == 'freebase':
                    #     new_trips = self.auto_expand_middle_ent(new_trips)

                    new_trips, new_ents = self.format_trips(new_trips)
                    if len(new_trips) > self.max_trips:
                        new_trips = self.filter_trips(new_trips, topk=self.max_trips)
                    source_ents = expand_ent_set
                    source_rels = selected_relations
                    # new_ent_set, new_trip_set = self.build_ent_set(new_trips, expand_ent_set)

                    #sub-step 2-2: filter got entities with conditions
                    # TODO: filter by tools
                    # if len(new_trips) > self.max_get_trips:
                    #     tmp_obs = f"  SELECT ENTITIES: {selected_entities}\n  SELECT RELATION: {selected_relations}\n  Observation: after EXPAND_KG {len(new_ents)} new entities, {len(new_trips)} new edges added.\nFILTER:"
                    #     tmp_state = self.get_tmp_state(ents=new_ents[:10], trips=new_trips[:10], cur_act="FILTER", obs=tmp_obs, 
                    #                                    selected_entities=selected_entities, selected_relations=selected_relations)

                    #     obs = {"obs":f"  After executing EXPAND_KG, {len(new_ents)} new entities are found.", "think": rel_reason}
                    #     filtered_trips, filtered_ents, obs = self.manage_short_memory(new_trips, new_ents, obs, tmp_state)
                    #     top_trips, top_ents = self.filter_trips(new_trips, new_ents, topk=20)       # if choose wrong functions
                    #     new_trips = set(top_trips) | set(filtered_trips)
                    #     new_ents = set(top_ents) | set(filtered_ents)
                    #     obs = tmp_obs + '\n' + obs

                    # new_trips, new_ents = self.filter_trips(new_trips, new_ents)        # only keep top-k related trips and ents
                else:
                    selected_relations = output.split("SELECT RELATION: ")[-1] if len(selected_relations) == 0 else selected_relations
                    obs = f"  SELECT ENTITIES: {selected_entities}\n  SELECT RELATION: {selected_relations}\n  Observation: Error, selected relation is invalid or has already been explored." if not obs else obs
            else:
                for ent in selected_entities:
                    self.state.unavailable_node.add(ent)
                obs = f"  SELECT ENTITIES: {selected_entities}\n  Observation: No available relations, or all relations for this entity have been explored, please try another entity." if not obs else obs  

            # sub-step 3: evaluate new state
            score = 0.0
            if not obs:
                obs = f"  SELECT ENTITIES: {selected_entities}\n  SELECT RELATION: {selected_relations}\n  Observation: after EXPAND_KG {len(new_ents)} new entities, {len(new_trips)} new edges added."
            # else:
            #     obs = f"{obs}\n  Observation: after FILTER {len(new_ents)} new entities, {len(new_trips)} new edges added."
            output = {'obs':obs,'new_trips':new_trips, 'source_ents':source_ents, 'source_rels': source_rels, 'used_func':f"({selected_entities}, {source_rels})"}
            if len(new_trips) > 0 and 'reflect' in self.args.exp_set:
                tmp_state = self.get_tmp_state(new_trips=new_trips, source_ents=source_ents, cur_act=cur_action, obs=obs)
                # only evaluate new added KG
                actions, score = self.agent.forward(inputs={'think': rel_reason, 'options': available_actions}, state=tmp_state, action="EVALUATE_STATE")
                score = float(re.search("\d\.\d+", score)[0]) if re.search("\d\.\d+", score) else 0.0

            output['score'] = score
            actions = "DEFAULT" # 'EVALUATE_STATE':

        # Do evaluation
        # elif "EVALUATE" in cur_action:
        #     inputs = {} if cur_action == 'EVALUATE_STATE' else {'answer': self.state.history[-1].split("ANSWER: ")[-1]}
        #     _, output = self.agent.forward(inputs=inputs, state=self.state, action=cur_action)
        #     output = float(re.search("\d\.\d+", output)[0]) if re.search("\d\.\d+", output) else 0.0
        #     if cur_action == 'EVALUATE_STATE':
        #         actions = "DEFAULT"
        #     elif cur_action == 'EVALUATE_ANS':
        #         actions = "FINISH" if output >= self.answer_T else "DEFAULT"

        elif cur_action == "FILTER":
            # sub-step 1: select a ent_set to process
            obs = ""
            actions = "DEFAULT"
            func_name = None
            func_input = None
            new_trips = []
            source_ents = []
            source_rels = []
            select_ents = "None"
            reason = "".join(re.findall(r"(?<=THINK:).*(?=[\n.])", str(pre_output), re.S|re.I)).replace("\n"," ").strip() 
            if not reason:
                reason = "".join(re.findall(r".*?(?=SELECT ACTION)", str(pre_output), re.S|re.I)).replace("\n"," ").strip()
            ents, ent_sets = self.state.get_entities(only_entity=True)
            if len(ent_sets) == 0:
                available_ents = [x[0] for x in ents]
            else:
                available_ents = [x for x in ent_sets]
            if len(available_ents) == 0:
                obs = f"  Fail: There are no entity sets to filter, please continue to explore KG."
                actions = "EXPAND_KG"
            else:
                all_match_ids, output = self.agent.forward({"obs": obs, "think": reason, "entities": available_ents}, state=self.state, action="FILTER")
                cands = set()
                for idx in self.state.set_to_ent_map.keys():
                    cands.add(idx)
                cands = sorted(list(cands), key=len, reverse=True)
                pattern = "(" + "|".join(cands).replace("(","\(").replace(")","\)") + ")"
                #select_ents = set(re.findall(r"ent_set_.+?", " ".join(re.findall(r"ent_set_.+", output)).replace(",","")))
                if cands:
                    select_ents = re.findall(pattern, "".join(re.findall(r"(?<=Select Entity Set\(s\): ).+", output)))
                else:
                    select_ents = set(all_match_ids.split("| "))
                func_name = "| ".join(re.findall(r"(?<=Select Function: ).+[FilterbyCondition|LogicOperation|FilterbyStr|FindRelation|Count|Verify]", output)).replace("'","").replace('"','').strip().split("| ")[0]
                func_input = "| ".join(re.findall(r"(?<=Function Input: ).+", output)).replace("'","").replace('"','').strip().split("| ")[0]
                if func_name and func_input and f"({select_ents}, {func_name}, {func_input})" not in self.state.used_edge:
                    func = (func_name, func_input)
                    input_ents = set()
                    filter_ents = set()
                    filter_trips = set()
                    for ent in select_ents:
                        if ent.startswith("Q") or ent.startswith("m."):
                            input_ents.add((ent, api.get_name_by_id(ent)))
                        elif ent in self.state.set_to_ent_map:
                            input_ents.update(self.state.set_to_ent_map[ent])
                    try:
                        filter_ents, filter_trips = api.filter_by_function(list(input_ents), func, select_ents, self)
                    except Exception as e:
                        print_error(e)
                        self.logger.info(f"\n-------------------------- FILTER BUG --------------------\n{e}")
                        obs = obs + f"  SELECT ENT_SETS: {list(select_ents)}\n  SELECT FUNCTION: {func_name}\n  FUNCTION INPUT: {func_input}\n    Observation: Fail, unavailable function input."
                        input_ents = []
                    if len(filter_ents) == len(input_ents) and len(input_ents)>0:
                        obs = obs + f"  SELECT ENT_SETS: {list(select_ents)}\n  SELECT FUNCTION: {func_name}\n  FUNCTION INPUT: {func_input}\n    Observation: all entities satisfy condition."
                    elif len(filter_ents) > 0:
                        obs = obs + f"  SELECT ENT_SETS: {list(select_ents)}\n  SELECT FUNCTION: {func_name}\n  FUNCTION INPUT: {func_input}\n    Observation: filter out {len(filter_ents)} entities."
                    elif len(input_ents):
                        obs = obs + f"  SELECT ENT_SETS: {list(select_ents)}\n  SELECT FUNCTION: {func_name}\n  FUNCTION INPUT: {func_input}\n    Observation: no entities satisfy condition or invalid function input."
                    if len(filter_trips) > 0:
                        new_trips, new_ents = self.format_trips(filter_trips)
                        if len(new_trips) > self.max_trips:
                            new_trips = self.filter_trips(new_trips, topk=self.max_trips)
                        #source_rels = [] #func_input if func_name in ['FilterbyCondition'] else None
                        source_ents = [x[0] for x in input_ents] if ("Filter" in func_name or "Find" in func_name) else filter_ents
                        # new_ent_set, new_trip_set = self.state.update(new_trips, ents=[x[0] for x in input_ents] if "Filter" in fn else filter_ents, rels=rel)
                elif func_name and func_input and f"({select_ents}, {func_name}, {func_input})" in self.state.used_edge:
                    obs = f"  SELECT ENT_SETS: {list(select_ents)}\n  SELECT FUNCTION: {func_name}\n  FUNCTION INPUT: {func_input}\n    Observation: Fail, function and input have already been performed, do not repeat them again!"
                else:
                    obs = f"  Fail: unavailable function selected or unavailable function inputs used."

            output = {'new_trips':new_trips, 'source_ents':source_ents, 'source_rels':source_rels, 'obs':obs, 'used_func':f"({select_ents}, {func_name}, {func_input})"}
                

        # Do answer
        elif cur_action == 'ANSWER':
            selected_to_extract = 'NONE'
            inputs = {}
            if 'base' in self.args.exp_set or 'short-mem' in self.args.exp_set:
                # sub-step 1: extract valid information
                reason = "".join(re.findall(r"(?<=THINK:).*(?=[\n.])", str(pre_output), re.S|re.I)).replace("\n"," ").strip()
                inputs['think'] = reason
                _, output = self.agent.forward(inputs=inputs, state=self.state, action="EXTRACT_ANS")
                if "SELECT: " in output:
                    selected_to_extract = "| ".join(re.findall(r"(?<=SELECT: ).+", output, re.S)).strip().split("| ")[-1]

            if "NO ANS" in selected_to_extract and self.cur_step <= self.max_step:
                actions = "DEFAULT"
                output = (0, "NO ANS")
            else:
                # sub-step 2: generate answer
                if selected_to_extract in self.state.set_to_ent_map:
                    extract_info = list(self.state.set_to_ent_map[selected_to_extract])
                    _, ids = cal_sim_score(self.state.query, [x[-1] for x in extract_info], self.agent.emb, topk=20, return_idx=True)
                    extract_info = [extract_info[idx] for idx in ids]
                    inputs['ent_set'] = (selected_to_extract, extract_info)
                elif "(" in selected_to_extract and ")" in selected_to_extract:
                    inputs['trip_set'] = selected_to_extract

                inputs['hint'] = selected_to_extract
                _, ans = self.agent.forward(inputs=inputs, state=self.state, action=cur_action)

                # sub-step 3: evaluate answer
                if 'gen_only' in self.args.ablation_set:
                    ans_score = 0
                else:
                    inputs = {'answer': ans.split("ANSWER: ")[-1]}
                    _, output = self.agent.forward(inputs=inputs, state=self.state, action="EVALUATE_ANS")
                    ans_score = float(re.search("\d\.\d+", output)[0]) if re.search("\d\.\d+", output) else 0.0

                if ans_score >= self.answer_T:
                    actions = "FINISH"
                else:
                    actions = "EXPAND_KG"       # get a low score answer, force to continue expand KG
                output = (ans_score, ans)

        # Not match any actions
        else:
            actions = "DEFAULT"
            output = "Get unavailable actions"
        return actions, output

    def auto_expand_middle_ent(self, trips):
        tmp_trips = set()
        middle_ent_group = {}
        for trip in trips:
            hid, r, tid = trip
            tmp_trips.add((hid, r, tid))
            if hid.startswith("m.") and api.convert_id_to_name(hid).startswith("m."):
                if r in middle_ent_group:
                    middle_ent_group[r].append(hid)  
                else:
                    middle_ent_group[r] = [hid]
            if tid.startswith("m.") and api.convert_id_to_name(tid).startswith("m."):
                if r in middle_ent_group:
                    middle_ent_group[r].append(tid)  
                else:
                    middle_ent_group[r] = [tid]
            
        for k, v in middle_ent_group.items():
            res = api.process_middle_node(v)
            for t in res:
                tmp_trips.add((t[0], t[1], t[2]))

        return list(tmp_trips)

    def manage_sub_KG(self, max_trips=50, state=None):
        # only keep top-50 edges
        trips = state.get_edges() if state else self.state.get_edges()
        self.logger.info(f'\n================= Current number of triples {len(trips)} exceeds the maximum limit {max_trips}, pruning... ==================\n')
        trips = trips[:self.max_get_trips]
        tmp_trips = [f"{x[0][-1], x[1], x[2][-1]}" for x in trips]
        _, index = cal_sim_score(self.state.query, tmp_trips, model=self.agent.emb, topk=self.max_get_trips, return_idx=True)
        trips = [trips[idx] for idx in index]
        new_kg = nx.DiGraph()
        for trip in trips:
            h,r,t = trip
            new_kg.add_node(h)
            new_kg.add_node(t)
            new_kg.add_edge(h, t, rel=r)
        
        # update trip_set
        # all_ents = set(new_kg.nodes())
        # for idx, dic in enumerate(self.trip_set):
        #     ent_set = dic['ents']
        #     tmp = set()
        #     for e in ent_set:
        #         if e in all_ents:
        #             tmp.add(e)
        #     self.trip_set[idx]['ents'] = tmp

        return new_kg
    
    def manage_short_memory(self, new_trips, new_ents, obs=None, tmp_state=None):
        all_filtered_ents = set()
        all_find_trips = set()
        used_func = set()
        cnt = 3     # try filter 3 times
        think = obs['think'] if obs and 'think' in obs else ""
        obs = obs['obs'] if obs and "obs" in obs else ""
        state = self.state if not tmp_state else tmp_state
        while cnt:
            cnt -= 1
            actions, output = self.agent.forward({"obs": obs, "think": think}, state=state, action="FILTER")
            func_name = "| ".join(re.findall(r"(?<=Select Function: ).+[FilterbyType|FilterbyAttribute|FilterbyRelation|LogicOperation|FilterbyStr|GetRelation|Count|Verify]", output)).replace("'","").replace('"','').strip().split("| ")
            func_input = "| ".join(re.findall(r"(?<=Inputs: ).+", output)).replace("'","").replace('"','').strip().split("| ")
            if len(func_name)==len(func_input) and len(func_name)>0:
                for fn, fi in zip(func_name, func_input):
                    if (fn, fi) not in used_func:
                        used_func.add((fn, fi))
                    else:
                        obs = obs + f"  SELECT FUNCTION: {fn}\n  FUNCTION INPUTS: {fi}\n    Observation: Fail, unavailable input.\n"
                        continue
                    func = [fn, fi]
                    filter_ents, find_trips = api.filter_by_function(new_ents, func, self)
                    if len(filter_ents) < self.max_get_trips:
                        all_filtered_ents.update(filter_ents)
                        all_find_trips.update(find_trips)
                    if len(filter_ents) == len(new_ents):
                        obs = obs + f"  SELECT FUNCTION: {fn}\n  FUNCTION INPUTS: {fi}\n    Observation: all entities satisfy condition.\n"
                    elif len(filter_ents) > 0:
                        obs = obs + f"  SELECT FUNCTION: {fn}\n  FUNCTION INPUTS: {fi}\n    Observation: filter out {len(filter_ents)} entities.\n"
                    else:
                        obs = obs + f"  SELECT FUNCTION: {fn}\n  FUNCTION INPUTS: {fi}\n    Observation: no entity satisfies condition.\n"

        new_ents = all_filtered_ents if len(all_filtered_ents) != 0 else new_ents
        all_find_trips, all_find_ents = self.format_trips(all_find_trips)
        new_trips = list(new_trips) + all_find_trips
        new_ents = list(new_ents) + all_find_ents
        new_trips = self.filter_trips_by_ents(new_trips, new_ents)
        
        return new_trips, new_ents, obs

    def planning(self, step_by_step=False):
        sub_questions = []
        cur_action = "PLAN"
        _, output = self.execute(cur_action)
        output = output.split("Sub-Questions:")[-1] if 'Sub-Questions:' in output else output
        sub_questions = re.findall(r"[1-9]\. .+?[\?|\n]|.*\.", str(output), re.I)[:5]      # max allow 5 sub-questions

        if len(sub_questions) > 0:
            obs = ""
            for sq in sub_questions:
                obs += f"\t{sq}\n"
            obs = f"Planning: The original query may be answered step-by-step through the following sub-questions:\n{obs}\n"
            if not step_by_step:
                self.step(cur_action, None, obs)

        return sub_questions

    def rollback(self):
        mem = self.memory.short_mem
        if len(mem) <= 3:       # min rollback step
            return 
        cur_score = mem[-1]['score']
        if cur_score <= self.roll_back_T:
            self.logger.info(f'\n================= Current state score {cur_score} is lower than threshold {self.roll_back_T}, rollback to pervious best state ==================\n')
            if self.args.roll_back == "max_score":
                # find pervious best state 
                best_state = [mem[0]['score'], mem[0]['state'], mem[0]['step']]
                for per_state in mem:
                    if per_state['score'] >= best_state[0]:
                        best_state = [per_state['score'], per_state['state'], per_state['step']]
                new_state = best_state[1]
                self.memory.action_chain.append(f"ROLLBACK:{self.cur_step}->{best_state[-1]}")
                self.state = deepcopy(new_state)
                return True

    def recall_long_mem(self):
        if not self.memory.long_mem['vector_store']:
            # not have long mem
            return
        
        def get_ent_typ(ents, q=None):
            ent_typs = []
            for ent in ents:
                if ent[0].startswith("m.") or ent[0].startswith("Q"):
                    ent_typ = api.get_types(ent[0])
                    if len(ent_typ) > 3:
                        top_typs = cal_sim_score(q, ent_typ, model=self.agent.emb, topk=3)
                    else:
                        top_typs = ent_typ

                    ent_typs.append([ent, set(top_typs)])

            return ent_typs
        
        query = self.state.cur_query
        topic_ents = self.state.topic_entities
        cur_ent_typs = get_ent_typ(topic_ents, q=self.state.cur_query)
        tmp_state = deepcopy(self.state)

        schemas = self.memory.recall_from_long_mem(query, k=3)       # recall most similar k samples
        max_recall_step = 3 #int(self.max_step * 0.3)
        query_schemas = {}          # rel: ent_typs
        aligned_query_schemas = {}  # rel: ents
        # combine recalled schemas
        for schema in schemas:
            for step, qg in enumerate(schema['schemas']):
                if step >= max_recall_step:
                    continue
                rel = qg['rel']
                if rel == 'Filterable by':
                    continue
                typs = qg['head_ent_typs'] | qg['tail_ent_typs']
                top_typs = cal_sim_score(qg['query'], list(typs), self.agent.emb, topk=3)
                if rel not in query_schemas:
                    query_schemas[rel] = set(top_typs)
                else:
                    query_schemas[rel].update(typs)

        for step in range(max_recall_step):
            for rel, typs in query_schemas.items():
                for cur_typ in cur_ent_typs:
                    cur_ent, cur_ent_typ = cur_typ
                    if cur_ent_typ & typs:
                        if rel not in aligned_query_schemas:
                            aligned_query_schemas[rel] = set([cur_ent])
                        else:
                            aligned_query_schemas[rel].add(cur_ent)

            drop_keys = []
            for rel, ents in aligned_query_schemas.items():
                selected_entities = [x[0] for x in ents]
                selected_relations = rel
                try:
                    new_trips, new_ents = api.get_all_entities(selected_entities, selected_relations, expand_attr=True)
                except:
                    continue
                if len(new_trips) == 0:
                    continue
                new_trips, new_ents = self.format_trips(new_trips)
                if len(new_trips) > self.max_trips:
                    new_trips, new_ents = self.filter_trips(new_trips, topk=self.max_trips, return_ents=True)
                obs = f"EXPAND_KG:\n  SELECT ENTITIES: {selected_entities[:5]}\n  SELECT RELATION: {selected_relations}\n  Observation: after EXPAND_KG {len(new_ents)} new entities, {len(new_trips)} new edges added."
                tmp_state.update(f"recall_long_mem_{step}", obs)
                tmp_state.update_kg(new_trips, selected_entities, use_attr_trips=False, env=self)
                new_ent_typs = get_ent_typ(new_ents, q=self.state.cur_query)
                for x in new_ent_typs:
                    if x not in cur_ent_typs:
                        cur_ent_typs.append(x)
                drop_keys.append(rel)

            for k in drop_keys:                 # drop used query graph
                aligned_query_schemas.pop(k)
                query_schemas.pop(k)

        # evaluate pre-builded sub-kg
        action, score = self.agent.forward(inputs={'think': "", 'options':  ["EXPAND_KG", "ANSWER"]}, state=tmp_state, action="EVALUATE_STATE")
        
        score = float(re.search("\d\.\d+", score)[0]) if re.search("\d\.\d+", score) else 0.0
        score = 0.0 if len(list(tmp_state.sub_kg.adj)) == 0 else score

        # if score >= 0.0:   
        self.logger.info(f'\n================= build init state based on recalled long_mem with {score} score ==================\n')
        tmp_state.history = []
        self.state = tmp_state
        self.memory.action_chain.append(f"recall_long_mem: {score}")
        self.memory.update_short_mem(self.cur_step, self.state, score)

            # try answer based on long-mem only
            # cur_action = "ANSWER"
            # self.state.update("ANSWER", f"ANSWER:")
            # next_action, output = self.execute(cur_action)
            # output = (abs(output[0] - 0.01), output[1])   # lower long-mem based ans score
            # self.step(cur_action, next_action, output)

    def get_tmp_state(self, new_trips, source_ents, cur_act, obs=None):
        tmp_state = deepcopy(self.state)
        # tmp_state.query = self.state.query
        # tmp_state.topic_entities = self.state.topic_entities

        tmp_state.update(cur_act, obs)
        tmp_state.update_kg(new_trips=new_trips, ents=source_ents, state=self.state, env=self)

        return tmp_state

    def filter_relations(self, relations, ents=[]):
        tmp_rels = relations.copy()
        for ent in ents:
            for rel in tmp_rels:
                if (ent, rel) in self.state.used_edge and rel in relations:
                        #if len(tmp_rels) <= 1 or (str((ent, rel)) in self.state.used_edge_score and self.state.used_edge_score[str((ent, rel))] > self.roll_back_T):        # already save in memory
                    relations.remove(rel)
        #TODO: process too much relations situation
        # similarity filter
        if len(relations) > 0:
            relations = cal_sim_score(self.state.query, relations, model=self.agent.emb, topk=self.max_get_trips)

        # random filter
        # relations = random.sample(relations, min(len(relations), 50))
        # relations = sorted(relations, key=len, reverse=True)
        return relations
    
    def filter_trips(self, trips, topk=None, return_ents=False):
        trips = list(trips)
        tmp_trips = [f"{x[0][-1], x[1], x[2][-1]}" for x in trips]
        _, index = cal_sim_score(self.state.query, tmp_trips, model=self.agent.emb, topk=self.max_trips if not topk else topk, return_idx=True)
        trips = [trips[idx] for idx in index]

        if return_ents:
            new_ents = set()
            for trip in trips:
                h,r,t = trip
                new_ents.add(h)
                new_ents.add(t)
            
            return trips, new_ents
        else:
            return trips
    

    def filter_trips_by_ents(self, trips, ents):
        tmp_trips = []
        for trip in trips:
            h,r,t = trip
            if h in ents or t in ents:
                tmp_trips.append(trip)
        return tmp_trips
    
    def format_trips(self, edges):
        node_set = set()
        edge_set = set()
        id2name = {}
        for edge in edges:
            hid, r, tid = edge
            if not isinstance(hid, tuple):
                if hid not in id2name:
                    hw = api.get_name_by_id(hid, self)
                    id2name[hid] = hw
                else:
                    hw = id2name[hid]
                h = (hid, hw)
            else:
                h = hid
            if not isinstance(tid, tuple):
                if tid not in id2name:
                    tw = api.get_name_by_id(tid, self)
                    id2name[tid] = tw
                else:
                    tw = id2name[tid]
                t = (tid, tw)
            else:
                t = tid
            edge_set.add((h, r, t))
            node_set.add(h)
            node_set.add(t)
        return list(edge_set), list(node_set)
    
    def _retry(self, cur_action):
        print("Fail to call LLM, retry...")
        next_action = "ERROR"
        cnt = 0
        max_try = 20
        while next_action == "ERROR" and cnt <= max_try:
            time.sleep(10)
            next_action, output = self.execute(cur_action)
            cnt += 1
        if cnt >= max_try:
            raise ValueError(f"Fail to call LLM after retry {max_try} times, Stop run")
        
        return next_action, output

    
if __name__ == "__main__":
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # Basic Setting
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--base_path', default="/raid_sdb/home/tsy/KG-Agent", type=str)
    parser.add_argument('--model_name', default='gpt-3.5-turbo', type=str)
    parser.add_argument('--data_name', default='wqsp_v4', type=str)
    parser.add_argument('--entity_linker_path', default="/raid_sdd/tsy/entity_linker", type=str)
    args = parser.parse_args()

    args.max_step = 10

    df = pd.read_csv(f"{args.base_path}/data/{args.data_name}.csv")
    with open(f"{args.base_path}/data/{args.data_name.split('_')[0]}_mid2name.pkl", 'rb') as f:
        id2name = pickle.load(f)
    
    df = df.sample(100)
    results = []
    env = Env(args)
    for i in trange(len(df)):
        data = {'query': df.iloc[i]['question'],
            'topic_entities': eval(df.iloc[i]['entities']),
            'id2name': id2name,
        }
        answer = env.run(data)
        results.append(answer)
