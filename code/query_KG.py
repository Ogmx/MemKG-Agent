import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import pickle
import os
import random
import networkx as nx
import torch
from multiprocessing import Pool, cpu_count
from SPARQLWrapper import SPARQLWrapper, JSON  
from urllib.parse import urlparse
sparql = SPARQLWrapper('http://localhost:3001/sparql')

def get_name(mid):
    sparql_query = ('PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>'
                    'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>'
                    'PREFIX : <http://rdf.freebase.com/ns/>'
                    'SELECT DISTINCT ?name where {'
                    f'VALUES ?x0 {{:{mid}}} '
                    '?x0 :type.object.name ?name '
                    'FILTER(LANGMATCHES(LANG(?name), "en")).'
                    '}')

    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    ret = sparql.queryAndConvert()
    results = []
    for res in ret["results"]["bindings"]:
        ans = [urlparse(res[k]['value']).path for k in res]
        results.append(ans)
    return results

def get_entity(mid):
    sparql_query = ("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
                    "PREFIX : <http://rdf.freebase.com/ns/> "
                    "SELECT DISTINCT ?s ?r ?o WHERE { "
                    f"VALUES ?s {{:{mid}}} "
                    "?s ?r ?o . "
                    # "FILTER(LANGMATCHES(LANG(?o), 'en'))."
                    "}")

    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    ret = sparql.queryAndConvert()
#     results = []
#     for res in ret["results"]["bindings"]:
#         ans = [urlparse(res[k]['value']).path for k in res]
#         results.append(ans)
    return ret

# https://github.com/RichardHGL/WSDM2021_NSM/tree/main/preprocessing/Freebase

def filter_result(ret):
    filter_domain = ['music.release', 'authority.musicbrainz', '22-rdf-syntax-ns#type', 'book.isbn',
                 'common.licensed_object', 'tv.tv_series_episode', 'type.namespace', 'type.content',
                 'type.permission', 'type.object.key', 'type.object.permission', 'type.type.instance',
                 'topic_equivalent_webpage', 'dataworld.freeq']

    res_lst = []
    for x in ret['results']['bindings']:
        if x['o']['type'] == 'literal':
            if "xml:lang" not in x['o'].keys() or x['o']['xml:lang'] != "en" :
                continue
            else:
                res_lst.append(x)
                
        if x['o']['value'].split("/")[-1] in filter_domain or x['r']['value'].split("/")[-1] in filter_domain:
            continue
            
        res_lst.append(x)
    return res_lst

def query_hop_1(mid):
    ret = get_entity(mid)
    res = filter_result(ret)
    trip_map = {}
    for trip in res:
        h,r,t = trip['s']['value'], trip['r']['value'], trip['o']['value']
        h = h.split("/")[-1]
        r = r.split("/")[-1]
        t = t.split("/")[-1]
    
        trip_map[r] = t

    hop1 = {str(mid): trip_map}
    
    return hop1

def run(args):
    idx, mid = args
    try:
        res = query_hop_1(mid)
    except:
        res = {mid: "ERROR"}
    return res


if __name__ == "__main__":

    df = pd.read_csv("/data/tsy/datasets/processed_data/FB_data.csv")

    # query hop1
    hop1_ents = set()
    for i in trange(len(df)):
        a_ents = [x['kb_id'] for x in eval(df.iloc[i]['answer'])]
        q_ents = [x['kb_id'] for x in eval(df.iloc[i]['q_ent'])]

        hop1_ents.update(a_ents)
        hop1_ents.update(q_ents)

    print("hop1 query ent num:",len(hop1_ents))

    # query hop2
    hop2_ents = set()
    f_read = open('FB_1hop_adj.pkl', 'rb')
    hop1_adj = pickle.load(f_read)
    f_read.close()

    for k,v in hop1_adj.items():
        if v == "ERROR":
            continue
        for typ,value in v.items():
            hop2_ents.add(value)
    
    print(f"hop2 query ent num:", len(hop2_ents), len(hop1_ents), len(hop2_ents - hop1_ents))
    hop2_ents = hop2_ents - hop1_ents

    # query hop3
    hop3_ents = set()
    f_read = open('FB_2hop_adj.pkl', 'rb')
    hop2_adj = pickle.load(f_read)
    f_read.close()

    for k,v in hop2_adj.items():
        for typ,value in v.items():
            hop3_ents.add(value)
    
    print(f"hop2 query ent num:", len(hop3_ents), len(hop2_ents), len(hop3_ents - hop2_ents))
    hop3_ents = hop3_ents - hop2_ents

    # adj = {}
    # for ent in tqdm(all_ents):
    #     hop1_adj = query_hop_1(ent)
    #     adj.update(hop1_adj)

    # f_save = open('FB_1hop_adj.pkl', 'wb')
    # pickle.dump(adj, f_save)
    # f_save.close()

    #  multiproces query
    all_ents = hop3_ents
    adj_dict = {}
    with Pool(processes=30) as p:
        with tqdm(total=len(all_ents), desc='query KG ing...') as pbar:
            for result in p.imap(run, enumerate(all_ents)):
                adj_dict.update(result)
                pbar.update()


    cnt = 0
    tmp_adj = {}
    for k, v in adj_dict.items():
        if v == "ERROR":
            cnt += 1
            continue
        tmp_adj.update({k:adj_dict[k]})
    print(f"{cnt} query not get correct response, {cnt / len(adj_dict)}. will auto remove them")

    f_save = open('FB_3hop_adj.pkl', 'wb')
    pickle.dump(tmp_adj, f_save)
    f_save.close()