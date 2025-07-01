import os
import pickle
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm, trange
from SPARQLWrapper import SPARQLWrapper, JSON  
from urllib.parse import urlparse
from query_KG import get_name, get_entity
sparql = SPARQLWrapper('http://localhost:3001/sparql')


def load_data(name, split, path):
    eid2mid = {}
    rid2mid = {}
    wid2word = {}
    df = pd.read_json(f"{path}/{data_name}/{split}_simple.json", lines=True)
    
    def map_id(dic, lst):
        cnt = 0
        for ids in lst:
            dic[cnt] = ids
            cnt += 1
        
    with open(f"{path}/{data_name}/entities.txt", "r") as f:
        ents = f.read().split("\n")
        map_id(eid2mid, ents)
    
    with open(f"{path}/{data_name}/relations.txt", "r") as f:
        rels = f.read().split("\n")
        map_id(rid2mid, rels)

    with open(f"{path}/{data_name}/vocab_new.txt", "r") as f:
        words = f.read().split("\n")
        map_id(wid2word, words)
        
        
    # replace oid to mid; and find all mid from all triples
    all_mid = set()
    for i in trange(len(df)):
        subkg = df.iloc[i]['subgraph']
        trips = subkg['tuples']
        new_trips = []
        for trip in trips:
            hid, rid, tid = trip[0], trip[1], trip[2]
            new_trip = [eid2mid[hid], rid2mid[rid], eid2mid[tid]]
            new_trips.append(new_trip)
        df.loc[i, 'subgraph'] = new_trips
        
        q_ents = []
        for q_ent in df.iloc[i]['entities']:
            q_ents.append(eid2mid[q_ent])
        df.loc[i, 'entities'] = q_ents
        
        all_ents = subkg['entities']
        for oid in all_ents:
            all_mid.add(eid2mid[oid])
            
    return df, all_mid
    

# step 2: find detail information of all mid

import requests
from bs4 import BeautifulSoup

def get_name_by_mid(mid):
    # method 1: from website
    url = f"https://freebase.toolforge.org/{mid.split('.')[0]}/{mid.split('.')[-1]}"
    
    response = requests.get(url)
    html = response.text
    
    # get name and attributes
    soup = BeautifulSoup(html, 'html.parser')
    tmp = soup.find("script")
    dic = eval(tmp.text)
    res = {}
    for key in dic.keys():
        if "@" not in key:
            for sd in dic[key]:
                if sd['@language'] == "en":
                    res[key] = sd['@value']
        else:
            res[key] = dic[key]
            
    # get other links and Qid
    res['links'] = []
    herfs = soup.find_all('a')
    for herf in herfs:
        if herf.get('class') and 'card-link' in herf.get('class'):
            res['links'].append(herf.get("href"))
            if herf.text == "Wikidata":
                res['qid'] = herf.get("href").split("/")[-1]
    return res


def find_name(res, mid):
    name_lst = ["/ns/type.object.name",
            "/ns/common.topic.alias",
            "/key/key/wikipedia.en_title",
            "/key/wikipedia.en",
            "/key/en",
            "/ns/common.notable_for.display_name",
            "/2000/01/rdf-schema",
           ]
    mid2text = {}
    mid2text[mid] = {}
    for tup in res['results']['bindings']:
        r, o = tup['r']['value'], tup['o']
        r = r.replace("http://rdf.freebase.com", "")
        if r in name_lst:
            if 'xml:lang' in o.keys() and o['xml:lang'] == 'en':
                mid2text[mid][r] = o['value']

        elif r == "/1999/02/22-rdf-syntax-ns":
            mid2text[mid][r] = o['value']
            
    return mid2text

def mid2name(args):
    idx, mid = args
    try:
        res = find_name(get_entity(mid), mid)
        
    except:
        res = {mid: "ERROR"}
    return res

def multiprocess_query(inputs, func):
    outputs = {}
    with Pool(processes=30) as p:
        with tqdm(total=len(inputs), desc='processing...') as pbar:
            for result in p.imap(func, enumerate(inputs)):
                outputs.update(result)
                pbar.update()
    return outputs



if __name__ == "__main__":

    data_name_lst = ["webqsp", "cwq"]
    split = "test"
    path = "/data/tsy/datasets/wqsp&cwq_from_NSM"

    for data_name in data_name_lst:
        print(f"now procssing {data_name}")
        
        df, all_mid = load_data(data_name, split, path)
    
        ents_info = multiprocess_query(inputs=all_mid, func=mid2name)
        
        with open(f'/data/tsy/datasets/{data_name}_mid2name.pkl', 'wb') as f:
            pickle.dump(ents_info, f)