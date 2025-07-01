# from https://github.com/YaooXu/GoG

from typing import List, Tuple, Union
from SPARQLWrapper import SPARQLWrapper, SPARQLExceptions, JSON
from retry import retry
import re
import json, random
import urllib
import datetime, time
from pathlib import Path
from numpy import tri
from tqdm import tqdm
import timeout_decorator
from utils import cal_sim_score
from utils import print_error
from interface.utils import add_ns, remove_ns

sparql = SPARQLWrapper('http://localhost:3001/sparql')
sparql.setReturnFormat(JSON)
ns_prefix = "http://rdf.freebase.com/ns/"


def execute_query(query):

    #@retry(SPARQLExceptions.EndPointInternalError, tries=2, backoff=5, max_delay=60)

    @timeout_decorator.timeout(60*3)
    def execute(query):
        sparql.setQuery(query)
        results = sparql.query().convert()
        results = results["results"]["bindings"]
        return results
    
    try:
        results = execute(query)
    except Exception as e:
        print("------------------- KG API ERROR ----------------\n")
        print_error(e)
        results = []

    return results
        

def replace_relation_prefix(relations):
    return [
        relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for relation in relations
    ]


def replace_entities_prefix(entities):
    return [
        entity["entity"]["value"].replace("http://rdf.freebase.com/ns/", "") for entity in entities
    ]


def get_tail_entity(entity_id, relation):
    sparql_pattern = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?entity
    WHERE {{
        ns:{head} ns:{relation} ?entity .
    }}
    """
    sparql_text = sparql_pattern.format(head=entity_id, relation=relation)
    entities = execute_query(sparql_text)

    entities = replace_entities_prefix(entities)
    entity_ids = [entity for entity in entities if not entity.startswith("g.")] #if entity.startswith("m.")]

    return entity_ids


def get_head_entity(entity_id, relation):
    sparql_pattern = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?entity
    WHERE {{
        ?entity ns:{relation} ns:{tail}  .
    }}
    """
    sparql_text = sparql_pattern.format(tail=entity_id, relation=relation)
    entities = execute_query(sparql_text)

    entities = replace_entities_prefix(entities)
    entity_ids = [entity for entity in entities if not entity.startswith("g.")] #if entity.startswith("m.")]

    return entity_ids

def get_out_relations(entity_id):
    sparql_pattern = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation
    WHERE {{
        ns:{head} ?relation ?x .
    }}
    """
    sparql_text = sparql_pattern.format(head=entity_id)
    relations = execute_query(sparql_text)

    relations = replace_relation_prefix(relations)

    return relations


def get_in_relations(entity_id):
    sparql_pattern = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation
    WHERE {{
        ?x ?relation ns:{tail} .
    }}"""
    sparql_text = sparql_pattern.format(tail=entity_id)
    relations = execute_query(sparql_text)

    relations = replace_relation_prefix(relations)

    return relations


def get_all_entities(entities:List[str], relations:List[str], constrain=None, expand_attr=True) -> Union[List, List]:
    new_trips = set()
    new_ents = set()
    if isinstance(entities, str):
        entities = [entities]
    if isinstance(relations, str):
        relations = [relations]
    def parser_attribute(source_ent, rel, attr, di):    # forward: (source_ent, rel, [ah, ar, at])  backward: ([ah, ar, at], rel, source_ent)
        all_attrs = get_1hop_triples(attr)
        for trip in all_attrs:
            ah, ar, at = trip
            if ah != attr:
                continue
            if di == 'forward':
                new_trips.add((source_ent, rel, f"[{ah}| {ar}| {at}]"))
            else:
                new_trips.add((f"[{ah}| {ar}| {at}]", rel, source_ent))

    all_attrs = set()
    for ent in entities:
        for rel in relations:
            if not ent.startswith("m."):
                continue
            head_ents = get_head_entity(ent, rel)
            tail_ents = get_tail_entity(ent, rel)
            for he in head_ents:
                if he.startswith("m.") and convert_id_to_name(he) == he:
                    all_attrs.add((ent, rel, he, 'backward'))
                new_trips.add((he, rel, ent))
                new_ents.add(he)
            for te in tail_ents:
                if te.startswith("m.") and convert_id_to_name(te) == te:
                    all_attrs.add((ent, rel, te, 'forward'))
                new_trips.add((ent, rel, te))
                new_ents.add(te)

    for attr in all_attrs:
        source_ent, rel, ent, di = attr
        parser_attribute(source_ent, rel, ent, di)
        
    return list(new_trips), list(new_ents)


def get_all_relations(entity_ids):
    # TODO: check
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]
    tmp_ents = set()
    for ent in entity_ids:
        if ent.startswith("m.") and len(ent)>2:
            tmp_ents.add(ent)
    entity_ids = tmp_ents
    rels = set()
    for entity_id in entity_ids:
        in_rels = get_in_relations(entity_id)
        out_rels = get_out_relations(entity_id)
        rels.update(in_rels)
        rels.update(out_rels)
    rels = pre_filter_relations(list(rels))
    return rels


def filter_entities_by_constrain(ents, constrain, all_trips=None):
    new_ents = set()
    new_trips = set()
    if constrain.startswith("m."):
        constrain = convert_id_to_name(constrain)
    constrain = constrain.lower()
    ent_set = set()
    if isinstance(ents, str):
        ents = re.findall(r"m.+", ents)
    elif len(ents) and isinstance(ents[0], str):
        ents = [convert_name_to_id(ent) for ent in ents]
    elif len(ents) and isinstance(ents[0], tuple):
        ents = [x[0] for x in ents]

    for ent in ents:
        if isinstance(ent, list):
            for e in ent:
                if e.startswith('m.'):
                    ent_set.add(e)
        elif ent.startswith('m.'):
            ent_set.add(ent)

    head_ent_name = {}
    for ent_id in ent_set:
        head_ent_name[ent_id] = convert_id_to_name(ent_id)

    #all_trips = get_1hop_triples(ent_set, return_name=True)
    for ent in ent_set:
        try:
            trips = get_1hop_triples(ent, return_name=True)
        except:
            continue
        for trip in trips:
            h,r,t = trip
            str_r = str(r).lower()
            if isinstance(t, tuple):
                tid, tw = t
                str_h = head_ent_name[h].lower()
                str_t = tw.lower()
                if (constrain in str_r
                    or constrain in str_t
                    or constrain in str_h):
                    new_trips.add((h,r,t[0]))
                    new_ents.add(t[0])
            elif isinstance(h, tuple):
                hid, hw = h
                str_t = hw.lower()
                str_h = head_ent_name[t].lower()
                if (constrain in str_r
                    or constrain in str_t
                    or constrain in str_h):
                    new_trips.add((t,r,h[0]))
                    new_ents.add(h[0])


    return list(new_trips), list(new_ents)


# def get_ent_triples_by_rel(entity_id, filtered_relations):
#     out_relations = get_out_relations(entity_id)
#     out_relations = list(set(out_relations))

#     in_relations = get_in_relations(entity_id)
#     in_relations = list(set(in_relations))

#     triples = []
#     for relation in filtered_relations:
#         if relation in out_relations:
#             tail_entity_ids = get_tail_entity(entity_id, relation)
#             triples.extend(
#                 [[entity_id, relation, tail_entity_id] for tail_entity_id in tail_entity_ids]
#             )
#         elif relation in in_relations:
#             head_entity_ids = get_head_entity(entity_id, relation)
#             triples.extend(
#                 [[head_entity_id, relation, entity_id] for head_entity_id in head_entity_ids]
#             )
#         else:
#             continue

#     id_to_lable = {}
#     for triple in triples:
#         for i in [0, -1]:
#             ent_id = triple[i]
#             if ent_id not in id_to_lable:
#                 id_to_lable[ent_id] = convert_id_to_name(ent_id)
#             triple[i] = id_to_lable[ent_id]

#     return triples, id_to_lable


# def convert_id_to_name_in_triples(triples, return_map=False):
#     id_to_label = {}
#     for triple in triples:
#         for i in [0, -1]:
#             ent_id = triple[i]
#             if ent_id[:2] in ["m.", "g."]:
#                 if ent_id not in id_to_label:
#                     id_to_label[ent_id] = convert_id_to_name(ent_id)
#                 triple[i] = id_to_label[ent_id]

#     if return_map:
#         return triples, id_to_label
#     else:
#         return triples


def convert_id_to_name(entity_id):
    if not entity_id.startswith("m."):
        return entity_id
    entity_id = remove_ns(entity_id)
    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?tailEntity
    WHERE {{
        {{
            ns:{entity} ns:type.object.name ?tailEntity .
            FILTER(langMatches(lang(?tailEntity), 'en'))
        }}
        UNION
        {{
            ns:{entity} ns:common.topic.alias ?tailEntity .
            FILTER(langMatches(lang(?tailEntity), 'en'))
        }}
        UNION
        {{
            ns:{entity} ns:type.object.key ?tailEntity .
            FILTER regex(?tailEntity, '/en/')
        }}
        UNION
        {{
            ns:{entity} ns:common.notable_for.display_name ?tailEntity .
            FILTER(langMatches(lang(?tailEntity), 'en'))
        }}
        UNION
        {{
            ns:{entity} ns:type.type.instance ?tailEntity .
        }}

    }}
    """.format(entity=entity_id)

    results = execute_query(query)

    if len(results) == 0:
        return entity_id
    else:
        return results[0]["tailEntity"]["value"].replace("http://rdf.freebase.com/ns/", "")


def convert_name_to_id(label):
    sparql_id = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?entity
    WHERE {{
        {{
            ?entity ns:type.object.name "%s"@en .
        }}
        UNION
        {{
            ?entity ns:common.topic.alias "%s"@en .
        }}
    }}
    """
    sparql_query = sparql_id % (label, label)

    results = execute_query(sparql_query)

    if len(results) == 0:
        return label
    else:
        return results[0]["entity"]["value"].replace(ns_prefix, "")


def pre_filter_relations(relations: List[str]):
    ignored_relations = [
        "type.object.type",
        "type.object.name",
    ]
    filtered_relations = []
    for relation in relations:
        if "/" in relation and ("ns/") not in relation:
            continue
        if (
            relation in ignored_relations
            or relation.startswith("freebase.")
            or relation.startswith("common.")
            or relation.startswith("kg.")
        ):
            continue
        else:
            filtered_relations.append(relation)
    return filtered_relations


def get_1hop_triples(entity_ids: Union[str, List], return_name=False):
    if type(entity_ids) is str:
        entity_ids = [entity_ids]

    tmp_ents = set()
    for ent in entity_ids:
        if ent.startswith("m.") and len(ent)>5:
            tmp_ents.add(ent)
    entity_ids = tmp_ents
    
    entity_ids = [add_ns(entity_id.strip()) for entity_id in entity_ids]

    query1 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?mid ?subject ?predicate ?object ?name WHERE {{
        VALUES ?mid {{ {entity_ids} }}
        {{ ?subject ?predicate ?mid }}
        OPTIONAL {{ 
        ?subject ns:type.object.name ?name.
        FILTER(langMatches(lang(?name), 'en'))
        }}
        FILTER regex(?predicate, "http://rdf.freebase.com/ns/")
        
    }}LIMIT 2000
    """.format(
        entity_ids=" ".join(entity_ids)
    )

    query2 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?mid ?subject ?predicate ?object WHERE {{
        VALUES ?mid {{ {entity_ids} }}
        {{ ?mid ?predicate ?object }}
        OPTIONAL {{ 
        ?object ns:type.object.name ?name.
        FILTER(langMatches(lang(?name), 'en'))
        }}
        FILTER regex(?predicate, "http://rdf.freebase.com/ns/")
        
    }}LIMIT 2000
    """.format(
        entity_ids=" ".join(entity_ids)
    )

    if len(tmp_ents) > 0:
        results = execute_query(query1) + execute_query(query2)
    else:
        results = []

    triples = set()
    for result in results:
        mid = result["mid"]["value"].replace(ns_prefix, "")
        predicate = result["predicate"]["value"].replace(ns_prefix, "")
        if "subject" in result:
            subject = result["subject"]["value"].replace(ns_prefix, "")
            name = result['name']['value'] if 'name' in result else subject
            if return_name:
                triples.add(((subject,name) , predicate, mid))
            else:
                triples.add((subject, predicate, mid))
        elif "object" in result:
            object = result["object"]["value"].replace(ns_prefix, "")
            name = result['name']['value'] if 'name' in result else object
            if return_name:
                triples.add((mid, predicate, (object, name)))
            else:
                triples.add((mid, predicate, object))

    relations = list(set([triple[1] for triple in triples]))
    relations = pre_filter_relations(relations)

    triples = [list(triple) for triple in triples if triple[1] in relations]

    return triples


def process_middle_node(entity_ids, latest=False):
    if type(entity_ids) is str:
        entity_ids = [entity_ids]
    entity_ids = [add_ns(entity_id.strip()) for entity_id in entity_ids]
    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?instance ?y WHERE {{
        VALUES ?mid {{ {entity_ids} }}
        {{ ?instance ns:type.type.instance ?mid }}
        UNION
        {{ ?mid ns:type.type.instance ?instance }}
    }}
    """.format(
        entity_ids=" ".join(entity_ids)
    )
    results = execute_query(query)
    instances = []
    for x in results:
        instances.append(x['instance']['value'].replace("http://rdf.freebase.com/ns/", ""))
    if instances:
        domain = instances[0].split(".")[0]
    else:
        domain = None

    query2 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?x ?y ?r WHERE {{
        VALUES ?x {{ {entity_ids} }}
        {{ ?y ?r ?x }}
        UNION
        {{ ?x ?r ?y }}
    }}
    """.format(
        entity_ids=" ".join(entity_ids), 
        #FILTER regex(?r, "{instance}") instance=" ".join(["_".join(instance.split("_")[:-1]) + "." + instance.split("_")[-1] for instance in instances])
    )
    
    results2 = execute_query(query2)
    mids = set()
    time_lst = []
    if len(results2) != 0:
        for res in results2:
            h = res['x']['value'].replace("http://rdf.freebase.com/ns/", "")
            r = res['r']['value'].replace("http://rdf.freebase.com/ns/", "") 
            t = res['y']['value'].replace("http://rdf.freebase.com/ns/", "")
            if (r.startswith("freebase.")
                or r.startswith("common.")
                or r.startswith("kg.")
                or r.startswith("http")
                or r.startswith("type")):
                continue
            if domain and domain not in r:
                continue
            mids.add((h,r,t))
            if 'datatype' in res['x']:
                time_lst.append((res['x']['value'], r, t))
            elif 'datatype' in res['y']:
                time_lst.append((res['y']['value'], r, h))

    time_lst = sorted(time_lst, reverse=True)
    if latest:
        latest_mid = time_lst[0][-1] if time_lst else None
    else:
        latest_mid = None
    res = set()

    for t in mids:
        if latest_mid and latest_mid not in t:
            continue
        name = convert_id_to_name(t[-1])
        if not name.startswith("m."):
            res.add((t[0],t[1],t[-1], name))
    return res


def get_types(entity_id: str) -> List[str]:
    if not entity_id.startswith("m."):
        return [entity_id]
    
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {{
    SELECT DISTINCT ?x0  WHERE {{
        ns:{entity_id} ns:type.object.type ?x0 . 
    }}
    }}
    """.format(
        entity_id=entity_id
    )

    results = execute_query(query)

    typs = []
    for result in results:
        typs.append(result["value"]["value"].replace("http://rdf.freebase.com/ns/", ""))

    return typs


def get_name_by_id(ent_id, env=None):
    from utils import cal_sim_score
    if not ent_id.startswith("m."):
        return ent_id
    else:
        name = convert_id_to_name(ent_id)
        if not env or not name:
            return name
        if name.startswith("m."):
            # not find name, use type instead
            name = get_types(ent_id)
            if len(name) > 1:
                name_lst = cal_sim_score(env.state.query, name, model=env.agent.emb, topk=1)
            else:
                name_lst = name
            name = name_lst[0] if name_lst else ent_id   # keep most related type
        if name != ent_id and "." in name:
            name = f"{name}({ent_id})"
        return name
    

def filter_by_function(ents, func, select_ent_set=None, env=None):
    func_name, func_input = func
    if len(ents) > env.args.max_trips:
        tmp = [x[1] for x in ents]
        _, ids = cal_sim_score(env.state.cur_query, tmp, env.agent.emb, topk=env.args.max_trips, return_idx=True)
        ents = [ents[id] for id in ids]
    ent_ids = set([x[0] for x in ents])
    filtered_ents = set()
    new_trips_in_filter = set()
    operator_map = {"=": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y}

    # if func_name == "FilterbyType":
        # func_input = func_input.replace("(", "").replace(")", "").replace("'","").replace('"',"")
        # func_input = convert_id_to_name(func_input)
        # if func_input in concept_name_set:
        #     typ_name = func_input
        # else:
        #     typ_name = cal_sim_score(func_input, list(concept_name_set), model=env.agent.emb, topk=1)[0]
        # for ent in ents:
        #     qid, name = ent
        #     typs = get_types(qid)
        #     typ_name_lst = []
        #     for typ in typs:
        #         typ_nams = id_to_name_map[typ]
        #         for name in typ_nams:
        #             typ_name_lst.append(name)

        #     if typ_name in typ_name_lst:
        #         filtered_ents.add(ent)

    if func_name in ["FilterbyCondition", "Count", "Verify"]:
        func_input = func_input.split("(")[-1]
        tmp = func_input.replace("(","").replace(")","").replace("'","").replace('"',"").split(", ")
        if "" in tmp:
            tmp.remove("")

        OP = re.findall(r"argmin|argmax", func_input) + re.findall(r">=|<=", func_input) + re.findall(r"=|>|<", func_input)
        OP = OP[0] if len(OP)>0 else []
        if len(OP) > 0:
            if OP in operator_map and len(tmp) == 3:            #(K, OP, V)
                K, _, V = tmp
            elif OP in ['argmax', 'argmin'] and len(tmp) >= 2:  #(K, argmin, V / none)
                K = tmp[0]
                V = "<NONE>"
            else:
                K = tmp[0]
                V = tmp[0]
        else:
            if len(tmp) == 2:                                   # (K, V)
                K,V = tmp
                OP = "="
            else:                                               #(K / V)
                K = tmp[0]
                V = tmp[0]
                OP = '='

        raw_V = V
        V = V.split(": ")[-1].split("=")[-1].strip()
        K = K.split(": ")[-1].split("=")[-1].strip()
        V = convert_name_to_id(V)
        V = ParseAttributeValue(V)
        value_lst = []

        def parse_trip(trip):
            raw_h, r, raw_t = trip
            if raw_h in ent_ids:
                source = raw_h
                tail = raw_t
            else:
                source = raw_t
                tail = raw_h

            t = ParseAttributeValue(tail)
            h = ParseAttributeValue(source)
            if OP in operator_map:
                if type(t) == type(V) and operator_map[OP](t, V):
                    filtered_ents.add(source)
                    new_trips_in_filter.add((source, f"({r} {OP} {raw_V})", tail))

                if type(h) == type(V) and operator_map[OP](h, V):
                    filtered_ents.add(source)
                    new_trips_in_filter.add((source, f"({r} {OP} {raw_V})", tail))

                if raw_V == "ANY":
                    filtered_ents.add(source)
                    new_trips_in_filter.add((source, r, tail))
            else:   # argmax, argmin
                if len(value_lst)>0 and isinstance(t, type(value_lst[0][0])):
                    value_lst.append((t, raw_t))
                elif len(value_lst)==0 and not isinstance(t, str):
                    value_lst.append((t, raw_t))


        rels = get_all_relations([x[0] for x in ents])
        if len(rels) > 0:                                   # h -> <K> -> t <op> <V>
            if K not in rels:
                # not input an available relation, use FilterbyStr instead
                V_trips, out_ents = filter_entities_by_constrain(ents, raw_V)
                K_trips, out_ents = filter_entities_by_constrain(ents, K)
                out_trips = V_trips + K_trips
                rels = set([x[1] for x in out_trips])
                matched_rel = cal_sim_score(K, list(rels), model=env.agent.emb, topk=3 if OP in operator_map else 1)
                for trip in out_trips:
                    h,r,t = trip
                    if r in matched_rel:
                        parse_trip(trip)
            else:
                for ent in ents:
                    qid, name = ent
                    if qid.startswith("m."):
                        trips, _ = get_all_entities(qid, K)
                        for trip in trips:
                            parse_trip(trip)
        else:                                            # h <OP> <V>
            for ent in ents:
                eid, name = ent
                parse_trip((eid, K, eid))

        # matched_rel = cal_sim_score(K, list(all_rels), model=env.agent.emb, topk=3)
        # if K == matched_rel[0]:
        #     matched_rel = [K]

        # tmp_trips = []
        # for rel in matched_rel:
        #     trip, out_ents = filter_entities_by_constrain(ents, constrain=rel)
        #     tmp_trips += trip
    
        # for trip in tmp_trips:
        #     h,r,t = trip
        #     if r in matched_rel:
        #         parse_trip(trip)

        if OP == "argmax":
            filtered_ents = [f"{list(select_ent_set)}"]
            new_trips_in_filter = [(f"{list(select_ent_set)}", f"MAX({K})", f"{max(value_lst)[-1]}")]
        if OP == 'argmin':
            filtered_ents = [f"{list(select_ent_set)}"]
            new_trips_in_filter = [(f"{list(select_ent_set)}", f"MIN({K})", f"{min(value_lst)[-1]}")]
        if func_name == 'Count':
            new_trips_in_filter = [(f"{list(select_ent_set)}", f"COUNT({K} {OP} {raw_V})", f"{len(filtered_ents)}")]
            filtered_ents = [f"{list(select_ent_set)}"]
        if func_name == "Verify":
            new_trips_in_filter = [(f"{list(select_ent_set)}", f"VERIFY({K} {OP} {raw_V})", f"{bool(filtered_ents)}")]
            filtered_ents = [f"{list(select_ent_set)}"]
        
    if func_name == "FindRelation":
        ent1, ent2 = func_input.replace("(", "").replace(")", "").replace("'","").replace('"',"").split(", ")[:2]
        if 'ent_set' in ent1:
            ent1_ids = set(re.findall(r"ent_set_\d+(?:-\d+)*", str(ent1)))
            ent1 = set()
            for idx in ent1_ids:
                ent_set = env.state.set_to_ent_map[idx]
                for ent in ent_set:
                    ent1.add(ent[0])
        else:
            ent1 = [convert_name_to_id(ent1.split(": ")[-1])]
        if 'ent_set' in ent2:
            ent2_ids = set(re.findall(r"ent_set_\d+(?:-\d+)*", str(ent2)))
            ent2 = set()
            for idx in ent2_ids:
                ent_set = env.state.set_to_ent_map[idx]
                for ent in ent_set:
                    ent2.add(ent[0])
        else:
            ent2 = [convert_name_to_id(ent2.split(": ")[-1])]
        
        for e1 in ent1:
            for e2 in ent2:
                out_trips, out_ents = filter_entities_by_constrain(e1, e2)
                filtered_ents.update(out_ents)
                new_trips_in_filter.update(out_trips)

    # if func_name == "FilterbyRelation":
    #     qr = "".join(re.findall(r"(?<=\().+(?=,)", str(func_input), re.I)).replace("'","").replace('"',"").strip()
    #     qr = cal_sim_score(qr, rel_name_lst, model=env.agent.emb, topk=1)[0] if qr not in rel_name_lst else qr

    #     qe = "".join(re.findall(r"(?<=,).+(?=\))", str(func_input), re.I)).replace("'","").replace('"',"").strip()
    #     if not qe.startswith("Q"):
    #         qe = cal_sim_score(qe, list(entity_name_set), model=env.agent.emb, topk=1)[0] if qe not in entity_name_set else qe
    #         qe = convert_name_to_id(qe)
        
    #     for ent in ents:
    #         qid, name = ent
    #         tmp_trips, _ = get_all_entities(qid, qr)
    #         for trip in tmp_trips:
    #             h,r,t = trip
    #             if h in qe or t in qe:
    #                 filtered_ents.add(ent)
    #                 new_trips_in_filter.add(trip)

    if func_name == "FilterbyStr":
        func_input = func_input.replace("(", "").replace(")", "").replace("'","").replace('"',"")
        out_trips, out_ents = filter_entities_by_constrain(ents, func_input)
        filtered_ents = out_ents
        new_trips_in_filter = out_trips

    if func_name == "LogicOperation":
        ent_set_ids = set(re.findall(r"ent_set_\d+(?:-\d+)*", str(func_input)))
        OP = "".join(re.findall(r"(intersect|union)", str(func_input))).strip()
        tmp = set()
        for idx in ent_set_ids:
            ent_set = env.state.set_to_ent_map[idx]
            if OP == 'intersect':
                tmp = tmp & set(ent_set)
            elif OP == "union":
                tmp = tmp | set(ent_set)

        for ent in tmp:
            new_trips_in_filter.add((f"{list(select_ent_set)}", f"{OP.upper()}", ent))
            filtered_ents.add(f"{list(select_ent_set)}")

        if len(new_trips_in_filter) == 0:
            new_trips_in_filter.add((f"{list(select_ent_set)}", f"{OP.upper()}", "NONE"))
            filtered_ents.add(f"{list(select_ent_set)}")
    
    return filtered_ents, new_trips_in_filter

def ParseAttributeValue(value):
    if ValidDate(value):
        return ValidDate(value)
    if value.isalpha():
        return value.lower()
    if value.isdigit():
        new_v = int(value)
        if str(new_v) == value:     # '010' != '10'
            return new_v
        else:
            return value.lower()
    try:
        value = float(value)
    except:
        value = value.lower()
    return value
    
def ValidDate(date):
    try:
        if ":" in date and "-" in date:
            return time.strptime(date, "%Y-%m:%S")
        if ":" in date:
            return time.strptime(date, "%Y-%m-%d %H:%M:%S")
        elif "-" in date:
            return time.strptime(date, "%Y-%m-%d")
        else:
            return time.strptime(date, "%Y")
    except:
        return False

