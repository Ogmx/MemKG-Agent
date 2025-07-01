# based on ToT
import numpy as np
import requests
import pickle
from retry import retry
from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, SPARQLExceptions, JSON
from utils import print_error

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

# KQApro only use a subset of WikiData
with open('/raid_sdb/home/tsy/KG-Agent/data/kga_kb_info.pkl', 'rb') as f:
    kga_kg_info = pickle.load(f)

available_relations = set(kga_kg_info['relations'])
available_nodes = set(kga_kg_info['id_to_ent_map'].keys())                     # QIDs

pid_to_label_map = {}
label_to_pid_map = {}
name_to_qid_map = kga_kg_info['ent_to_id_map']
qid_to_name_map = kga_kg_info['id_to_ent_map']

global filter_ent, filter_rel    
filter_ent = True      # if only ues available_nodes
filter_rel = True       # if only ues available_relations

def pre_filter_relations(rels):
    tmp = []
    for rel in rels:
        pid, label = rel
        if filter_rel and label not in available_relations:
            continue
        else:
            tmp.append(label)
        pid_to_label_map[pid] = label
        label_to_pid_map[label] = pid
    return tmp

def pre_filter_entity(trips):
    trip_set = set()
    ent_set = set()
    for trip in trips:
        h, r, t = trip
        if h in available_nodes or t in available_nodes:
            trip_set.add(trip)
            ent_set.add(h)
            ent_set.add(t)
    return list(trip_set), list(ent_set)


def convert_name_to_id_by_web(name):
    
    def fetch_wikidata(params):
        url = 'https://www.wikidata.org/w/api.php'
        try:
            return requests.get(url, params=params)
        except:
            return None
     
    # Which parameters to use
    params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'search': name,
            'language': 'en'
        }
     
    # Fetch API
    data = fetch_wikidata(params)
     
    #show response as JSON
    data = data.json()
    if data and data['search']:
        return data['search'][0]['id']
    else:
        return name
    
#@retry(SPARQLExceptions.EndPointInternalError, tries=3, backoff=2)
def convert_name_to_id(title):
    title_to_id_query = """SELECT ?lemma ?item WHERE {{
      VALUES ?lemma {{
        {titles}
      }}
      {{?sitelink schema:about ?item;
        schema:isPartOf <https://en.wikipedia.org/>;
        schema:name ?lemma.}}
      UNION
      {{?lemma rdfs:label ?item}}
      UNION
      {{?item rdfs:label ?lemma}}
    }}"""

    if filter_ent:
        if title in name_to_qid_map:
            return name_to_qid_map[title]
        else:
            return title
        
    kga_kg_info['ent_to_id_map'][title]
    titles_text = [title.replace('"', '\\"').replace("'","\\'")]
    titles_text = " ".join([f"\'{title}\'@en" for title in titles_text])
    query = title_to_id_query.format(titles=titles_text)
    results = execute_query(query)
    # title_to_qid_map = {}
    # for r in results:
    #     title_to_qid_map[r['lemma']['value']] = r['item']['value'].split('/')[-1]
    if len(results):
        qid = results[0]['item']['value'].split("/")[-1]
    else:
        qid = convert_name_to_id_by_web(title)
    return qid

def get_name_by_id(wid, env=None):
    if wid.startswith("Q"):
        wid =  convert_id_to_name(wid)
    elif wid.startswith("P"):
        wid = pid_to_label_map[wid] if wid in pid_to_label_map else wid
    return wid

# @retry(SPARQLExceptions.EndPointInternalError, tries=3, backoff=2, max_delay=60)
def convert_id_to_name(wid):
    if not wid.startswith("Q") and not wid.startswith("P"):
         return wid
    query = """SELECT ?id ?r ?y WHERE {{
      {{?y rdfs:label wd:{id}}}
      UNION
      {{ wd:{id} rdfs:label ?y}}
      FILTER(LANG(?y) = "en").
    }}""".format(id=wid)
    results = execute_query(query)
    if len(results):
        label = results[0]['y']['value']
    else:
        label = wid
    return label

def get_forward_relations(entities):
    # Gets predicates of subject
    query_forward_properties = """
    SELECT ?property ?predicateType ?label WHERE {{
      {{
        SELECT DISTINCT ?property ?predicateType WHERE {{
          VALUES ?subject {{ {entities} }}
          ?subject ?predicateType ?object.
          ?property wikibase:directClaim ?predicateType.
        }}
      }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?property rdfs:label ?label.
      }}

      FILTER (!regex(str(?label), "ID"))
    }}
    ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'http://www.wikidata.org/entity/P')))
    LIMIT 200
    """
    return get_available_relations(entities, query_forward_properties)


def get_backward_relations(entities):
    # Gets predicates of object
    query_backward_properties = """
    SELECT ?property ?predicateType ?label WHERE {{
      {{
        SELECT DISTINCT ?property ?predicateType WHERE {{
          VALUES ?object {{ {entities} }}
          ?subject ?predicateType ?object.
          ?property wikibase:directClaim ?predicateType.
        }} LIMIT 50
      }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?property rdfs:label ?label.
      }}

      FILTER (!regex(str(?label), "ID"))
    }}
    ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'http://www.wikidata.org/entity/P')))
    LIMIT 200
    """
    return get_available_relations(entities, query_backward_properties)

#@retry(SPARQLExceptions.EndPointInternalError, tries=3, backoff=2, max_delay=60)
def get_available_relations(entities, base_query):
    entities_str = " ".join(f"wd:{e}" for e in entities)
    query = base_query.format(entities=entities_str)
    results = execute_query(query)
    rels = set()
    for r in results:
        pid = r['property']['value'].replace("http://www.wikidata.org/entity/","")
        label = r['label']['value']
        rels.add((pid, label))
    return rels

def get_all_relations(entities):
    if isinstance(entities, str):
        entities = [entities]
    all_rels = set()
    fr = get_forward_relations(entities)
    br = get_backward_relations(entities)

    all_rels.update(fr)
    all_rels.update(br)

    all_rels = pre_filter_relations(list(all_rels))
    return all_rels


def handle_wikidata_datetime(time, precision):
    if time[0] == '-':
        era = " BCE"
        time = time[1:]
    else:
        era = ""
    try:
        dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return time
    if precision ==  11:  # day precision (month day, year)
            time = dt.strftime("%B %d, %Y") + era
    elif precision == 10:  # month (month year)
            time = dt.strftime("%B %Y") + era
    elif precision == 9:  # year
            time = dt.strftime("%Y") + era
    elif precision == 8:  # decade
            time = dt.strftime("%Ys") + era
    elif precision == 7:  # century
            century = (dt.year - 1) // 100 + 1
            suffix = "th" if 11 <= century <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(century % 10, "th")
            time = f"{century}{suffix} century{era}"
    return time


#@retry(SPARQLExceptions.EndPointInternalError, tries=3, backoff=2, max_delay=60)
def get_edges(entities, property, use_incoming):
    """
    :param entities: entities from which to expand
    :param property: relation to expand along
    :param use_incoming: (bool) if true does queries of the form (?, r, o). If false does (s, r, ?)
    :return: list of edges, list of entity objects, entity labels, entity descriptions
    """
    query_outgoing = """SELECT ?baseEntity ?statement ?object ?label ?description ?timeprecision
    WHERE {{
      VALUES ?baseEntity {{ {entities} }}
      ?baseEntity p:{property} ?statement.
      ?statement ps:{property} ?object.
      ?baseEntity wdt:{property} ?object.
      OPTIONAL {{
        ?statement psv:{property}/wikibase:timePrecision ?timeprecision.
      }}
      OPTIONAL {{
        ?object rdfs:label ?label.
        FILTER(LANG(?label) = "en").
      }}
      OPTIONAL {{
        ?object schema:description ?description.
        FILTER(LANG(?description) = "en").
      }}
    }}
    LIMIT 200
    """

    query_incoming = """
    SELECT ?baseEntity ?statement ?object ?label ?description
    WHERE {{
      VALUES ?baseEntity {{ {entities} }}
      ?statement ps:{property} ?baseEntity.
      ?object p:{property} ?statement.
      ?object wdt:{property} ?baseEntity.
      OPTIONAL {{
        ?object rdfs:label ?label.
        FILTER(LANG(?label) = "en").
      }}
      OPTIONAL {{
        ?object schema:description ?description.
        FILTER(LANG(?description) = "en").
      }}
    }}
    LIMIT 200
    """

    query = query_incoming if use_incoming else query_outgoing
    entities_str = " ".join(f"wd:{e}" for e in entities)
    query = query.format(entities=entities_str, property=property)
    sparql.setQuery(query)
    results = execute_query(query)
    trips = set()
    
    new_entities = {}
    new_entity_pos = 'object'
    for r in results:
        source = r['baseEntity']['value'].replace("http://www.wikidata.org/entity/","")
        rel = property
        object = r['object']['value'].replace("http://www.wikidata.org/entity/","")
        try:
            timeprecision = int(r['timeprecision']['value'])
            object = handle_wikidata_datetime(object, timeprecision)
        except:
            timeprecision = None

        if use_incoming:
            trips.add((object, rel, source))
        else:
            trips.add((source, rel, object))
        
        new_ent = r[new_entity_pos]['value'].replace("http://www.wikidata.org/entity/","")  # just get the QID
        if 'label' in r:
            label = r['label']['value']
            desc = r['description']['value'] if 'description' in r else ""
        else:  # if object is not an entity (e.g. datetime), it won't have label and description
            label = new_ent
            desc = None
        new_entities[new_ent] = {'label': label, 'description': desc, 'wikidata': new_ent}

    return trips, new_entities


def get_all_entities(entities:list, relation:str, constrain=None):
    if isinstance(relation, list):
        relation = relation[0]
    if not relation.startswith("P"):
         if relation in label_to_pid_map:
            rid = label_to_pid_map[relation]
         else:
            rid = convert_name_to_id(relation)

    out_trips = []
    out_ents = set()

    def parser_results(trips, new_ents, out_trips=out_trips, out_ents=out_ents):
        for trip in trips:
            h, r, t = trip
            if filter_ent:
                if  h in available_nodes and t in available_nodes:
                    out_trips.append([h, pid_to_label_map[r], t])
            else:
                out_trips.append([h, pid_to_label_map[r], t])
        for ent in new_ents:
            if filter_ent:
                if ent in available_nodes:
                    out_ents.add(ent)
            else:
                out_ents.add(ent)

    new_trips, new_ents = get_edges(entities, property=rid, use_incoming=True)
    parser_results(new_trips, new_ents)
    new_trips, new_ents = get_edges(entities, property=rid, use_incoming=False)
    parser_results(new_trips, new_ents)

    if constrain:
        filtered_ents = filter_entities_by_constrain(list(out_ents), constrain)
        if filtered_ents:
            out_ents = filtered_ents
            tmp = []
            for trip in out_trips:
                h,r,t = trip
                if h in out_ents or t in out_ents:
                    tmp.append(trip)
            out_trips = tmp

    return out_trips, list(out_ents)


def filter_entities_by_constrain(entities, constrain):
    O = constrain[0]
    V = constrain[1]

    query_ents = []
    for ent in entities:
        if ent.startswith("Q"):
            query_ents.append(ent)

    if len(query_ents) == 0:
        return []
    
    query1= """SELECT ?x ?y ?r ?r_label ?y_label
    WHERE {{
      VALUES ?x {{ {entities} }}
      ?x ?r ?y.
      ?r_pid wikibase:directClaim ?r.
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?r_pid rdfs:label ?r_label.
      }}
      OPTIONAL {{
        ?y rdfs:label ?y_label.
        FILTER(LANG(?y_label) = "en").
      }}
      }}
      LIMIT 500""".format(entities=" ".join(f"wd:{e}" for e in query_ents), V=V)
    
    query2= """SELECT ?x ?y ?r ?r_label ?y_label
    WHERE {{
      VALUES ?x {{ {entities} }}
      ?y ?r ?x.
      ?r_pid wikibase:directClaim ?r.
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?r_pid rdfs:label ?r_label.
      }}
      OPTIONAL {{
        ?y rdfs:label ?y_label.
        FILTER(LANG(?y_label) = "en").
      }}
      }}
      LIMIT 500""".format(entities=" ".join(f"wd:{e}" for e in query_ents), V=V)
    results = execute_query(query1) + execute_query(query2)

    filtered_ents = set()
    for res in results:
        x = res['x']['value'].replace("http://www.wikidata.org/entity/","")
        rid = res['r']['value']
        yid = res['y']['value']
        r_label = res['r_label']['value'] if 'r_label' in res else "NONE"
        y_label = res['y_label']['value'] if 'y_label' in res else "NONE"
        if V.lower() in r_label.lower() or V.lower() in y_label.lower():
            filtered_ents.add(x)
        elif V in rid or V in yid:
            filtered_ents.add(x)

    return list(filtered_ents)

def qid_to_mid(qid):
    query = """
        SELECT DISTINCT ?mid WHERE {{
            <http://www.wikidata.org/entity/{entity}> <http://www.wikidata.org/prop/direct/P646> ?mid
        }}
        """.format(
    entity=qid
    )
    results = execute_query(query)
    if len(results):
        mid = results[0]['mid']['value'].replace("/",".")[1:]
    else:
        mid = None

    return mid

@retry(SPARQLExceptions.EndPointInternalError, tries=3, backoff=2, max_delay=60)
def get_types(entity_id: str):
    if not entity_id.startswith("Q"):
        return [entity_id]
    # P31: instance of
    query = """
    SELECT DISTINCT ?y ?label  WHERE {{
        wd:{entity_id} wdt:P31 ?y .
        OPTIONAL {{
            ?y rdfs:label ?label.
            FILTER(LANG(?label) = "en").
          }}
    }}
    """.format(
        entity_id=entity_id
    )
    results = execute_query(query)
    typs = []
    for result in results:
        typs.append(result['label']['value'])

    return typs
    
def process_middle_node(entities):
    ...


def execute_query(query):

    @retry(SPARQLExceptions.EndPointInternalError, tries=3, backoff=2, max_delay=60)
    def execute(query):
        sparql.setQuery(query)
        results = sparql.query().convert()
        results = results["results"]["bindings"]
        return results

    try:
        results = execute(query)
    except Exception as e:
        print("------------------- API ERROR ----------------\n")
        print_error(e)
        results = []

    return results