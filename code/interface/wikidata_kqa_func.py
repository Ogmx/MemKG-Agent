import json
import re
import numpy as np
import datetime, time
from collections import Counter
#from utils.misc import init_vocab
from datetime import date
from queue import Queue
from utils import cal_sim_score
#from utils.value_class import ValueClass

class ValueClass():
    def __init__(self, type, value, unit=None):
        """
        When type is
            - string, value is a str
            - quantity, value is a number and unit is required
            - year, value is a int
            - date, value is a date object
        """
        self.type = type
        self.value = value
        self.unit = unit
        self.preDefinedUnits = {'minute': ['minutes', 'minute'], 'metre': ['metres', 'metre']}

    def isTime(self):
        return self.type in {'year', 'date'}

    def can_compare(self, other):
        if self.type == 'string':
            if(other.type == 'string'):
                return True
            elif(other.type == 'quantity'):
                try:
                    other.value = str(int(other.value))
                    other.type = 'string'
                    print("Changing datatype to string!")
                except:
                    pass
            else:
                try:
                    other.value = str(other.value)
                    other.type = 'string'
                    print("Changing datatype to string!")
                except:
                    pass

            return other.type == 'string'
        elif self.type == 'quantity':
            # NOTE: for two quantity, they can compare only when they have the same unit
            if(other.type == 'string' or other.type == 'year'):
                other.value = str(other.value)
                try:
                    if ' ' in other.value:
                        vs = other.value.split()
                        v = vs[0]
                        unit = ' '.join(vs[1:])
                        other.value = float(v)
                        other.unit = unit
                    else:
                        v = other.value
                        unit = '1'
                        other.value = float(v)
                        other.unit = unit
                    print("Changing datatype to quantity!")
                except:
                    pass
            

            if(self.unit in self.preDefinedUnits.keys()):
                return other.type == 'quantity' and other.unit in self.preDefinedUnits[self.unit]
            elif(other.unit in self.preDefinedUnits.keys()):
                return other.type == 'quantity' and self.unit in self.preDefinedUnits[other.unit]
            return other.type == 'quantity' and other.unit == self.unit
        else:
            # year can compare with date
            if(other.type == 'quantity' or other.type == 'string'):
                try:
                    value = other.value
                    if(other.type == 'quantity'):
                        value = str(int(other.value))

                    if '/' in value or ('-' in value and '-' != value[0]):
                        split_char = '/' if '/' in value else '-'
                        p1, p2 = value.find(split_char), value.rfind(split_char)
                        y, m, d = int(value[:p1]), int(value[p1+1:p2]), int(value[p2+1:])
                        other.value = date(y, m, d)
                        other.type = 'date'
                    else:
                        other.value = int(value)
                        other.type = 'year'
                    print("Changing datatype to date/year!")
                except:
                    pass

            


            return other.type == 'year' or other.type == 'date'


"""
knowledge json format:
    'concepts':
    {
        'id':
        {
            'name': '',
            'instanceOf': ['<concept_id>'],
        }
    },
    'entities': # exclude concepts
    {
        'id': 
        {
            'name': '<entity_name>',
            'instanceOf': ['<concept_id>'],
            'attributes':
            [
                {
                    'key': '<key>',
                    'value': 
                    {
                        'type': 'string'/'quantity'/'date'/'year'
                        'value':  # float or int for quantity, int for year, 'yyyy/mm/dd' for date
                        'unit':   # for quantity
                    },
                    'qualifiers':
                    {
                        '<qk>': 
                        [
                            <qv>, # each qv is a dictionary like value, including keys type,value,unit
                        ]
                    }
                }
            ]
            'relations':
            [
                {
                    'predicate': '<predicate>',
                    'object': '<object_id>', # NOTE: it may be a concept id
                    'direction': 'forward' or 'backward',
                    'qualifiers':
                    {
                        '<qk>': 
                        [
                            <qv>, # each qv is a dictionary like value
                        ]
                    }
                }
            ]
        }
    }
"""
def get_kb_vocab(kb_json, min_cnt=1):
    counter = Counter()
    kb = json.load(open(kb_json))
    for i in kb['concepts']:
        counter.update([i, kb['concepts'][i]['name']])
    for i in kb['entities']:
        counter.update([i, kb['entities'][i]['name']])
        for attr_dict in kb['entities'][i]['attributes']:
            counter.update([attr_dict['key']])
            values = [attr_dict['value']]
            for qk, qvs in attr_dict['qualifiers'].items():
                counter.update([qk])
                values += qvs
            for value in values:
                u = value.get('unit', '')
                if u:
                    counter.update([u])
                counter.update([str(value['value'])])
        for rel_dict in kb['entities'][i]['relations']:
            counter.update([rel_dict['predicate'], rel_dict['direction']])
            values = []
            for qk, qvs in rel_dict['qualifiers'].items():
                counter.update([qk])
                values += qvs
            for value in values:
                u = value.get('unit', '')
                if u:
                    counter.update([u])
                counter.update([str(value['value'])])

    vocab = init_vocab()
    for v, c in counter.items():
        if v and c >= min_cnt and v not in vocab:
            vocab[v] = len(vocab)
    return kb, vocab

def init_vocab():
    return {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3
    }

def load_as_graph(kb_json, max_desc=200, min_cnt=1):
    kb, vocab = get_kb_vocab(kb_json, min_cnt)
    id2idx = {}
    pred2idx = {}
    node_descs = []
    triples = []
    for i, info in kb['concepts'].items():
        id2idx[i] = len(id2idx)
        desc = [info['name']]
        node_descs.append(desc)
    for i, info in kb['entities'].items():
        id2idx[i] = len(id2idx)
        desc = [info['name']]
        for attr_info in info['attributes']:
            desc.append(attr_info['key'])
            desc.append(str(attr_info['value']['value']))
            u = attr_info['value'].get('unit', '')
            if u:
                desc.append(u)
        node_descs.append(desc)
        for rel_info in info['relations']:
            obj_id = rel_info['object']
            if obj_id not in id2idx:
                continue
            pred = rel_info['predicate']
            if pred not in pred2idx:
                pred2idx[pred] = len(pred2idx)
            pred_idx = pred2idx[pred]
            sub_idx = id2idx[i]
            obj_idx = id2idx[obj_id]
            if rel_info['direction'] == 'forward':
                triples.append((sub_idx, pred_idx, obj_idx))
            else:
                triples.append((obj_idx, pred_idx, sub_idx))
    # encode and pad desc
    for i, desc in enumerate(node_descs):
        desc = [vocab.get(w, vocab['<UNK>']) for w in desc]
        while len(desc) < max_desc:
            desc.append(vocab['<PAD>'])
        node_descs[i] = desc[:max_desc]

    return vocab, node_descs, triples, id2idx, pred2idx



def load_as_key_value(kb_json, min_cnt=1):
    """
    For KVMemNN
    Load each triple (s, r, o) as kv pairs (s+r, o) and (o+r_, s)
    """
    keys = ['<PAD>'] # use <PAD> as the first key
    values = ['<PAD>']
    def add_sro(s, r, o):
        keys.append('{} {}'.format(s, r))
        values.append(o)
        keys.append('{} {}_'.format(o, r))
        values.append(s)

    kb = json.load(open(kb_json))
    for i in kb['concepts']:
        for j in kb['concepts'][i]['instanceOf']:
            s = kb['concepts'][i]['name']
            o = kb['concepts'][j]['name']
            add_sro(s, 'instanceOf', o)
    for i in kb['entities']:
        for j in kb['entities'][i]['instanceOf']:
            s = kb['entities'][i]['name']
            o = kb['concepts'][j]['name']
            add_sro(s, 'instanceOf', o)
        name = kb['entities'][i]['name']
        for attr_dict in kb['entities'][i]['attributes']:
            o = '{} {}'.format(attr_dict['value']['value'], attr_dict['value'].get('unit', ''))
            add_sro(name, attr_dict['key'], o)
            s = '{} {} {}'.format(name, attr_dict['key'], o)
            for qk, qvs in attr_dict['qualifiers'].items():
                for qv in qvs:
                    o = '{} {}'.format(qv['value'], qv.get('unit', ''))
                    add_sro(s, qk, o)

        for rel_dict in kb['entities'][i]['relations']:
            if rel_dict['direction'] == 'backward': # we add reverse relation in add_sro
                continue
            o = kb['entities'].get(rel_dict['object'], kb['concepts'].get(rel_dict['object'], None))
            if o is None: # wtf, why are some objects not in kb?
                continue
            o = o['name']
            add_sro(name, rel_dict['predicate'], o)
            s = '{} {} {}'.format(name, rel_dict['predicate'], o)
            for qk, qvs in rel_dict['qualifiers'].items():
                for qv in qvs:
                    o = '{} {}'.format(qv['value'], qv.get('unit', ''))
                    add_sro(s, qk, o)
    print('length of kv pairs: {}'.format(len(keys)))
    counter = Counter()
    for i in range(len(keys)):
        keys[i] = keys[i].lower().split()
        values[i] = values[i].lower().split()
        counter.update(keys[i])
        counter.update(values[i])

    vocab = init_vocab()
    for v, c in counter.items():
        if v and c >= min_cnt and v not in vocab:
            vocab[v] = len(vocab)
    return vocab, keys, values


class DataForSPARQL(object):
    def __init__(self, kb_path):
        kb = json.load(open(kb_path))
        self.concepts = kb['concepts']
        self.entities = kb['entities']

        # replace adjacent space and tab in name, which may cause errors when building sparql query
        for con_id, con_info in self.concepts.items():
            con_info['name'] = ' '.join(con_info['name'].split())
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        # get all attribute keys and predicates
        self.attribute_keys = set()
        self.predicates = set()
        self.key_type = {}
        
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                self.attribute_keys.add(attr_info['key'])
                self.key_type[attr_info['key']] = attr_info['value']['type']
                for qk in attr_info['qualifiers']:
                    self.attribute_keys.add(qk)
                    for qv in attr_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                self.predicates.add(rel_info['predicate'])
                for qk in rel_info['qualifiers']:
                    self.attribute_keys.add(qk)
                    for qv in rel_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        self.attribute_keys = list(self.attribute_keys)
        self.predicates = list(self.predicates)
        # Note: key_type is one of string/quantity/date, but date means the key may have values of type year
        self.key_type = { k:v if v!='year' else 'date' for k,v in self.key_type.items() }

        # parse values into ValueClass object
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                attr_info['value'] = self._parse_value(attr_info['value'])
                for qk, qvs in attr_info['qualifiers'].items():
                    attr_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk, qvs in rel_info['qualifiers'].items():
                    rel_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]

        # add has instance
        for ent_id, ent_info in self.entities.items():
            instance_ids = self.get_all_concepts(ent_id)
            for id in instance_ids:
                if id in self.entities:
                    if 'hasInstance ' in self.entities[id]:
                        self.entities[id]['hasInstance'].add(ent_id)
                    else:
                        self.entities[id]['hasInstance'] = set([ent_id])
                        
                if id in self.concepts:
                    if 'hasInstance' in self.concepts[id]:
                        self.concepts[id]['hasInstance'].add(ent_id)
                    else:
                        self.concepts[id]['hasInstance'] = set([ent_id])

        for ent_id, ent_info in self.concepts.items():
            instance_ids = self.get_all_concepts(ent_id)
            for id in instance_ids:
                if id in self.entities:
                    if 'hasInstance' in self.entities[id]:
                        self.entities[id]['hasInstance'].add(ent_id)
                    else:
                        self.entities[id]['hasInstance'] = set([ent_id])
                        
                if id in self.concepts:
                    if 'hasInstance' in self.concepts[id]:
                        self.concepts[id]['hasInstance'].add(ent_id)
                    else:
                        self.concepts[id]['hasInstance'] = set([ent_id])
                        
    def _parse_value(self, value):
        if value['type'] == 'date':
            x = value['value']
            p1, p2 = x.find('/'), x.rfind('/')
            y, m, d = int(x[:p1]), int(x[p1+1:p2]), int(x[p2+1:])
            result = ValueClass('date', date(y, m, d))
        elif value['type'] == 'year':
            result = ValueClass('year', value['value'])
        elif value['type'] == 'string':
            result = ValueClass('string', value['value'])
        elif value['type'] == 'quantity':
            result = ValueClass('quantity', value['value'], value['unit'])
        else:
            raise Exception('unsupported value type')
        return result

    def get_direct_concepts(self, ent_id):
        """
        return the direct concept id of given entity/concept
        """
        if ent_id in self.entities:
            return self.entities[ent_id]['instanceOf']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['instanceOf']
        else:
            return []

    def get_subclass(self, ent_id):
        """
        return the subclass id of given entity/concept
        """
        if ent_id in self.entities and 'hasInstance' in self.entities[ent_id]:
            return list(self.entities[ent_id]['hasInstance'])
        elif ent_id in self.concepts and 'hasInstance' in self.concepts[ent_id]:
            return list(self.concepts[ent_id]['hasInstance'])
        else:
            return []

    def get_all_concepts(self, ent_id):
        """
        return a concept id list
        """
        ancestors = []
        q = Queue()
        for c in self.get_direct_concepts(ent_id):
            q.put(c)
        while not q.empty():
            con_id = q.get()
            ancestors.append(con_id)
            for c in self.concepts[con_id]['instanceOf']:
                q.put(c)

        return ancestors

    def get_name(self, ent_id):
        if ent_id in self.entities:
            return self.entities[ent_id]['name']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['name']
        else:
            return ent_id

    def is_concept(self, ent_id):
        return ent_id in self.concepts

    def get_attribute_facts(self, ent_id, key=None, unit=None):
        if key:
            facts = []
            for attr_info in self.entities[ent_id]['attributes']:
                if attr_info['key'] == key:
                    if unit:
                        if attr_info['value'].unit == unit:
                            facts.append(attr_info)
                    else:
                        facts.append(attr_info)
        else:
            facts = self.entities[ent_id]['attributes']
        facts = [(f['key'], f['value'], f['qualifiers']) for f in facts]
        return facts

    def get_relation_facts(self, ent_id):
        facts = self.entities[ent_id]['relations']
        facts = [(f['predicate'], f['object'], f['direction'], f['qualifiers']) for f in facts]
        return facts
#---------------------------------------- above from KQA_pro ------------------------------------------

# load KG
print("loading KQA_pro KG...")
kb_path = '/raid_sdb/tsy/KG-Agent/data/kb.json'
KG = DataForSPARQL(kb_path)

# build mappings
global name_to_id_map, id_to_name_map, entity_name_set, concept_name_set, attr_name_lst, rel_name_lst, attr_map
attr_map = {}
name_to_id_map = {}
id_to_name_map = {}
attr_map = {}
entity_name_set = set()
concept_name_set = set()
attr_name_lst = KG.attribute_keys
rel_name_lst = KG.predicates

def init_KG_map():
    for id in KG.entities:
        name = KG.get_name(id)
        entity_name_set.add(name)
        if name in name_to_id_map:
            name_to_id_map[name].append(id)
        else:
            name_to_id_map[name] = [id]

        if id in id_to_name_map:
            id_to_name_map[id].append(name)
        else:
            id_to_name_map[id] = [name]

    for id in KG.concepts:
        name = KG.get_name(id)
        concept_name_set.add(name)
        if name in name_to_id_map:
            name_to_id_map[name].append(id)
        else:
            name_to_id_map[name] = [id]

        if id in id_to_name_map:
            id_to_name_map[id].append(name)
        else:
            id_to_name_map[id] = [name]

init_KG_map()

def convert_name_to_id(name, env=None):
    if name in name_to_id_map:
        ids =  name_to_id_map[name]
        if len(ids) > 1:   # one name to multi ids
            ids_lst = []
            for id in ids:
                typ_ids = KG.get_all_concepts(id)
                if len(typ_ids) > 0:
                    typ = KG.get_name(typ_ids[0])
                    ids_lst.append([id, f"{typ}:{name}"])
                else:
                    ids_lst.append([id, f"{name}"])
            return ids_lst
        else:
            return ids[0]
    else:
        return name
    
def convert_id_to_name(id):
    if id in id_to_name_map:
        return id_to_name_map[id][0]
    else:
        return id

def get_name_by_id(wid, env=None):
    return convert_id_to_name(wid)

def get_id_by_name(ent):
    ent = str(ent)
    qid = ent
    if ": " in ent:
        ent_lst = ent.split(": ")
        if len(ent_lst) == 2 and "Q" in ent_lst[0]:    # Qid: name
            qid, name = ent_lst
        elif len(ent_lst) == 3 and "ent_set" in ent_lst[0] and "Q" in ent_lst[1]:    # ent_set_id: Qid: name
            ent_set_id, qid, name = ent_lst
        else:
            qid = convert_name_to_id(ent_lst[-1])
    else:
        qid = convert_name_to_id(ent)           # name
    
    if isinstance(qid, list):
        qid = qid[0]

    return qid

def filter_entities_by_constrain(ents, constrain):
    all_ents = set()
    all_trips = set()
    if isinstance(constrain, list):
        constrain = constrain[0]
    if constrain.startswith("Q"):
        constrain = convert_id_to_name(constrain)
    constrain = constrain.lower()
    ent_set = set()
    if isinstance(ents, str):
        ents = re.findall(r"Q\d+", ents)
    elif len(ents) and isinstance(ents[0], str):
        ents = [get_id_by_name(ent) for ent in ents]
    elif len(ents) and isinstance(ents[0], tuple):
        ents = [x[0] for x in ents]

    for ent in ents:
        if isinstance(ent, list):
            for e in ent:
                ent_set.add(e)
        else:
            ent_set.add(ent)

    for ent in ent_set:
        h = ent
        str_h = KG.get_name(h).lower()
        if h in KG.entities:
            # if constrain in relations
            rels = KG.get_relation_facts(ent)
            for rel in rels:        
                r,t,d,q = rel
                srt_r = str(r).lower()
                str_t = KG.get_name(t).lower()
                if isinstance(t, ValueClass):
                    t = str(t.value)
                if (constrain in srt_r 
                    or constrain in str_t 
                    or constrain in str_h):
                    if d == 'forward':
                        all_trips.add((h,r,t))
                        all_ents.add(t)
                    else:
                        all_trips.add((t,r,h))
                        all_ents.add(t)
                if q:
                    for k,v in q.items():
                        if (constrain in str_t 
                            or constrain in k.lower() 
                            or constrain in str(v[0].value).lower()):
                            if d == 'forward':
                                all_trips.add((h, r, f"[{t}| {k}| {str(v[0].value)}]"))
                            else:
                                all_trips.add((f"[{t}| {k}| {str(v[0].value)}]", r, h))
                            all_ents.add(t)
            # if constrain in attributes
            attrs = KG.get_attribute_facts(ent)
            for rel in attrs:       
                r,t,q = rel
                str_r = str(r).lower()
                if isinstance(t, ValueClass):
                    t = str(t.value)
                str_t = KG.get_name(t).lower()
                if (constrain in str_r 
                    or constrain in str_t 
                    or constrain in str_h):
                    all_trips.add((h,r,t))
                    all_ents.add(t)
                    if q:
                        for k,v in q.items():
                            if (constrain in str_t 
                                or constrain in k 
                                or constrain in str(v[0].value).lower()):
                                all_trips.add((h, r, f"[{t}| {k}| {str(v[0].value)}]"))
                                all_ents.add(t)

        # if constrain in instance
        instances = KG.get_all_concepts(ent) + KG.get_subclass(ent)
        for t in instances:         
            if constrain in 'instance of' or constrain in KG.get_name(t).lower():
                all_trips.add((h,'instance of',t))
                all_ents.add(t)

    return list(all_trips), list(all_ents)

def get_all_relations(ents):
    if isinstance(ents, str):
        ents = [ents]
    all_rels = set()
    for qid in ents:
        if qid in KG.entities:
            rels = KG.get_relation_facts(qid)
            attrs = KG.get_attribute_facts(qid)
            all_rels.update(set(x[0] for x in rels))
            all_rels.update(set(x[0] for x in attrs))
        # if qid in attr_map:                             # find tmp attributes
        #     all_rels.update(set(attr_map[qid]))

        instance = KG.get_direct_concepts(qid)
        instances = KG.get_all_concepts(qid) if not instance else instance
        if instances:
            all_rels.add("instance of")

        subclass = KG.get_subclass(qid)
        if subclass:
            all_rels.add("instance of")
            
    return list(all_rels)
    
def get_all_entities(ents, query_rels, constrain=None, expand_attr=True):
    if isinstance(ents, str):
        ents = [ents]
    if isinstance(query_rels, str):
        query_rels = [query_rels]

    all_ents = set()
    all_trips = set()
    for ent in ents:
        for query_rel in query_rels:
            h = ent
            if h in KG.entities:
                rels = KG.get_relation_facts(ent) if query_rel in rel_name_lst else []
                for rel in rels:
                    r,t,d,q = rel
                    if r != query_rel:
                        continue
                    if d == 'forward':
                        all_trips.add((h,r,t))
                        all_ents.add(t)
                        if q:
                            for k,v in q.items():
                                if expand_attr:
                                    all_trips.add((h, r, f"[{t}| {k}| {str(v[0].value)}]"))
                    else:
                        all_trips.add((t,r,h))
                        all_ents.add(t)
                        if q:
                            for k,v in q.items():
                                if expand_attr:
                                    all_trips.add((f"[{t}| {k}| {str(v[0].value)}]", r, h))


                attrs = KG.get_attribute_facts(ent) if query_rel in attr_name_lst else []
                for rel in attrs:
                    r,t,q = rel
                    if r != query_rel:
                        continue
                    if isinstance(t, ValueClass):
                        t = str(t.value)
                    all_trips.add((h,r,t))
                    all_ents.add(t)
                    if q:
                        for k,v in q.items():
                            # if t not in attr_map:
                            #     attr_map[t] = {}
                            # attr_map[t][k] = ((str(v[0].value),(h, r)))
                            if expand_attr:
                                all_trips.add((h, r, f"[{t}, {k}, {str(v[0].value)}]"))
            if h in attr_map:
                for r, t in attr_map[h].items():
                    if r == query_rel:
                        all_trips.add(h, r, t[0])
            if query_rel == 'instance of':
                instances = KG.get_all_concepts(ent)
                for t in instances:
                    all_trips.add((h,'instance of',t))
                    all_ents.add(t)
                #reverse edge of 'instance of'
                subclass = KG.get_subclass(ent)
                for t in subclass:
                    all_trips.add((t,'instance of',h))
                    all_ents.add(t)
    if constrain:
        _, filtered_ents = filter_entities_by_constrain(list(all_ents), constrain)
        if filtered_ents:
            all_ents = filtered_ents
            tmp = []
            for trip in all_trips:
                h,r,t = trip
                if h in all_ents or t in all_ents:
                    tmp.append(trip)
            all_trips = tmp
            
    return list(all_trips), list(all_ents)

def filter_by_function(ents, func, select_ent_set=None, env=None):
    func_name, func_input = func
    ent_ids = set([x[0] for x in ents])
    filtered_ents = set()
    new_trips_in_filter = set()
    operator_map = {"=": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y}

    if func_name == "FilterbyType":
        func_input = func_input.replace("(", "").replace(")", "").replace("'","").replace('"',"")
        func_input = convert_id_to_name(func_input)
        if func_input in concept_name_set:
            typ_name = func_input
        else:
            typ_name = cal_sim_score(func_input, list(concept_name_set), model=env.agent.emb, topk=1)[0]
        for ent in ents:
            qid, name = ent
            typs = get_types(qid)
            typ_name_lst = []
            for typ in typs:
                typ_nams = id_to_name_map[typ]
                for name in typ_nams:
                    typ_name_lst.append(name)

            if typ_name in typ_name_lst:
                filtered_ents.add(ent)

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
                V = "ANY"
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
        V = get_id_by_name(V)
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
        if len(rels) > 0:
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
                    trips, _ = get_all_entities(qid, K)
                    for trip in trips:
                        parse_trip(trip)
        else:                                            # h <OP> <V>
            for ent in ents:
                eid, name = ent
                parse_trip((eid, K, eid))

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
            ent1 = [get_id_by_name(ent1)]
        if 'ent_set' in ent2:
            ent2_ids = set(re.findall(r"ent_set_\d+(?:-\d+)*", str(ent2)))
            ent2 = set()
            for idx in ent2_ids:
                ent_set = env.state.set_to_ent_map[idx]
                for ent in ent_set:
                    ent2.add(ent[0])
        else:
            ent2 = [get_id_by_name(ent2)]
        
        for e1 in ent1:
            for e2 in ent2:
                out_trips, out_ents = filter_entities_by_constrain(e1, e2)
                filtered_ents.update(out_ents)
                new_trips_in_filter.update(out_trips)

    if func_name == "FilterbyRelation":
        qr = "".join(re.findall(r"(?<=\().+(?=,)", str(func_input), re.I)).replace("'","").replace('"',"").strip()
        qr = cal_sim_score(qr, rel_name_lst, model=env.agent.emb, topk=1)[0] if qr not in rel_name_lst else qr

        qe = "".join(re.findall(r"(?<=,).+(?=\))", str(func_input), re.I)).replace("'","").replace('"',"").strip()
        if not qe.startswith("Q"):
            qe = cal_sim_score(qe, list(entity_name_set), model=env.agent.emb, topk=1)[0] if qe not in entity_name_set else qe
            qe = get_id_by_name(qe)
        
        for ent in ents:
            qid, name = ent
            tmp_trips, _ = get_all_entities(qid, qr)
            for trip in tmp_trips:
                h,r,t = trip
                if h in qe or t in qe:
                    filtered_ents.add(ent)
                    new_trips_in_filter.add(trip)

    if func_name == "FilterbyStr":
        func_input = func_input.replace("(", "").replace(")", "").replace("'","").replace('"',"")
        out_trips, out_ents = filter_entities_by_constrain(ents, func_input)        # h -> any_r -> t == <str>
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

def get_types(entity_id: str):
    typs = KG.get_all_concepts(entity_id)
    if len(typs) == 0:
        if entity_id in KG.concepts:
            typs = [entity_id]
    
    return typs


def ParseAttributeValue(value):
    if isinstance(value, list):
        value = value[0]
        
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
