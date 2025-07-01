# Entity linking
# from https://github.com/dki-lab/GrailQA

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import pickle
import sys
sys.path.append("/data/tsy/datasets/entity_linker") 

from bert_entity_linker import BertEntityLinker
import surface_index_memory
from aaqu_entity_linker import IdentifiedEntity
from aaqu_util import normalize_entity_name, remove_prefixes_from_name, remove_suffixes_from_name

from sklearn.metrics import precision_score, recall_score, f1_score


def load_entity_linker():
    print("loading surface_index_memory...")
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
                "/data/tsy/datasets/entity_linker/data/entity_list_file_freebase_complete_all_mention", 
                "/data/tsy/datasets/entity_linker/data/surface_map_file_freebase_complete_all_mention",
                "/data/tsy/datasets/entity_linker/freebase_complete_all_mention")


    print("loading pretrained model...")
    entity_linker = BertEntityLinker(surface_index, model_path="/data/tsy/datasets/entity_linker/BERT_NER/trained_ner_model/", 
                                    device="cuda:0")
    return entity_linker

# identified_entities = entity_linker.identify_entities("safety and tolerance of intermittent intravenous and oral zidovudine therapy in human immunodeficiency virus-infected pediatric patients. pediatric zidovudine phase i study group. is a medical trial for what?")

def link_entities(text, entity_linker):
    identified_entities = entity_linker.identify_entities(text)

    entities = set()
    for res in identified_entities:
        entities.add((res.entity.id, res.entity.name, res.entity.score))

    return entities


if __name__ == "__main__":
    df = pd.read_csv("/data/tsy/KG-Agent/data/cwq_v4.csv")
    entity_linker = load_entity_linker()

    results = []
    metrics = {"R":[], "P":[], "F1":[]}

    for i in trange(len(df)):
        question = df.iloc[i]['question']
        true_entities_set = set(eval(df.iloc[i]['entities']))
        entities = link_entities(question, entity_linker)
        link_entities_set = set([x[0] for x in entities])
        results.append(entities)
        
        R = len(link_entities_set & true_entities_set) / len(link_entities_set) if len(link_entities_set) else 0
        P = len(link_entities_set & true_entities_set) / len(true_entities_set) if len(true_entities_set) else 0
        F1 = 2*P*R / (P+R) if P+R != 0 else 0

        metrics["R"].append(R)
        metrics["P"].append(P)
        metrics["F1"].append(F1)

    tmp_dict = {}
    for k,v in metrics.items():
        print(f"avg {k} is: {np.mean(v)}")
        tmp_dict[f"avg_{k}"] = np.mean(v)
    metrics.update(tmp_dict)
    results.append(metrics)

    with open('cwq_entity_link.pkl', 'wb') as file:
        pickle.dump(results, file)