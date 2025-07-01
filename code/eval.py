import pickle
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from agent import EntityLinker


class BM25Retrieve:
    def __init__(self, filepath="Freebase/virtuoso-opensource/database/FilterFreebase") -> None:
        self.name_to_ids = parallel_process_file(
            filepath,
            ConstructName2id,
            10,
        )
        self.all_fns = list(self.name_to_ids.keys())
        self.tokenized_all_fns = [fn.split() for fn in self.all_fns]
        print(self.tokenized_all_fns)
        self.bm25_all_fns = BM25Okapi(self.tokenized_all_fns)
        
            
def fuzz_name2id(name, bm25_name2id, n=1):
    mids = None
    if name.lower() not in bm25_name2id.name_to_ids:
        tokenized_query = name.lower().split()
        fn = bm25_name2id.bm25_all_fns.get_top_n(tokenized_query, bm25_name2id.all_fns, n=n)[0]
    else:
        fn = name
    if fn.lower() in bm25_name2id.name_to_ids:
        mids = bm25_name2id.name_to_ids[fn.lower()]
    if mids:
        return mids
    else:
        return []
    

def cal_h1(gen_ans, true_ans, entity_linker, mid_to_qid=None):
    ans_mid = []
    if isinstance(gen_ans, list):
        gen_ans = " ; ".join(gen_ans)
    gen_ans = gen_ans.lower().strip()
    # try match at text-level
    for ta in true_ans:
        if ta['text'] and ta['text'] != "":
            ta_text = ta['text'].strip().lower()
            if ta_text in gen_ans:              # BUG: number e.g. 1,2,3..
                return True
            # for ga in gen_ans.split(" "):
            #     if ga in ta_text or ta_text in ga:
            #         return True
                
        if ta['kb_id'] and ta['kb_id'] != "":
            ta_id = ta['kb_id'].strip().lower()
            if ta_id in gen_ans:
                return True
            # for ga in gen_ans.split(" "):
            #     if ga in ta_id or ta_id in ga:
            #         return True
                
    # try match at mid-level
    if isinstance(entity_linker, EntityLinker):
        try:
            mids = entity_linker.extract_entities(str(gen_ans))  # mids:Tuple(mid, name, score)
        except:
            mids = []
        for mid in mids:
            ans_mid.append(mid[0])
        if mid_to_qid is not None:
            for mid in ans_mid:
                mid = "/" + mid.replace(".","/")
                if mid in mid_to_qid['freebase_id'].values:
                    qid = mid_to_qid[mid_to_qid['freebase_id'] == mid]['wikidata_id'].values[0]
                    ans_mid.append(qid)
    else:
        ans_mid = []
    
    return len(set(ans_mid) & set([x['kb_id'] for x in true_ans])) > 0


entity_linker = None
def eval_metrics(res, df, args):
    global entity_linker

    if len(res) != len(df):
        print(f"NOTE: There are {len(df)} samples in total, but only {len(res)} results are obtained")
        df = df[:len(res)]
        
    metrics = {"avg_h1":[],"max_h1":[],"logs":[]}
    # load bm25 entity_linker
    # with open(f"{args.base_path}/data/bm25.pkl", "rb") as f:
    #     print("loading bm25_name2id for fuzzy match...")
    #     bm25_name2id = pickle.load(f)
    
    # load GrailQA entity_linker
    if args.kg == 'freebase':
        if not entity_linker:
            entity_linker = EntityLinker(args)
        mid_to_qid = None
    elif args.kg == 'wikidata':
        if not entity_linker:
            entity_linker = EntityLinker(args)
        # https://huggingface.co/datasets/kdm-daiict/freebase-wikidata-mapping
        mid_to_qid = pd.read_csv(f"{args.base_path}/data/fb_wiki_mapping.tsv",sep='\t')
    else:
        assert "not available entity_linker for current KG"

    for i in trange(len(df)):
        # get true answers
        if 'QA' in args.task:
            true_ans = eval(df.iloc[i]['answers'])
        elif "entity" in args.task:
            true_ans = [{"kb_id":x, "text":None} for x in eval(df.iloc[i]['entities'])]

        # get generated results
        eval_res = {'idx':df.iloc[i]['idx'], 'question':df.iloc[i]['question']}
        h1_lst = []
        gen_ans = str(res[i]).strip()
        if "[ANS]" in gen_ans:
            gen_ans = gen_ans.split("[ANS]")
        else:
            gen_ans = [gen_ans]
        # calculate metrics on each turn of each sample
        for idx, ga in enumerate(gen_ans):
            ga = str(ga).lower().strip()
            if ga == 'error' or ga == 'none':
                continue
            h1 = int(cal_h1(ga, true_ans, entity_linker, mid_to_qid))
            eval_res[f'ans_{idx}'] = ga
            eval_res[f'h1_{idx}'] = h1
            h1_lst.append(h1)
        if len(h1_lst) == 0:
            continue
        # calculate metrics on each sample
        metrics['avg_h1'].append(np.mean(h1_lst))
        metrics['random_h1'].append(float(h1_lst[0]))
        metrics['max_h1'].append(float(np.max(h1_lst)))
        metrics['logs'].append(eval_res)
        args.res_log.loc[i, 'h1'] = np.max(h1_lst)
    print(f"\nresults of {len(metrics['max_h1'])}/{len(df)} samples:")
    tmp_dict = {}
    # calculate metrics on all dataset
    for k, v in metrics.items():
        if k == 'logs':
            continue
        print(f"{k}: {np.mean(v)}")
        tmp_dict[f'overall_{k}'] = np.mean(v)
    metrics.update(tmp_dict)

    return metrics