from collections import Counter
import json
import os
from pathlib import Path
import libem.prepare.datasets as ds
from libem.prepare.datasets import (abt_buy, amazon_google, beer, dblp_acm, 
                                    dblp_scholar, fodors_zagats, itunes_amazon, 
                                    walmart_amazon)

dataset_list = {
    'abt-buy': abt_buy,
    'amazon-google': amazon_google,
    'beer': beer,
    'dblp-acm': dblp_acm,
    'dblp-scholar': dblp_scholar,
    'fodors-zagats': fodors_zagats,
    'itunes-amazon': itunes_amazon,
    'walmart-amazon': walmart_amazon,
    }
    
def read_all(ds):
    l = list(ds.read_train())
    l.extend(list(ds.read_test()))
    try:
        l.extend(list(ds.read_valid()))
    except:
        pass
    return l
    
datasets = {
    name: read_all(d)
    for name, d in dataset_list.items()
}
for name in datasets.keys():
    cluster_id = 0
    clusters, added = [], set()
    
    for i, p1 in enumerate(datasets[name]):
        if p1['label'] == 0 or i in added:
            continue
        
        added.add(i)
        cluster = [{'cluster_id': cluster_id, **p1['left']}, 
                   {'cluster_id': cluster_id, **p1['right']}]
        
        for j, p2 in enumerate(datasets[name]):
            if p2['label'] == 1 and p2['left'] == p1['left'] and i != j:
                added.add(j)
                cluster.append({'cluster_id': cluster_id, **p2['right']})
                
            elif p2['label'] == 1 and p2['right'] == p1['right'] and i != j:
                added.add(j)
                cluster.append({'cluster_id': cluster_id, **p2['left']})

        # remove duplicate records
        new_cluster = []
        for c in cluster:
            if c not in new_cluster:
                new_cluster.append(c)
        
        clusters.extend(new_cluster)
        cluster_id += 1

    # create output folder + file
    folder = os.path.join(os.path.join(ds.LIBEM_SAMPLE_DATA_PATH, 'clustering'), name)
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(folder, 'test.ndjson'), 'w') as out:
        for c in clusters:
            out.write(json.dumps(c) + '\n')
    
    # write README
    file = os.path.join(folder, 'README.md')
    with open(file, 'w') as f:
        f.write("Processed version of the dataset originating from BatchER: https://github.com/fmh1art/batcher.")
    