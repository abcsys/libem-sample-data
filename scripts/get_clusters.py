from collections import Counter
import json
import os
from pathlib import Path
import libem.prepare.datasets as ds
from libem.prepare.datasets import (abt_buy, amazon_google, beer, dblp_acm, 
                                    dblp_scholar, fodors_zagats, itunes_amazon, 
                                    walmart_amazon)

home_dir = os.path.join(ds.LIBEM_SAMPLE_DATA_PATH, 'clustering')

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

cluster_metadata = {}

for name in datasets.keys():
    cluster_id = 0
    clusters, cluster_sizes, added = [], [], set()
    left_set = set([json.dumps(d['left']) for d in datasets[name]])
    right_set = set([json.dumps(d['right']) for d in datasets[name]])
    
    for i, p1 in enumerate(datasets[name]):
        if p1['label'] == 0 or i in added:
            continue
        
        cluster = [{'cluster_id': cluster_id, **p1['left']}, 
                   {'cluster_id': cluster_id, **p1['right']}]
        added.add(i)
        try:
            left_set.remove(json.dumps(p1['left']))
        except KeyError:
            pass
        try:
            right_set.remove(json.dumps(p1['right']))
        except KeyError:
            pass
        
        # find all records with label 1 that match either left or right
        for j, p2 in enumerate(datasets[name]):
            if p2['label'] == 1 and p2['left'] == p1['left'] and i != j:
                cluster.append({'cluster_id': cluster_id, **p2['right']})
                added.add(j)
                try:
                    right_set.remove(json.dumps(p2['right']))
                except KeyError:
                    pass
                
            elif p2['label'] == 1 and p2['right'] == p1['right'] and i != j:
                cluster.append({'cluster_id': cluster_id, **p2['left']})
                added.add(j)
                try:
                    left_set.remove(json.dumps(p2['left']))
                except KeyError:
                    pass

        # remove duplicate records and add to clusters
        new_cluster = []
        for c in cluster:
            if c not in new_cluster:
                new_cluster.append(c)
        
        clusters.extend(new_cluster)
        cluster_sizes.append(len(new_cluster))
        cluster_id += 1
        
    # add everything still in left_set and right_set as cluster size of 1
    for i in left_set:
        clusters.append({'cluster_id': cluster_id, **json.loads(i)})
        cluster_sizes.append(1)
        cluster_id += 1
    for i in right_set:
        clusters.append({'cluster_id': cluster_id, **json.loads(i)})
        cluster_sizes.append(1)
        cluster_id += 1
    
    cluster_metadata[name] = {
        'size': cluster_id,
        'dist': dict(sorted(Counter(cluster_sizes).items()))
    }

    # create output folder + file
    folder = os.path.join(home_dir, name)
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(folder, 'test.ndjson'), 'w') as out:
        for c in clusters:
            out.write(json.dumps(c) + '\n')
    
    # write README
    file = os.path.join(folder, 'README.md')
    with open(file, 'w') as f:
        f.write(f"Processed version of the {name} dataset originating from BatchER: https://github.com/fmh1art/batcher.")

# write root README
file = os.path.join(home_dir, 'README.md')
with open(file, 'w') as f:
    f.write("## Clustering Distributions\n")
    f.write("|dataset name|total number of clusters|number of items in each cluster: number of such clusters|\n")
    f.write("|---|:-:|---|\n")
    for name in datasets.keys():
        f.write(f"|{name}|{cluster_metadata[name]['size']}|{str(cluster_metadata[name]['dist'])[1:-1]}|\n")