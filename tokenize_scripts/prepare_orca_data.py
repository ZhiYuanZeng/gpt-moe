import os
import json
root_dir = "/mnt/inspurfs/feizhaoye/train_data/oppen_2stage_deliver_shuffle/orca/"
output_dir = "./data/openorca"
for f in os.listdir(root_dir):
    p = root_dir + '/' + f +'/' + 'merged-0.jsonl'
    
    with open(p, 'r') as fi, open(output_dir+'/'+f+'.jsonl', 'w') as fo:
        for l in fi:
            d = json.loads(l)
            text = d['prompt'] + '[[OUTPUT]]\n' + d['output']
            
            fo.write(json.dumps({"text": text}) +'\n')
