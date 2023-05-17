# CRmod: Context-aware Rule-guided Reasoning over Temporal Knowledge Graph

<h3>Installation </h3>

  Tensorflow (>= 1.1.0)  
  Python 3.x (tested on Python 3.6)  

<h3> How to run </h3>
After installing the requirements,run the following command to reproduce results for CRmod:    
  python apply.py -d dataset -r r[1,2,3]_n200_exp_s12_rules.json -l 1 2 3 -w 0 -p 8   
  python evaluate.py -d dataset -c r[1,2,3]_n200_exp_s12_cands_r[1,2,3]_w0_score_12[0.1,0.5].json    

<h3> Parameters </h3>

`--dataset`, `-d`: Dataset name.

`--rule_lengths`, `-l`:  Lengths of rules that will be learned.

`--num_walks`, `-n`:  Number of walks that will be extracted during rule learning.

`--rules`, `-r`:  Name of the rules file.

