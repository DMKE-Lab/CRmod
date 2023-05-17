# CRmod: Context-aware Rule-guided Reasoning over Temporal Knowledge Graph

<h3>Installation </h3>

Install Tensorflow (>= 1.1.0)
Python 3.x (tested on Python 3.6)

<h3> How to run </h3>

python apply.py -d dataset -r 101221144026_r[1,2,3]_n200_exp_s12_rules.json -l 1 2 3 -w 0 -p 8
python evaluate.py -d dataset -c 101221144026_r[1,2,3]_n200_exp_s12_cands_r[1,2,3]_w0_score_12[0.1,0.5].json

<h3> Parameters </h3>

`--dataset`, `-d`: str. Dataset name.

`--rule_lengths`, `-l`: int. Length(s) of rules that will be learned.

`--num_walks`, `-n`: int. Number of walks that will be extracted during rule learning.

`--rules`, `-r`: str. Name of the rules file.

`--window`, `-w`: int. Size of the time window before the query timestamp for rule application.

`--top_k`: int. Minimum number of candidates. The rule application stops for a query if this number is reached.
