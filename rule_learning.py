import json
import numpy as np
import pandas as pd
from collections import Counter

from temporal_walk import store_edges


def filter_rules(rules_dict, min_conf, min_body_supp, rule_lengths):
    """
    Filter for rules with a minimum confidence, minimum body support, and
    specified rule lengths.

    Parameters.
        rules_dict (dict): rules
        min_conf (float): minimum confidence value
        min_body_supp (int): minimum body support value
        rule_lengths (list): rule lengths

    Returns:
        new_rules_dict (dict): filtered rules
    """

    new_rules_dict = dict()
    for k in rules_dict:
        new_rules_dict[k] = []
        for rule in rules_dict[k]:
            cond = (
                (rule["conf"] >= min_conf)
                and (rule["body_supp"] >= min_body_supp)
                and (len(rule["body_rels"]) in rule_lengths)
            )
            if cond:
                new_rules_dict[k].append(rule)

    return new_rules_dict


def get_window_edges(all_data, test_query_ts, learn_edges, window=-1):
    """
    Get the edges in the data (for rule application) that occur in the specified time window.
    If window is 0, all edges before the test query timestamp are included.
    If window is -1, the edges on which the rules are learned are used.
    If window is an integer n > 0, all edges within n timestamps before the test query
    timestamp are included.

    Parameters:
        all_data (np.ndarray): complete dataset (train/valid/test)
        test_query_ts (np.ndarray): test query timestamp
        learn_edges (dict): edges on which the rules are learned
        window (int): time window used for rule application

    Returns:
        window_edges (dict): edges in the window for rule application
    """

    if window > 0:
        mask = (all_data[:, 3] < test_query_ts) * (
            all_data[:, 3] >= test_query_ts - window
        )
        window_edges = store_edges(all_data[mask])
    elif window == 0:
        mask = all_data[:, 3] < test_query_ts
        window_edges = store_edges(all_data[mask])
    elif window == -1:
        window_edges = learn_edges

    return window_edges


def match_body_relations(rule, edges, test_query_sub):
    """
    Find edges that could constitute walks (starting from the test query subject)
    that match the rule.
    First, find edges whose subject match the query subject and the relation matches
    the first relation in the rule body. Then, find edges whose subjects match the
    current targets and the relation the next relation in the rule body.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject

    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [
            np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
        ]  # [sub, obj, ts]
        cur_targets = np.array(list(set(walk_edges[0][:, 1])))

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(
                    np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
                )  # [sub, obj, ts]
                cur_targets = np.array(list(set(walk_edges[i][:, 1])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]

    return walk_edges


def match_body_relations_complete(rule, edges, test_query_sub):
    """
    Find edges that could constitute walks (starting from the test query subject)
    that match the rule.
    First, find edges whose subject match the query subject and the relation matches
    the first relation in the rule body. Then, find edges whose subjects match the
    current targets and the relation the next relation in the rule body.

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject

    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [new_edges]
        cur_targets = np.array(list(set(walk_edges[0][:, 2])))

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(new_edges)
                cur_targets = np.array(list(set(walk_edges[i][:, 2])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]

    return walk_edges


def get_walks(rule, walk_edges):
    """
    Get walks for a given rule. Take the time constraints into account.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """

    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_" + str(0), "entity_" + str(1), "timestamp_" + str(0)],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    if not rule["var_constraints"]:
        del df["entity_" + str(0)]
    df_edges.append(df)
    df = df[0:0]  # Memory efficiency

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_" + str(i), "entity_" + str(i + 1), "timestamp_" + str(i)],
            dtype=np.uint16,
        )  # Change type if necessary
        df_edges.append(df)
        df = df[0:0]

    rule_walks = df_edges[0]
    df_edges[0] = df_edges[0][0:0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        rule_walks = rule_walks[
            rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
        ]
        if not rule["var_constraints"]:
            del rule_walks["entity_" + str(i)]
        df_edges[i] = df_edges[i][0:0]

    for i in range(1, len(rule["body_rels"])):
        del rule_walks["timestamp_" + str(i)]

    return rule_walks


def get_walks_complete(rule, walk_edges):
    """
    Get complete walks for a given rule. Take the time constraints into account.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """

    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=[
            "entity_" + str(0),
            "relation_" + str(0),
            "entity_" + str(1),
            "timestamp_" + str(0),
        ],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    df_edges.append(df)

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=[
                "entity_" + str(i),
                "relation_" + str(i),
                "entity_" + str(i + 1),
                "timestamp_" + str(i),
            ],
            dtype=np.uint16,
        )  # Change type if necessary
        df_edges.append(df)

    rule_walks = df_edges[0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        rule_walks = rule_walks[
            rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
        ]

    return rule_walks




def save_candidates(
    rules_file, dir_path, all_candidates, rule_lengths, window, score_func_str
):
    """
    Save the candidates.

    Parameters:
        rules_file (str): name of rules file
        dir_path (str): path to output directory
        all_candidates (dict): candidates for all test queries
        rule_lengths (list): rule lengths
        window (int): time window used for rule application
        score_func_str (str): scoring function

    Returns:
        None
    """

    all_candidates = {int(k): v for k, v in all_candidates.items()}
    for k in all_candidates:
        all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}
    filename = "{0}_cands_r{1}_w{2}_{3}.json".format(
        rules_file[:-11], rule_lengths, window, score_func_str
    )
    filename = filename.replace(" ", "")
    with open(dir_path + filename, "w", encoding="utf-8") as fout:
        json.dump(all_candidates, fout)


def verbalize_walk(walk, data):
    """
    Verbalize walk from rule application.

    Parameters:
        walk (pandas.core.series.Series): walk that matches the rule body from get_walks
        data (grapher.Grapher): graph data

    Returns:
        walk_str (str): verbalized walk
    """

    l = len(walk) // 3
    walk = walk.values.tolist()

    walk_str = data.id2entity[walk[0]] + "\t"
    for j in range(l):
        walk_str += data.id2relation[walk[3 * j + 1]] + "\t"
        walk_str += data.id2entity[walk[3 * j + 2]] + "\t"
        walk_str += data.id2ts[walk[3 * j + 3]] + "\t"

    return walk_str[:-1]
