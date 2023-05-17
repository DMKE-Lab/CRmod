import json
import numpy as np


class Grapher(object):
    def __init__(self, dataset_dir):
        """
        Store information about the graph (train/valid/test set).
        Add corresponding inverse quadruples to the data.

        Parameters:
            dataset_dir (str): path to the graph dataset directory

        Returns:
            None
        """

        self.dataset_dir = dataset_dir
        self.entity2id = json.load(open(dataset_dir + "entity2id.json"))
        self.relation2id_old = json.load(open(dataset_dir + "relation2id.json"))
        self.relation2id = self.relation2id_old.copy()
        counter = len(self.relation2id_old)
        for relation in self.relation2id_old:
            self.relation2id["_" + relation] = counter  # Inverse relation
            counter += 1
        self.ts2id = json.load(open(dataset_dir + "ts2id.json"))
        self.id2entity = dict([(v, k) for k, v in self.entity2id.items()])
        self.id2relation = dict([(v, k) for k, v in self.relation2id.items()])
        self.id2ts = dict([(v, k) for k, v in self.ts2id.items()])

        self.inv_relation_id = dict()
        num_relations = len(self.relation2id_old)
        for i in range(num_relations):
            self.inv_relation_id[i] = i + num_relations
        for i in range(num_relations, num_relations * 2):
            self.inv_relation_id[i] = i % num_relations

        self.train_idx = self.create_store("train.txt")
        self.valid_idx = self.create_store("valid.txt")
        self.test_idx = self.create_store("test.txt")
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        print("Grapher initialized.")



    def map_to_idx(self, quads):
        """
        Map quadruples to their indices.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form [subject, relation, object, timestamp].

        Returns:
            quads (np.ndarray): indices of quadruples
        """

        subs = [self.entity2id[x[0]] for x in quads]
        rels = [self.relation2id[x[1]] for x in quads]
        objs = [self.entity2id[x[2]] for x in quads]
        tss = [self.ts2id[x[3]] for x in quads]
        quads = np.column_stack((subs, rels, objs, tss))

        return quads

