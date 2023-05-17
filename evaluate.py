import json
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--candidates", "-c", default="", type=str)
parsed = vars(parser.parse_args())

def rank_test_score(self, score_head, score_tail, dataset, scores_org, num_test, output_rank=False, with_axiom=True):
    # head/tail_score: [num_test, num_entity]
    head_score = score_head.reshape(-1, self.data.num_entity)
    tail_score = score_tail.reshape(-1, self.data.num_entity)
    if dataset == 'valid':
        test_ids = np.asarray(self.data.valid_ids)[: num_test, :]
    elif dataset == 'test':
        test_ids = np.asarray(self.data.test_ids)[:num_test, :]
    elif dataset == 'train':
        test_ids = np.asarray(self.data.train_ids)[:num_test, :]
    else:
        raise NotImplementedError
    head_score_rank_id = np.argsort(head_score, axis=1)
    tail_score_rank_id = np.argsort(tail_score, axis=1)

    rank_h, rank_t, frank_h, frank_t = [[] for i in range(4)]

    num = 0
    filter_head = 0
    filter_tail = 0
    for triple, head_rank_id, tail_rank_id, head_s, tail_s in zip(test_ids, head_score_rank_id, tail_score_rank_id,
                                                                  head_score, tail_score):
        num += 1
        print('testing %d/%d' % (num, num_test), end='\r')
        h, r, t, tim = triple

        # rank without axiom
        rank_head = self.data.num_entity - np.where(head_rank_id == h)[0][0]
        rank_tail = self.data.num_entity - np.where(tail_rank_id == t)[0][0]
        rank_head_filter = rank_head
        rank_tail_filter = rank_tail
        for i in range(rank_head - 1):
            if head_rank_id[self.data.num_entity - 1 - i] in self.data.trtim_h_all[(t, r, tim)]:
                rank_head_filter -= 1
                filter_head += 1

        for i in range(rank_tail - 1):
            if tail_rank_id[self.data.num_entity - 1 - i] in self.data.hrtim_t_all[(h, r, tim)]:
                rank_tail_filter -= 1
                filter_tail += 1

        rank_h.append(rank_head)
        rank_t.append(rank_tail)
        frank_h.append(rank_head_filter)
        frank_t.append(rank_tail_filter)
        aa = np.shape(rank_h)
    print('\n')
    print('filter_head:', filter_head)
    print('filter_tail:', filter_tail)
    print('rank_head_filter:', rank_head_filter)
    print('rank_tail_filter:', rank_tail_filter)

    print('aa:', aa)

    rank_h, rank_t, frank_h, frank_t = map(lambda x: np.asarray(x), [rank_h, rank_t, frank_h, frank_t])

    MR_h, MR_t, FMR_h, FMR_t = map(lambda x: np.mean(x), [rank_h, rank_t, frank_h, frank_t])
    MRR_h, MRR_t, FMRR_h, FMRR_t = map(lambda x: np.mean(1.0 / x), [rank_h, rank_t, frank_h, frank_t])
    H1_h, H1_t, FH1_h, FH1_t = map(lambda x: np.mean(np.asarray(x <= 1, dtype=float)),
                                   [rank_h, rank_t, frank_h, frank_t])
    H3_h, H3_t, FH3_h, FH3_t = map(lambda x: np.mean(np.asarray(x <= 3, dtype=float)),
                                   [rank_h, rank_t, frank_h, frank_t])
    H10_h, H10_t, FH10_h, FH10_t = map(lambda x: np.mean(np.asarray(x <= 10, dtype=float)),
                                       [rank_h, rank_t, frank_h, frank_t])
    MR, FMR, MRR, FMRR, H1, FH1, H3, FH3, H10, FH10 = map(lambda x, y: (x + y) / 2.0,
                                                          [MR_h, FMR_h, MRR_h, FMRR_h, H1_h, FH1_h, H3_h, FH3_h, H10_h,
                                                           FH10_h],
                                                          [MR_t, FMR_t, MRR_t, FMRR_t, H1_t, FH1_t, H3_t, FH3_t, H10_t,
                                                           FH10_t])
    if output_rank:
        with open('./save_rank/rank_h_noaxiom.pickle', 'wb') as f: pickle.dump(rank_h, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_rank/rank_t_noaxiom.pickle', 'wb') as f: pickle.dump(rank_t, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_rank/frank_h_noaxiom.pickle', 'wb') as f: pickle.dump(frank_h, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_rank/frank_t_noaxiom.pickle', 'wb') as f: pickle.dump(frank_t, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_rank/test_ids_noaxiom.pickle', 'wb') as f: pickle.dump(test_ids, f, pickle.HIGHEST_PROTOCOL)

    return MR, MR_h, MR_t, \
           MRR, MRR_h, MRR_t, \
           H1, H1_h, H1_t, \
           H3, H3_h, H3_t, \
           H10, H10_h, H10_t, \
           FMR, FMR_h, FMR_t, \
           FMRR, FMRR_h, FMRR_t, \
           FH1, FH1_h, FH1_t, \
           FH3, FH3_h, FH3_t, \
           FH10, FH10_h, FH10_t


def rank_test_score_with_axiom(self, score_head, score_tail, dataset, scores_org, num_test, output_rank=False,
                               with_axiom=True):
    head_score = score_head.reshape(-1, self.data.num_entity)
    tail_score = score_tail.reshape(-1, self.data.num_entity)
    filter_head = 0
    filter_tail = 0

    if dataset == 'valid':
        test_ids = np.asarray(self.data.valid_ids)[: num_test, :]
    elif dataset == 'test':
        test_ids = np.asarray(self.data.test_ids)[:num_test, :]
    elif dataset == 'train':
        test_ids = np.asarray(self.data.train_ids)[:num_test, :]
    else:
        raise NotImplementedError

    rank_h, rank_t, frank_h, frank_t = [[] for i in range(4)]

    num = 0
    self.data._reset_valid_axiom_entailment()
    for triple, head_s, tail_s in zip(test_ids, head_score, tail_score):
        num += 1
        print('testing %d/%d' % (num, num_test), end='\r')
        h, r, t, tim = triple

        if h in self.data.infered_trtim_h[(t, r, tim)]:
            filter_head += 1
            rank_head = 1
            rank_head_filter = 1
        else:
            head_rank_id = np.argsort(head_s)
            rank_head = self.data.num_entity - np.where(head_rank_id == h)[0][0]
            rank_head_filter = rank_head
            for i in range(rank_head - 1):
                if head_rank_id[self.data.num_entity - 1 - i] in self.data.trtim_h_all[(t, r, tim)]:
                    rank_head_filter -= 1

        if t in self.data.infered_hrtim_t[(h, r, tim)]:
            filter_tail += 1
            rank_tail = 1
            rank_tail_filter = 1
        else:
            tail_rank_id = np.argsort(tail_s)
            rank_tail = self.data.num_entity - np.where(tail_rank_id == t)[0][0]
            rank_tail_filter = rank_tail
            for i in range(rank_tail - 1):
                if tail_rank_id[self.data.num_entity - 1 - i] in self.data.hrtim_t_all[(h, r, tim)]:
                    rank_tail_filter -= 1

        rank_h.append(rank_head)
        rank_t.append(rank_tail)
        frank_h.append(rank_head_filter)
        frank_t.append(rank_tail_filter)

    print('\n')
    print('filter_head:', filter_head)
    print('filter_tail:', filter_tail)
    print('rank_head_filter:', rank_head_filter)
    print('rank_tail_filter:', rank_tail_filter)
    rank_h, rank_t, frank_h, frank_t = map(lambda x: np.asarray(x), [rank_h, rank_t, frank_h, frank_t])

    MR_h, MR_t, FMR_h, FMR_t = map(lambda x: np.mean(x), [rank_h, rank_t, frank_h, frank_t])
    MRR_h, MRR_t, FMRR_h, FMRR_t = map(lambda x: np.mean(1.0 / x), [rank_h, rank_t, frank_h, frank_t])
    H1_h, H1_t, FH1_h, FH1_t = map(lambda x: np.mean(np.asarray(x <= 1, dtype=float)),
                                   [rank_h, rank_t, frank_h, frank_t])
    H3_h, H3_t, FH3_h, FH3_t = map(lambda x: np.mean(np.asarray(x <= 3, dtype=float)),
                                   [rank_h, rank_t, frank_h, frank_t])
    H10_h, H10_t, FH10_h, FH10_t = map(lambda x: np.mean(np.asarray(x <= 10, dtype=float)),
                                       [rank_h, rank_t, frank_h, frank_t])
    MR, FMR, MRR, FMRR, H1, FH1, H3, FH3, H10, FH10 = map(lambda x, y: (x + y) / 2.0,
                                                          [MR_h, FMR_h, MRR_h, FMRR_h, H1_h, FH1_h, H3_h, FH3_h, H10_h,
                                                           FH10_h],
                                                          [MR_t, FMR_t, MRR_t, FMRR_t, H1_t, FH1_t, H3_t, FH3_t, H10_t,
                                                           FH10_t])
    if output_rank:
        with open('./save_rank/rank_h.pickle', 'wb') as f: pickle.dump(rank_h, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_rank/rank_t.pickle', 'wb') as f: pickle.dump(rank_t, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_rank/frank_h.pickle', 'wb') as f: pickle.dump(frank_h, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_rank/frank_t.pickle', 'wb') as f: pickle.dump(frank_t, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_rank/test_ids.pickle', 'wb') as f: pickle.dump(test_ids, f, pickle.HIGHEST_PROTOCOL)

    return MR, MR_h, MR_t, \
           MRR, MRR_h, MRR_t, \
           H1, H1_h, H1_t, \
           H3, H3_h, H3_t, \
           H10, H10_h, H10_t, \
           FMR, FMR_h, FMR_t, \
           FMRR, FMRR_h, FMRR_t, \
           FH1, FH1_h, FH1_t, \
           FH3, FH3_h, FH3_t, \
           FH10, FH10_h, FH10_t


def check_infered(self, triple, head_score, tail_score):
    h, r, t, tim = triple
    head_score = list(head_score)
    tail_score = list(tail_score)
    head_id = [i for i in range(len(head_score))]
    tail_id = [i for i in range(len(tail_score))]
    assert len(head_id) == self.data.num_entity
    assert len(tail_id) == self.data.num_entity
    infer_id_h, infer_s_h, infer_id_t, infer_s_t = [[] for i in range(4)]
    left_id_h, left_s_h, left_id_t, left_s_t = [[] for i in range(4)]

    for i in range(len(head_score)):
        if i in self.data.infered_trtim_h[(t, r, tim)]:
            infer_id_h.append(head_id[i])
            infer_s_h.append(head_score[i])
        else:
            left_id_h.append(head_id[i])
            left_s_h.append(head_score[i])

        if i in self.data.infered_hrtim_t[(h, r, tim)]:
            infer_id_t.append(head_id[i])
            infer_s_t.append(head_score[i])
        else:
            left_id_t.append(head_id[i])
            left_s_t.append(head_score[i])

    return infer_id_h, infer_s_h, infer_id_t, infer_s_t, \
           left_id_h, left_s_h, left_id_t, left_s_t


def rank_score(self, triple, head_scores, tail_scores):
    h, r, t, tim = triple
    head_score_axiom = []
    head_score_left = []
    tail_score_axiom = []
    tail_score_left = []
    assert len(head_scores) == len(tail_scores)
    for i in range(len(head_scores)):
        # check axiom entailment for head prediction
        if (i, r, t, tim) in self.data.train_inject_triples:
            head_score_axiom.append(head_scores[i])
        else:
            head_score_left.append(head_scores[i])

        # check aixom entailment for tail prediction
        if (h, r, i, tim) in self.data.train_inject_triples:
            tail_score_axiom.append(tail_scores[i])
        else:
            tail_score_left.append(tail_scores[i])
    # sort the score
    head_score_axiom_rank = -np.sort(-np.asarray(head_score_axiom))
    head_score_left_rank = -np.sort(-np.asarray(head_score_left))
    tail_score_axiom_rank = -np.sort(-np.asarray(tail_score_axiom))
    tail_score_left_rank = -np.sort(-np.asarray(tail_score_left))

    head_score_rank = np.concatenate([head_score_axiom_rank, head_score_left_rank], axis=0)
    tail_score_rank = np.concatenate([tail_score_axiom_rank, tail_score_left_rank], axis=0)
    return head_score_rank, tail_score_rank


def one_epoch_train(self, epoch):
    self.data.reset(self.option.batch_size)
    learning_rate = self.learning_rate
    if self.option.delay_lr_epoch is not None and epoch > self.option.delay_lr_epoch:
        learning_rate = self.learning_rate / 10
    print('learning_rate', learning_rate)
    positive_triple_generator = self.data.generate_train_batch()

    # axiom_generator = self.data.generate_axiom_batch()

    # prepare the positive training tripels
    # each dat is a batch of training data
    for dat in positive_triple_generator:
        self.queue_raw_training_data.put(dat)
    print('raw training data is initialized')

    loss_epoch = 0.0
    loss_epoch_reg = 0.0
    for batch in range(self.data.num_batch_train):
        start = time.time()

        positive_ids_labels, negative_ids_labels = self.queue_training_data.get()

        positive = positive_ids_labels[:, :4]

        negative = negative_ids_labels[:, :4]
        positive_labels = np.reshape(positive_ids_labels[:, -1], [-1, 1])
        negative_labels = np.reshape(negative_ids_labels[:, -1], [-1, 1])

        feed = {self.model.pos_triples: positive,
                self.model.neg_triples: negative,
                self.model.pos_labels: positive_labels,
                self.model.neg_labels: negative_labels,
                self.model.learning_rate: learning_rate}
        loss_batch, loss_reg = self.model.run_train_batch(self.sess, feed)

        loss_epoch += loss_batch
        loss_epoch_reg += loss_reg

        if batch % 20 == 0:
            print('batch/num_batch: %d/%d, loss: %.6f, loss_reg: %.6f' % (
            batch, self.data.num_batch_train, loss_batch, loss_reg), end='\r')

    ent_embed, rel_embed, tim_embed = self.sess.run(
        [self.model.entity_embeddings, self.model.relation_embeddings, self.model.time_embeddings])

    return loss_epoch, loss_epoch_reg
