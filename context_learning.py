def __init__(self, sess, option, model, data, saver):
    self.sess = sess
    self.option = option
    self.model = model
    self.data = data
    self.saver = saver
    self.start = time.time()
    self.epoch = 0
    self.learning_rate = self.option.lr

    # set and init the training data generator
    self.queue_raw_training_data = JoinableQueue()
    self.queue_training_data = Queue()
    self.data_generators = list()
    for i in range(option.triple_generator):
        self.data_generators.append(Process(target=self.data.negative_triple_generator,
                                            args=(self.queue_raw_training_data, self.queue_training_data)))
        self.data_generators[-1].start()


def update_valid_axioms(self, input):
    # this function is used to select high probability axioms as valid axioms
    # and record their scores

    valid_axioms = [self._select_high_probability(list(prob), axiom) for prob, axiom in zip(input, self.axiompool)]

    self.valid_reflexive, self.valid_symmetric, self.valid_transitive, \
    self.valid_inverse1, self.valid_inverse2, self.valid_subproperty, self.valid_equivalent, \
    self.valid_inferencechain1, self.valid_inferencechain2, \
    self.valid_inferencechain3, self.valid_inferencechain4 = valid_axioms
    # update the batchsize of axioms and entailments
    self._reset_valid_axiom_entailment()

    '''
    logging.debug('the valid axioms after updated: %s'%(valid_axioms))
    logging.debug('the valid reflexive axioms: %s'%(self.valid_reflexive))
    logging.debug('the valid symmetric axioms: %s'%(self.valid_symmetric))
    logging.debug('the valid inverse axioms: %s'%(self.valid_inverse))
    logging.debug('the valid inferencevhain axioms: %s'%(self.valid_inferencechain))
    '''


def _select_high_probability(self, prob, axiom):
    # select the high probability axioms and recore their probabilities
    valid_axiom = [[(axiom[prob.index(p)], [p])] for p in prob if p > self.select_probability]
    with open('./save_rank/rank.txt', 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write(str(valid_axiom))

    return valid_axiom


def reset(self, batch_size):
    self.batch_size = batch_size
    self.train_start = 0
    self.valid_start = 0
    self.test_start = 0
    self.num_train_triples = len(self.train_ids_labels)
    self.num_batch_train = round(self.num_train_triples / self.batch_size)


def _reset_valid_axiom_entailment(self):
    self.infered_hrtim_t = defaultdict(set)
    self.infered_trtim_h = defaultdict(set)

    self.valid_reflexive2entailment, self.valid_reflexive_p = \
        self._valid_axiom2entailment(self.valid_reflexive, self.reflexive2entailment)

    self.valid_symmetric2entailment, self.valid_symmetric_p = \
        self._valid_axiom2entailment(self.valid_symmetric, self.symmetric2entailment)

    self.valid_transitive2entailment, self.valid_transitive_p = \
        self._valid_axiom2entailment(self.valid_transitive, self.transitive2entailment)

    self.valid_inverse21entailment, self.valid_inverse1_p = \
        self._valid_axiom2entailment(self.valid_inverse1, self.inverse21entailment)

    self.valid_inverse22entailment, self.valid_inverse2_p = \
        self._valid_axiom2entailment(self.valid_inverse2, self.inverse22entailment)

    self.valid_subproperty2entailment, self.valid_subproperty_p = \
        self._valid_axiom2entailment(self.valid_subproperty, self.subproperty2entailment)

    self.valid_equivalent2entailment, self.valid_equivalent_p = \
        self._valid_axiom2entailment(self.valid_equivalent, self.equivalent2entailment)

    self.valid_inferencechain12entailment, self.valid_inferencechain1_p = \
        self._valid_axiom2entailment(self.valid_inferencechain1, self.inferencechain12entailment)

    self.valid_inferencechain22entailment, self.valid_inferencechain2_p = \
        self._valid_axiom2entailment(self.valid_inferencechain2, self.inferencechain22entailment)

    self.valid_inferencechain32entailment, self.valid_inferencechain3_p = \
        self._valid_axiom2entailment(self.valid_inferencechain3, self.inferencechain32entailment)
    self.valid_inferencechain42entailment, self.valid_inferencechain4_p = \
        self._valid_axiom2entailment(self.valid_inferencechain4, self.inferencechain42entailment)


def _valid_axiom2entailment(self, valid_axiom, axiom2entailment):
    valid_axiom2entailment = []
    valid_axiom_p = []
    for axiom_p in valid_axiom:

        axiom = tuple(axiom_p[0][0])
        p = axiom_p[0][1]

        for entailment in axiom2entailment[axiom]:
            valid_axiom2entailment.append(entailment)
            valid_axiom_p.append(p)
            h, r, t, tim = entailment[-4:]
            self.infered_hrtim_t[(h, r, tim)].add(t)
            self.infered_trtim_h[(t, r, tim)].add(h)

    return valid_axiom2entailment, valid_axiom_p


def generate_train_batch(self):
    origin_triples = self.train_ids_labels
    inject_triples = self.train_ids_labels_inject

    inject_num = self.inject_triple_percent * len(self.train_ids_labels)
    if len(inject_triples) > inject_num and inject_num > 0:
        np.random.shuffle(inject_triples)
        inject_triples = inject_triples[:inject_num]

    # inject_triples = np.reshape([], [-1, 5])

    train_triples = np.concatenate([origin_triples, inject_triples], axis=0)

    self.num_train_triples = len(train_triples)
    self.num_batch_train = round(self.num_train_triples / self.batch_size)
    print('self.num_batch_train', self.num_batch_train)
    np.random.shuffle(train_triples)
    for i in range(self.num_batch_train):
        t1 = time.time()
        start = i * self.batch_size
        end = min(start + self.batch_size, self.num_train_triples)
        positive = train_triples[start:end]
        t2 = time.time()
        yield positive


def negative_triple_generator(self, input_postive_queue, output_queue):
    while True:
        dat = input_postive_queue.get()
        if dat.all() == None:
            break
        positive = np.asarray(dat)
        replace_h, replace_t = [np.random.randint(self.num_entity, size=len(positive) * self.neg_samples) for i in
                                range(2)]
        replace_r = np.random.randint(self.num_relation, size=len(positive) * self.neg_samples)
        replace_tim = np.random.randint(self.num_time, size=len(positive) * self.neg_samples)
        neg_h, neg_r, neg_t, neg_tim = [np.copy(np.repeat(positive, self.neg_samples, axis=0)) for i in range(4)]
        neg_h[:, 0] = replace_h
        neg_r[:, 1] = replace_r
        neg_t[:, 2] = replace_t
        neg_tim[:, 3] = replace_tim
        negative = np.concatenate((neg_h, neg_r, neg_t, neg_tim), axis=0)
        output_queue.put((positive, negative))


# add the new triples from axioms to training triple
def update_train_triples(self, epoch=0, update_per=10):
    reflexive_triples, symmetric_triples, transitive_triples, inverse1_triples, inverse2_triples, \
    equivalent_triples, subproperty_triples, inferencechain1_triples, \
    inferencechain2_triples, inferencechain3_triples, inferencechain4_triples = [np.reshape(np.asarray([]), [-1, 4]) for
                                                                                 i in range(self.axiom_types)]
    reflexive_p, symmetric_p, transitive_p, inverse1_p, inverse2_p, \
    equivalent_p, subproperty_p, inferencechain1_p, \
    inferencechain2_p, inferencechain3_p, inferencechain4_p = [np.reshape(np.asarray([]), [-1, 1]) for i in
                                                               range(self.axiom_types)]
    if epoch >= 20:
        print("len(self.valid_reflexive2entailment):", len(self.valid_reflexive2entailment))
        print("len(self.valid_symmetric2entailment):", len(self.valid_symmetric2entailment))
        print("len(self.valid_transitive2entailment)", len(self.valid_transitive2entailment))
        print("len(self.valid_inverse21entailment)", len(self.valid_inverse21entailment))
        print("len(self.valid_inverse22entailment)", len(self.valid_inverse22entailment))
        print("len(self.valid_equivalent2entailment)", len(self.valid_equivalent2entailment))
        print("len(self.valid_subproperty2entailment)", len(self.valid_subproperty2entailment))

        valid_reflexive2entailment, valid_symmetric2entailment, valid_transitive2entailment, \
        valid_inverse21entailment, valid_inverse22entailment, valid_equivalent2entailment, valid_subproperty2entailment, \
        valid_inferencechain12entailment, valid_inferencechain22entailment, \
        valid_inferencechain32entailment, valid_inferencechain42entailment = [[] for i in range(11)]

        if len(self.valid_reflexive2entailment) > 0:
            valid_reflexive2entailment = np.reshape(np.asarray(self.valid_reflexive2entailment), [-1, 4])
            reflexive_triples = np.asarray(valid_reflexive2entailment)[:, -4:]
            reflexive_p = np.reshape(np.asarray(self.valid_reflexive_p), [-1, 1])

        if len(self.valid_symmetric2entailment) > 0:
            valid_symmetric2entailment = np.reshape(np.asarray(self.valid_symmetric2entailment), [-1, 8])
            symmetric_triples = np.asarray(valid_symmetric2entailment)[:, -4:]
            symmetric_p = np.reshape(np.asarray(self.valid_symmetric_p), [-1, 1])

        if len(self.valid_transitive2entailment) > 0:
            valid_transitive2entailment = np.reshape(np.asarray(self.valid_transitive2entailment), [-1, 12])
            transitive_triples = np.asarray(valid_transitive2entailment)[:, -4:]
            transitive_p = np.reshape(np.asarray(self.valid_transitive_p), [-1, 1])

        if len(self.valid_inverse21entailment) > 0:
            valid_inverse21entailment = np.reshape(np.asarray(self.valid_inverse21entailment), [-1, 8])
            inverse1_triples = np.asarray(valid_inverse21entailment)[:, -4:]
            inverse1_p = np.reshape(np.asarray(self.valid_inverse1_p), [-1, 1])

        if len(self.valid_inverse22entailment) > 0:
            valid_inverse22entailment = np.reshape(np.asarray(self.valid_inverse22entailment), [-1, 8])
            inverse2_triples = np.asarray(valid_inverse22entailment)[:, -4:]
            inverse2_p = np.reshape(np.asarray(self.valid_inverse2_p), [-1, 1])

        if len(self.valid_equivalent2entailment) > 0:
            valid_equivalent2entailment = np.reshape(np.asarray(self.valid_equivalent2entailment), [-1, 8])
            equivalent_triples = np.asarray(valid_equivalent2entailment)[:, -4:]
            equivalent_p = np.reshape(np.asarray(self.valid_equivalent_p), [-1, 1])

        if len(self.valid_subproperty2entailment) > 0:
            valid_subproperty2entailment = np.reshape(np.asarray(self.valid_subproperty2entailment), [-1, 8])
            subproperty_triples = np.asarray(valid_subproperty2entailment)[:, -4:]
            subproperty_p = np.reshape(np.asarray(self.valid_subproperty_p), [-1, 1])

        if len(self.valid_inferencechain12entailment) > 0:
            valid_inferencechain12entailment = np.reshape(np.asarray(self.valid_inferencechain12entailment), [-1, 12])
            inferencechain1_triples = np.asarray(valid_inferencechain12entailment)[:, -4:]
            inferencechain1_p = np.reshape(np.asarray(self.valid_inferencechain1_p), [-1, 1])

        if len(self.valid_inferencechain22entailment) > 0:
            valid_inferencechain22entailment = np.reshape(np.asarray(self.valid_inferencechain22entailment), [-1, 12])
            inferencechain2_triples = np.asarray(valid_inferencechain22entailment)[:, -4:]
            inferencechain2_p = np.reshape(np.asarray(self.valid_inferencechain2_p), [-1, 1])

        if len(self.valid_inferencechain32entailment) > 0:
            valid_inferencechain32entailment = np.reshape(np.asarray(self.valid_inferencechain32entailment), [-1, 12])
            inferencechain3_triples = np.asarray(valid_inferencechain32entailment)[:, -4:]
            inferencechain3_p = np.reshape(np.asarray(self.valid_inferencechain3_p), [-1, 1])

        if len(self.valid_inferencechain42entailment) > 0:
            valid_inferencechain42entailment = np.reshape(np.asarray(self.valid_inferencechain42entailment), [-1, 12])
            inferencechain4_triples = np.asarray(valid_inferencechain42entailment)[:, -4:]
            inferencechain4_p = np.reshape(np.asarray(self.valid_inferencechain4_p), [-1, 1])

        # pickle.dump(self.reflexive_entailments, open(os.path.join(self.axiom_dir, 'reflexive_entailments'), 'wb'))
        # store all the injected triples
        entailment_all = (valid_reflexive2entailment, valid_symmetric2entailment, valid_transitive2entailment,
                          valid_inverse21entailment, valid_inverse22entailment, valid_equivalent2entailment,
                          valid_subproperty2entailment,
                          valid_inferencechain12entailment, valid_inferencechain22entailment,
                          valid_inferencechain32entailment, valid_inferencechain42entailment)
        pickle.dump(entailment_all, open(os.path.join(self.axiom_dir, 'valid_entailments.pickle'), 'wb'))

    train_inject_triples = np.concatenate(
        [reflexive_triples, symmetric_triples, transitive_triples, inverse1_triples, inverse2_triples,
         equivalent_triples, subproperty_triples, inferencechain1_triples,
         inferencechain2_triples, inferencechain3_triples, inferencechain4_triples],
        axis=0)

    train_inject_triples_p = np.concatenate([reflexive_p, symmetric_p, transitive_p, inverse1_p, inverse2_p,
                                             equivalent_p, subproperty_p, inferencechain1_p,
                                             inferencechain2_p, inferencechain3_p, inferencechain4_p],
                                            axis=0)

    self.train_inject_triples = train_inject_triples
    inject_labels = np.reshape(np.ones(len(train_inject_triples)), [-1, 1]) * self.axiom_weight * train_inject_triples_p
    train_inject_ids_labels = np.concatenate([train_inject_triples, inject_labels],
                                             axis=1)

    self.train_ids_labels_inject = train_inject_ids_labels





