import numpy as np


def score1(rule, c=0):
    def train(self):
        if self.option.load_dir is not None:
            self.saver.restore(self.sess, self.option.load_dir)
            print('load model from %s' % (self.option.load_dir), self.outdatafile)
            self.epoch = self.option.load_epoch

        while (self.epoch <= self.option.max_epoch):

            print('\n')
            print('epoch:', self.epoch)
            logging.info('epoch: %d' % (self.epoch))

            if self.epoch == 0:
                ent_embed, rel_embed, tim_embed = self.sess.run(
                    [self.model.entity_embeddings, self.model.relation_embeddings, self.model.time_embeddings])
                print('init ent embedding:', ent_embed[:10])
                print('init rel embedding:', rel_embed[:10])
                print('init tim embedding:', tim_embed[:10])

            # train for one epoch
            loss_epoch, loss_epoch_reg = self.one_epoch_train(epoch=self.epoch)

            print('[epoch:%d] --loss:%.4f  --reg loss:%.4f' % (self.epoch, loss_epoch, loss_epoch_reg))

            # save model
            if self.epoch % self.option.save_per == 0:
                model_path = self.saver.save(self.sess,
                                             self.option.save_dir,
                                             global_step=self.epoch)
                print('Model saved at %s' % (model_path))

            # test sample triples
            if self.epoch != 0 and (self.epoch % self.option.test_per_iter == 0 or self.epoch == self.option.max_epoch):
                if self.option.num_test != -1:
                    test_num = self.option.num_test
                    valid_num = self.option.num_test
                    train_num = self.option.num_test
                else:
                    test_num = self.data.num_test_triples
                    valid_num = self.data.num_valid_triples
                    train_num = self.data.num_train_triples

                self.test('test', num_test=test_num)

            if self.epoch % self.option.update_axiom_per == 0 and self.epoch != 0:
                # axioms include probability for each axiom in axiom pool
                # order: ref, sym, tran, inver, sub, equi, inferC
                # update_axioms:
                #			1) calculate probability for each axiom in axiom pool with current embeddings
                #			2) update the valid_axioms
                axioms_probability = self.update_axiom()
                self.data.update_train_triples(epoch=self.epoch, update_per=self.option.update_axiom_per)

                logging.info('axiom_probability: %s' % (axioms_probability))

            self.epoch += 1

    def test_only(self, load_model, axiom=False):
        self.saver.restore(self.sess, load_model)
        print('load model from %s' % (self.option.load_dir))
        if axiom:
            axioms_probability = self.update_axiom()

            self.test('test', num_test=self.option.num_test, output_rank=True, axiom=axiom)
        else:
            self.test('test', num_test=self.data.num_test_triples, output_rank=True)

    def test(self, dataset, num_test, output_rank=False, axiom=False):
        # test_triples: [num_test*num_entity*2, 3]
        # for each test triple, replace head and tail entity
        test_batch = round(num_test / self.option.test_batch_size)
        # scores_head(tail): [num_entity, 0]
        scores_head = np.asarray([]).reshape([-1, self.data.num_entity])
        scores_tail = np.asarray([]).reshape([-1, self.data.num_entity])
        scores_org = []

        for batch in range(test_batch):
            print('test %d/%d' % (batch, test_batch), end='\r')
            self.test_triples, self.test_triples_org = self.generate_test_triples_batch(dataset, batch, num_test)
            feed = {self.model.input_test_triples: self.test_triples}
            scores = self.model.run_test(self.sess, feed)
            scores_org += list(self.model.run_test(self.sess, {self.model.input_test_triples: self.test_triples_org}))
            # score_reshape: [num_test * 2, num_entity]

            scores_reshape = scores.reshape([-1, self.data.num_entity])
            # score_head(tail): [num_test, num_entity]
            head = scores_reshape[:int(len(scores_reshape) / 2), :]
            tail = scores_reshape[int(len(scores_reshape) / 2):, :]

            scores_head = np.concatenate((scores_head, head), axis=0)
            scores_tail = np.concatenate((scores_tail, tail), axis=0)

        if not axiom:
            MR, MR_h, MR_t, MRR, MRR_h, MRR_t, \
            H1, H1_h, H1_t, H3, H3_h, H3_t, H10, H10_h, H10_t, \
            FMR, FMR_h, FMR_t, FMRR, FMRR_h, FMRR_t, \
            FH1, FH1_h, FH1_t, FH3, FH3_h, FH3_t, \
            FH10, FH10_h, FH10_t = self.rank_test_score(scores_head, scores_tail, dataset, scores_org, num_test,
                                                        output_rank=output_rank)
        else:
            MR, MR_h, MR_t, MRR, MRR_h, MRR_t, \
            H1, H1_h, H1_t, H3, H3_h, H3_t, H10, H10_h, H10_t, \
            FMR, FMR_h, FMR_t, FMRR, FMRR_h, FMRR_t, \
            FH1, FH1_h, FH1_t, FH3, FH3_h, FH3_t, \
            FH10, FH10_h, FH10_t = self.rank_test_score_with_axiom(scores_head, scores_tail, dataset, scores_org,
                                                                   num_test,
                                                                   output_rank=output_rank)
        with open('./output/icews_min/scores.txt', 'a') as score_file:
            score_file.write("epoch: {0:7.4f}".format(self.epoch))
            score_file.write("/n")
            score_file.write("MR: {0:7.4f} --MR_h: {0:7.4f} --MR_t: {0:7.4f} ".format(MR, MR_h, MR_t))
            score_file.write("/n")
            score_file.write("MRR: {0:7.4f} --MRR_h: {0:7.4f} --MRR_t: {0:7.4f} ".format(MRR, MRR_h, MRR_t))
            score_file.write("/n")
            score_file.write("H1: {0:7.4f} --H1_h: {0:7.4f} --H1_t: {0:7.4f} ".format(H1, H1_h, H1_t))
            score_file.write("/n")
            score_file.write("H3: {0:7.4f} --H3_h: {0:7.4f} --H3_t: {0:7.4f} ".format(H3, H3_h, H3_t))
            score_file.write("/n")
            score_file.write("H10: {0:7.4f} --H10_h: {0:7.4f} --H10_t: {0:7.4f} ".format(H10, H10_h, H10_t))
            score_file.write("/n")
            score_file.write("FMR: {0:7.4f} --FMR_h: {0:7.4f} --FMR_t: {0:7.4f} ".format(FMR, FMR_h, FMR_t))
            score_file.write("/n")
            score_file.write("FMRR: {0:7.4f} --FMRR_h: {0:7.4f} --FMRR_t: {0:7.4f} ".format(FMRR, FMRR_h, FMRR_t))
            score_file.write("/n")
            score_file.write("FH1: {0:7.4f} --FH1_h: {0:7.4f} --FH1_t: {0:7.4f} ".format(FH1, FH1_h, FH1_t))
            score_file.write("/n")
            score_file.write("FH3: {0:7.4f} --FH3_h: {0:7.4f} --FH3_t: {0:7.4f} ".format(FH3, FH3_h, FH3_t))
            score_file.write("/n")
            score_file.write("FH10: {0:7.4f} --FH10_h: {0:7.4f} --FH10_t: {0:7.4f} ".format(FH10, FH10_h, FH10_t))
            score_file.write("/n")
            score_file.write("/n")

        print('[%s][epoch:%d] --MR:%.2f	--MR_h: %.2f	--MR_t:%.2f' % (dataset, self.epoch, MR, MR_h, MR_t))
        print('[%s][epoch:%d] --MRR:%.3f	--MRR_h: %.3f	--MRR_t:%.3f' % (dataset, self.epoch, MRR, MRR_h, MRR_t))
        print("[%s][epoch:%d] --H1:%.3f	--H1_h: %.3f 	--H1_t:%.3f" % (dataset, self.epoch, H1, H1_h, H1_t))
        print("[%s][epoch:%d] --H3:%.3f	--H3_h: %.3f 	--H3_t:%.3f" % (dataset, self.epoch, H3, H3_h, H3_t))
        print("[%s][epoch:%d] --H10:%.3f	--H10_h: %.3f 	--H10_t:%.3f" % (dataset, self.epoch, H10, H10_h, H10_t))

        print('[%s][epoch:%d] --FMR:%.2f	--FMR_h: %.2f	--FMR_t:%.2f' % (dataset, self.epoch, FMR, FMR_h, FMR_t))
        print('[%s][epoch:%d] --FMRR:%.3f	--FMRR_h: %.3f	--FMRR_t:%.3f' % (
        dataset, self.epoch, FMRR, FMRR_h, FMRR_t))
        print("[%s][epoch:%d] --FH1:%.3f	--FH1_h: %.3f 	--FH1_t:%.3f" % (dataset, self.epoch, FH1, FH1_h, FH1_t))
        print("[%s][epoch:%d] --FH3:%.3f	--FH3_h: %.3f 	--FH3_t:%.3f" % (dataset, self.epoch, FH3, FH3_h, FH3_t))
        print(
            "[%s][epoch:%d] --FH10:%.3f	--FH10_h: %.3f 	--FH10_t:%.3f" % (
            dataset, self.epoch, FH10, FH10_h, FH10_t))

    def update_axiom(self):
        time_s = time.time()
        axiom_pro = self.model.run_axiom_probability(self.sess, self.data)
        time_e = time.time()
        print('calculate axiom score:', time_e - time_s)
        with open('./save_axiom_prob/axiom_prob.pickle', 'wb') as f: pickle.dump(axiom_pro, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_axiom_prob/axiom_pools.pickle', 'wb') as f: pickle.dump(self.data.axiompool, f,
                                                                                  pickle.HIGHEST_PROTOCOL)
        self.data.update_valid_axioms(axiom_pro)
        return self.model.run_axiom_probability(self.sess, self.data)

    def generate_test_triples_batch(self, type, batch, num_test):
        start = min(num_test, batch * self.option.test_batch_size)
        end = min(start + self.option.test_batch_size, num_test)
        if type == 'test':
            test_triple_ids = self.data.test_ids[start:end]
        elif type == 'valid':
            test_triple_ids = self.data.valid_ids[start:end]
        elif type == 'train':
            test_triple_ids = self.data.train_ids[start:end]
        else:
            raise NotImplementedError
        test_triple_replace = self.replace_test_triple(test_triple_ids)

        return test_triple_replace, test_triple_ids

    def replace_test_triple(self, input_triples):
        input_triples = np.asarray(input_triples)
        replace_head_rt = input_triples[:, 1:]
        replace_tail_hr1 = input_triples[:, :2]
        replace_tail_hr2 = input_triples[:, 3:]
        # replace_tail_hr = np.concatenate((replace_tail_hr1, replace_tail_hr2),axis=1)

        replace_ids = np.asarray([i for i in range(self.data.num_entity)] * len(input_triples)).reshape([-1, 1])
        replace_head_repeat = np.repeat(replace_head_rt, self.data.num_entity, axis=0)
        replace_tail_repeat1 = np.repeat(replace_tail_hr1, self.data.num_entity, axis=0)
        replace_tail_repeat2 = np.repeat(replace_tail_hr2, self.data.num_entity, axis=0)

        replace_head = np.concatenate((replace_ids, replace_head_repeat), axis=1)
        replace_tail1 = np.concatenate((replace_tail_repeat1, replace_ids), axis=1)
        replace_tail = np.concatenate((replace_tail1, replace_tail_repeat2), axis=1)

        score = np.concatenate((replace_head, replace_tail), axis=0)
        return score
