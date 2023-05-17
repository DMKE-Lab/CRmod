import json
import time
import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from rule_learning import rules_statistics
from score_functions import score_12


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--rules", "-r", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default=1, type=int, nargs="+")
parser.add_argument("--window", "-w", default=-1, type=int)
parser.add_argument("--top_k", default=20, type=int)
parser.add_argument("--num_processes", "-p", default=1, type=int)
parser.add_argument('--debug', dest='debug', type=bool, default=True)
# data property
parser.add_argument('--datadir', dest='datadir', type=str, default="./datasets/icews" )
# model architecture
parser.add_argument('--model', dest='model', type=str, default='ANALOGY')
parser.add_argument('--dim', dest='dim', type=int, default=8)
parser.add_argument('--loss_type', dest='loss_type', type=str, default='ANALOGY')
parser.add_argument('--axiom_probability', dest='axiom_probability', type=float, default=0.8)
# optimization
parser.add_argument('--lr', dest='lr', type=float, default=0.01)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1024)
parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
parser.add_argument('--regularizer_type', dest='regularizer_type', type=str, default='L2')
parser.add_argument('--regularizer_weight', dest='regularizer_weight', type=float, default=1e-3)
# experiment
parser.add_argument('--train', dest='train', type=bool, default=True)
parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=10)
parser.add_argument('--update_axiom_per', dest='update_axiom_per', type=int, default=2)
parser.add_argument('--DEBUG', dest='DEBUG', type=bool, default=True)
parser.add_argument('--neg_samples', dest='neg_samples', type=int, default=1)
parser.add_argument('--axiom_weight', dest='axiom_weight', type=float, default=0.1)
parser.add_argument('--device', dest='device', type=str, default='/cpu')
parser.add_argument('--triple_generator', dest='triple_generator', type=int, default=4)
parser.add_argument('--axiom_generator', dest='axiom_generator', type=int, default=4)
parser.add_argument('--save_per', dest='save_per', type=int, default=50)
parser.add_argument('--save_dir', dest='save_dir', type=str, default='./save/AE')
parser.add_argument('--init_bound', dest='init_bound', type=float, default=1e-2)
parser.add_argument('--load_dir', dest='load_dir', type=str, default=None)
parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=None)
parser.add_argument('--delay_lr_epoch', dest='delay_lr_epoch', type=int, default=None)
parser.add_argument('--max_entailment', dest='max_entailment', type=int, default=1000)
parser.add_argument('--inject_triple_percent', dest='inject_triple_percent', type=float, default=-1.0)

# evaluation
parser.add_argument('--num_test', dest='num_test', type=int, default=100)
parser.add_argument('--test', dest='test', type=bool, default=False)
parser.add_argument('--test_per_iter', dest='test_per_iter', type=int, default=1)
parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=10)
parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
top_k = parsed["top_k"]
num_processes = parsed["num_processes"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths

dataset_dir = "../data/" + dataset + "/"
dir_path = "../output/" + dataset + "/"
data = Grapher(dataset_dir)
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
rules_dict = json.load(open(dir_path + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
print("Rules statistics:")
rules_statistics(rules_dict)
rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
)
print("Rules statistics after pruning:")
rules_statistics(rules_dict)
learn_edges = store_edges(data.train_idx)

score_func = score_12
# It is possible to specify a list of list of arguments for tuning
args = [[0.1, 0.5]]
self.entity_embeddings = tf.get_variable('ent_embed', [self.data.num_entity, self.dim],
												 initializer=tf.random_uniform_initializer(minval=-bound,
																						   maxval=bound,
																						   seed=123))
		self.relation_embeddings = tf.get_variable('rel_embed', [self.data.num_relation, self.dim],
												   initializer=tf.random_uniform_initializer(minval=-bound,
																							 maxval=bound,
																							 seed=124))
		self.time_embeddings = tf.get_variable('tim_embed', [self.data.num_time, self.dim],
												   initializer=tf.random_uniform_initializer(minval=-bound,
																							 maxval=bound,
																							 seed=124))
		self.variables = [self.entity_embeddings, self.relation_embeddings]

		# embedding lookup for embeddings
		# [None, 3, 20]
		self.pos_hrt = self.embedding_lookup_triples(self.pos_triples, 1)
		self.neg_hrt = self.embedding_lookup_triples(self.neg_triples, 1)

		# normalize the entity embedding
		entity_embeddings_norm = self.entity_embeddings

		self.pos_h_embedding = tf.nn.embedding_lookup(entity_embeddings_norm, self.pos_triples[:, 0])
		self.pos_t_embedding = tf.nn.embedding_lookup(entity_embeddings_norm, self.pos_triples[:, 2])
		self.pos_r_embedding = tf.nn.embedding_lookup(entity_embeddings_norm, self.pos_triples[:, 1])

		self.neg_h_embedding = tf.nn.embedding_lookup(entity_embeddings_norm, self.neg_triples[:, 0])
		self.neg_t_embedding = tf.nn.embedding_lookup(entity_embeddings_norm, self.neg_triples[:, 2])
		self.neg_r_embedding = tf.nn.embedding_lookup(entity_embeddings_norm, self.neg_triples[:, 1])


	def _build_axiom_probability_input(self):
		with tf.device('/cpu'):
			self.reflexive_pool = tf.placeholder(tf.int32, [None, 1])
			self.symmetric_pool = tf.placeholder(tf.int32, [None, 1])
			self.transitive_pool = tf.placeholder(tf.int32, [None, 1])
			self.inverse_pool = tf.placeholder(tf.int32, [None, 2])
			self.subproperty_pool = tf.placeholder(tf.int32, [None,2])
			self.equivalent_pool = tf.placeholder(tf.int32, [None,2])
			self.inferenceChain_pool = tf.placeholder(tf.int32, [None,3])
		# look up the embeddings for each axiom in axiom pool
		# axiom_embed: [?, arity, dim]
		self.reflexive_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.reflexive_pool)
		self.symmetric_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.symmetric_pool)
		self.transitive_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.transitive_pool)
		self.inverse_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.inverse_pool)
		self.subproperty_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.subproperty_pool)
		self.equivalent_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.equivalent_pool)
		self.inferencechain_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.inferenceChain_pool)
		self.identity = tf.expand_dims(tf.concat((tf.ones(self.dim-self.dim/4),tf.zeros(self.dim/4)),0),0)

	def _build_test_input(self):
		with tf.device('/cpu'):
			# input test triples
			self.input_test_triples = tf.placeholder(tf.int32, [None, 3])
			# look up for embeddings of input triples
			# test_embed including embeddings for h,r,t
			self.test_embed = self.embedding_lookup_triples(self.input_test_triples, 1)


	def embedding_lookup_triples(self, triples,num_triples):
		output = None
		for i in range(num_triples):
			start_slice = i*3
			# hrt: [?, 3, dim]
			hrt = self.embedding_lookup_triple(triples, start_slice)
			if i == 0:
				output = hrt
			else:
				output = tf.concat([output, hrt], 1)
		return output

	def embedding_lookup_triple(self, triples, start_slice):
		# h,r,t: [?, 1, dim]
		# normalize the entity embeddings
		#entity_embedding_norm = tf.nn.l2_normalize(self.entity_embeddings, dim=1)
		entity_embedding_norm = self.entity_embeddings
		h = tf.expand_dims(tf.nn.embedding_lookup(entity_embedding_norm, triples[:, start_slice + 0]), 1)
		t = tf.expand_dims(tf.nn.embedding_lookup(entity_embedding_norm, triples[:, start_slice + 2]), 1)
		r = tf.expand_dims(tf.nn.embedding_lookup(self.relation_embeddings, triples[:, start_slice + 1]), 1)
		# outputembeddings: [?, 3, dim]
		outputembeddings = tf.concat([h, r, t], 1)
		return outputembeddings

	def triple_score(self, triples):
		# triples: [None, 3, dim]
		# h,r,t: [None, 1, dim] -> [None, dim]
		# [100 + 50(x) + 50(y)]
		# x, y
		# -y x
		h,r,t = tf.split(triples, [1,1,1], 1)
		h = tf.squeeze(h,1)
		r = tf.squeeze(r,1)
		t = tf.squeeze(t,1)
		if self.model == 'ANALOGY':
			# h_scalar: [None, dim/2]
			# h_x, h_y: [None, dim/4]
			h_scalar, h_x ,h_y = self.split_embedding(h)
			r_scalar, r_x, r_y = self.split_embedding(r)
			t_scalar, t_x, t_y = self.split_embedding(t)
			# score_scalar: [None]
			score_scalar = tf.reduce_sum(h_scalar * r_scalar * t_scalar, axis=1)
			# score_block: [None]
			score_block = tf.reduce_sum(h_x * r_x * t_x
										+ h_y * r_x * t_y
										+ h_x * r_y * t_y
										- h_y * r_y * t_x, axis=1)

			# score: [None]
			score = score_scalar + score_block

			return score

	def split_embedding(self, embedding):
		# embedding: [None, dim]
		assert self.dim % 4 == 0
		num_scalar = self.dim // 2
		num_block = self.dim // 4
		embedding_scalar = embedding[:, 0:num_scalar]
		embedding_x = embedding[:, num_scalar:-num_block]
		embedding_y = embedding[:, -num_block:]
		return embedding_scalar, embedding_x, embedding_y

	def axiom_loss(self, score, confidence, type):
		if type == 1:
			pi = score[0]
		else:
			pi_b = score[0]
			if type==2:
				pi_a = score[1]
			elif type==3:
				pi_a = score[1]*score[2]
			else:
				raise NotImplementedError
			pi = pi_a * pi_b - pi_a +1
		loss = tf.reduce_mean(tf.maximum(0.0, confidence-pi))
		return loss

	def axiom_loss_triple(self, input_score, confidence):
		score = input_score[0]
		loss = tf.reduce_mean(tf.clip_by_value(-confidence * tf.log(score), 1e-32, 1.0))
		return loss

	# calculate the similrity between two matrices
	# head: [?, dim]
	# tail: [?, dim] or [1,dim]
	def sim(self, head=None, tail=None, arity=None):
		if arity == 1:
			A_scalar, A_x, A_y = self.split_embedding(head)
		elif arity == 2:
			M1_scalar, M1_x, M1_y = self.split_embedding(head[0])
			M2_scalar, M2_x, M2_y = self.split_embedding(head[1])
			A_scalar= M1_scalar * M2_scalar
			A_x = M1_x*M2_x - M1_y*M2_y
			A_y = M1_x*M2_y + M1_y*M2_x
		elif arity==3:
			M1_scalar, M1_x, M1_y = self.split_embedding(head[0])
			M2_scalar, M2_x, M2_y = self.split_embedding(head[1])
			M3_scalar, M3_x, M3_y = self.split_embedding(head[2])
			M1M2_scalar = M1_scalar * M2_scalar
			M1M2_x = M1_x * M2_x - M1_y * M2_y
			M1M2_y = M1_x * M2_y + M1_y * M2_x
			A_scalar = M1M2_scalar * M3_scalar
			A_x = M1M2_x * M3_x - M1M2_y * M3_y
			A_y = M1M2_x * M3_y + M1M2_y * M3_x
		else:
			raise NotImplemented
		B_scala, B_x, B_y = self.split_embedding(tail)

		similarity = tf.concat([(A_scalar - B_scala)**2, (A_x - B_x)**2, (A_x - B_x)**2, (A_y - B_y)**2, (A_y - B_y)**2 ], axis=1)
		similarity = tf.sqrt(tf.reduce_sum(similarity, axis=1))

		#recale the probability
		probability = (tf.reduce_max(similarity)-similarity)/(tf.reduce_max(similarity)-tf.reduce_min(similarity))

		return probability

	# generate a probality for each axiom in axiom pool
	def run_axiom_probability(self, sess, data):
		if len(data.axiompool_reflexive) != 0: reflexive_prob = sess.run(self.reflexive_probability, {self.reflexive_pool: data.axiompool_reflexive})
		else: reflexive_prob = []


		if len(data.axiompool_symmetric) != 0: symmetric_prob = sess.run(self.symmetric_probability, {self.symmetric_pool: data.axiompool_symmetric})
		else: symmetric_prob = []

		if len(data.axiompool_transitive) != 0: transitive_prob = sess.run(self.transitive_probability, {self.transitive_pool: data.axiompool_transitive})
		else: transitive_prob = []

		if len(data.axiompool_inverse) != 0: inverse_prob = sess.run(self.inverse_probability, {self.inverse_pool: data.axiompool_inverse})
		else: inverse_prob = []

		if len(data.axiompool_subproperty) != 0: subproperty_prob = sess.run(self.subproperty_probability, {self.subproperty_pool: data.axiompool_subproperty})
		else: subproperty_prob = []

		if len(data.axiompool_equivalent) != 0: equivalent_prob = sess.run(self.equivalent_probability, {self.equivalent_pool: data.axiompool_equivalent})
		else: equivalent_prob = []

		if len(data.axiompool_inferencechain1) != 0:
			inferencechain1_prob = sess.run(self.inferenceChain1_probability,
										   {self.inferenceChain_pool: data.axiompool_inferencechain1})
		else:
			inferencechain1_prob = []

		if len(data.axiompool_inferencechain2) != 0:
			inferencechain2_prob = sess.run(self.inferenceChain2_probability,
										   {self.inferenceChain_pool: data.axiompool_inferencechain2})
		else:
			inferencechain2_prob = []

		if len(data.axiompool_inferencechain3) != 0:
			inferencechain3_prob = sess.run(self.inferenceChain3_probability,
										   {self.inferenceChain_pool: data.axiompool_inferencechain3})
		else:
			inferencechain3_prob = []

		if len(data.axiompool_inferencechain4) != 0:
			inferencechain4_prob = sess.run(self.inferenceChain4_probability,
										   {self.inferenceChain_pool: data.axiompool_inferencechain4})
		else:
			inferencechain4_prob = []

		output = [reflexive_prob, symmetric_prob, transitive_prob, inverse_prob,
				  subproperty_prob,equivalent_prob,inferencechain1_prob, inferencechain2_prob,
				  inferencechain3_prob, inferencechain4_prob]
		return output

	def run_train_batch(self, sess, feed):
		_, loss, loss_reg, score_pos, score_neg, score_pos_sig, score_neg_sig, score_neg_sig2, grads_and_vars_clip,\
			pos_embedding, neg_embeddding \
			= sess.run([self.optimizer_step, self.loss,self.loss_regularizer,
					  self.pos_score, self.neg_score,
					  self.pos_score_sig, self.neg_score_sig, self.neg_score_sig2,self.grads_and_vars ,
					  self.pos_hrt, self.neg_hrt,
					  ], feed_dict=feed)