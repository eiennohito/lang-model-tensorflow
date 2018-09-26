import tensorflow as tf
import numpy as np
import argparse
import os, sys
import time


class Vocabulary(object):
    def __init__(self, args):
        self.filename = args.vocab_file
        with open(self.filename, encoding='utf-8') as fin:
            header = fin.readline()  # ignore header
            self.lines = [x.split('\t')[0] for x in fin]
        self.str2id = tf.contrib.lookup.index_table_from_tensor(
            mapping=self.lines,
            default_value=1
        )
        self.id2str = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping=self.lines,
            default_value='<u>'
        )

        self.size = len(self.lines)


class InputPipeline(object):
    def make_boundaries(self, start, end, increase):
        value = int(start)
        result = [value]
        while value < end:
            value = int(value * increase)
            result.append(value)
        return result

    def make_batch_sizes(self, num_data, boundaries):
        result = []
        for x in boundaries:
            result.append(int(num_data / x) + 1)
        result.append(1)
        return result

    def _tfrecords(self):
        pass

    def _textfile(self):

        def parse_txt(line):
            line = tf.reshape(line, [1])
            words = tf.string_split(line, delimiter=" ").values
            words = self.vocab.str2id.lookup(words)
            words = tf.to_int32(words)
            words = tf.concat([words, tf.to_int32([3])], axis=0)
            words = tf.concat([tf.to_int32([2]), words], axis=0)

            return {
                'data': words,
                'length': tf.shape(words)[0]
            }

        inpt = self.input
        if inpt.endswith('.gz'):
            compression = 'GZIP'
        else:
            compression = ''

        pipe = tf.data.Dataset.list_files(inpt)

        pipe = pipe.repeat(self.args.epochs)
        pipe = pipe.shuffle(50)  # shuffle files

        def _read_file(fn):
            return tf.data.TextLineDataset(
                filenames=fn,
                buffer_size=4 * 1024 * 1024,
                compression_type=compression
            )

        pipe = pipe.apply(
            tf.contrib.data.parallel_interleave(
                _read_file,
                cycle_length=4,
                sloppy=True
            )
        )

        return pipe.map(
            map_func=parse_txt,
            num_parallel_calls=self.args.parallel_map
        )

    def __init__(self, args, input, batch_size, vocab):
        self.args = args
        self.vocab = vocab
        self.input = input
        self.batch_size = batch_size
        tf_records = args.tf_records
        if tf_records:
            pipe = self._tfrecords()
        else:
            pipe = self._textfile()

        pipe: tf.data.Dataset = pipe
        if args.shuffle_size > 0:
            pipe = pipe.shuffle(
                buffer_size=args.shuffle_size
            )

        boundaries = self.make_boundaries(8, 200, 1.2)
        pipe = pipe.apply(
            tf.contrib.data.bucket_by_sequence_length(
                lambda x: x['length'],
                bucket_boundaries=boundaries,
                bucket_batch_sizes=self.make_batch_sizes(batch_size, boundaries),
                padded_shapes={
                    'length': [],
                    'data': [None]
                }
            )
        )

        self.pipe = pipe.prefetch(2)


class ModelInput(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.handle = tf.placeholder(dtype=tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            string_handle=self.handle,
            output_types={
                'length': tf.int32,
                'data': tf.int32
            },
            output_shapes={
                'length': [None],
                'data': [None, None]
            }
        )

        self.next = self.iterator.get_next()
        self.data = self.next['data']
        self.length = self.next['length']
        data_shape = tf.shape(self.data)
        self.batch_size = data_shape[0]
        self.batch_maxlen = data_shape[1]


class LanguageModel(object):
    def __init__(self, args, data: ModelInput, infer=False):
        self.args = args
        self.data = data

        with tf.variable_scope('model', reuse=infer) as scope:
            self.global_step = tf.train.get_or_create_global_step()
            self.cur_example = tf.get_variable(
                name='cur_example',
                shape=[],
                dtype=tf.int64,
                trainable=False,
                initializer=tf.zeros_initializer(dtype=tf.int64)
            )

            self.inc_num_examples = self.cur_example.assign_add(tf.to_int64(data.batch_size))

            self.embedding = tf.get_variable(
                name='embedding',
                shape=[self.data.vocab.size, args.embed_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    stddev=args.embed_size ** -0.5,
                    dtype=tf.float32
                )
            )

            self.embedded_input = tf.nn.embedding_lookup(
                params=self.embedding,
                ids=data.data
            )

            cell = tf.contrib.rnn.LSTMBlockCell(
                num_units=args.lstm_size
            )

            self.rnn_out, self.rnn_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=self.embedded_input,
                sequence_length=data.length,
                swap_memory=True,
                scope=scope,
                dtype=self.embedded_input.dtype
            )

            if args.lstm_size != args.embed_size:
                self.output_layer = tf.layers.Dense(
                    units=args.embed_size,
                    name="out_proj"
                )
                self.outputs = self.output_layer(self.rnn_out)
            else:
                self.outputs = self.rnn_out

            self.outputs = self.outputs[:, :-1]
            self.answer_indices = data.data[:, 1:]

            if args.sample_loss != 0:
                sample_labels = tf.reshape(self.answer_indices, [-1, 1])
                sample_input = tf.reshape(self.outputs, [-1, args.embed_size])
                sample_output = tf.nn.sampled_softmax_loss(
                    weights=self.embedding,
                    biases=tf.constant(0, dtype=self.embedding.dtype, shape=[data.vocab.size]),
                    labels=sample_labels,
                    inputs=sample_input,
                    num_sampled=args.sample_loss,
                    num_classes=data.vocab.size
                )
                self.raw_loss = tf.reshape(sample_output, [data.batch_size, data.batch_maxlen])
            else:
                logits = tf.einsum('blk,nk->bln', self.outputs, self.embedding)
                self.raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.answer_indices,
                    logits=logits
                )

            self.mask = tf.not_equal(self.answer_indices, 0)
            self.masked_loss = tf.boolean_mask(self.raw_loss, self.mask)
            self.loss = tf.reduce_mean(self.masked_loss)


class Training(object):
    def __init__(self, model: LanguageModel):
        args = model.args

        # Learning rate logic from Transformer paper
        base_rate = model.args.embed_size ** -0.5
        if args.lrate is not None:
            base_rate = args.lrate

        step = tf.to_float(model.inc_num_examples) / 5000
        self.lrate = base_rate * tf.minimum(tf.rsqrt(step), step * args.warmup ** -1.5)

        self.optimizer = tf.train.AdamOptimizer(
            self.lrate
        )

        self.optimize_op = self.optimizer.minimize(
            loss=model.loss,
            global_step=model.global_step
        )


class Summaries(object):
    def __init__(self, model: LanguageModel, train: Training):
        self.mdl = model
        self.train = train
        d = model.data
        tf.summary.scalar('misc/lrate', train.lrate)
        tf.summary.scalar('misc/nonzero_ratio',
                          tf.reduce_sum(tf.to_float(model.mask)) / tf.to_float(d.batch_size * d.batch_maxlen))
        tf.summary.scalar('misc/batch_size', d.batch_size)
        tf.summary.scalar('misc/batch_maxlen', d.batch_maxlen)
        tf.summary.scalar('train/loss', model.loss)

        self.num_examples = tf.placeholder(
            dtype=tf.float32,
            shape=[]
        )

        self.time_summaries = tf.placeholder(
            dtype=tf.float32,
            shape=[]
        )

        tf.summary.scalar('misc/example_rate', self.num_examples / self.time_summaries)

        self.train_summaries = tf.summary.merge_all()

    def projector(self, writer):
        import tensorboard.plugins.projector as prj
        cfg = prj.ProjectorConfig()
        embed = cfg.embeddings.add()
        embed.tensor_name = self.mdl.embedding.name
        embed.metadata_path = self.mdl.data.vocab.filename
        prj.visualize_embeddings(writer, cfg)


class DevCalculator(object):
    def __init__(self, args, idata):
        self.idata = idata
        self.model = LanguageModel(args, idata, True)
        self.logits = tf.einsum('blk,nk->bln', self.model.outputs, self.model.embedding)
        self.normalizer = tf.reduce_logsumexp(self.logits, axis=-1, keep_dims=True)
        self.log_probs = self.logits - self.normalizer
        self.probs = tf.exp(self.log_probs)
        self.log2_probs = self.log_probs / tf.log(2.0)
        self.entropy = -tf.reduce_sum(self.log2_probs * self.probs, axis=-1)
        self.perplexity = tf.pow(2.0, self.entropy)
        self.avg_perplexity = tf.reduce_mean(tf.boolean_mask(self.perplexity, self.model.mask))

        self.pipeline = InputPipeline(args, args.dev_path, args.dev_batch, idata.vocab)
        self.iterator = self.pipeline.pipe.make_initializable_iterator()

        self.ppx_value = tf.placeholder(tf.float32, [])
        self.ppx_summary = tf.summary.scalar("dev/ppx", self.ppx_value, [])

    def run(self, sess, writer, step):
        _, handle = sess.run([self.iterator.initializer, self.iterator.string_handle()])
        feed_dict = {self.idata.handle: handle}
        ppx = 0.0
        cnt = 0
        try:
            while True:
                ppx += sess.run(self.avg_perplexity, feed_dict=feed_dict)
                cnt += 1
        except tf.errors.OutOfRangeError:
            summ = sess.run(self.ppx_summary, {self.ppx_value: ppx / cnt})
            writer.add_summary(summ, step)


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.vocab = Vocabulary(args)
        self.idata = ModelInput(self.vocab)
        self.model = LanguageModel(args, self.idata, False)
        self.training = Training(self.model)
        self.summ = Summaries(self.model, self.training)

        mgr = tf.train.SessionManager(local_init_op=[
            tf.local_variables_initializer(),
            tf.tables_initializer()
        ])
        self.saver = tf.train.Saver()

        self.sess = mgr.prepare_session('', checkpoint_dir=args.snapshot_dir, init_op=[
            tf.global_variables_initializer()
        ], saver=self.saver)

        self.summ_wr = tf.summary.FileWriter(args.snapshot_dir)

        os.makedirs(args.snapshot_dir, exist_ok=True)

        self.save_path = args.snapshot_dir + "/snap"
        if args.dev_path is not None:
            self.dev = DevCalculator(args, self.idata)

    def train(self):
        try:
            self.summ.projector(self.summ_wr)
            self._train()
        except tf.errors.OutOfRangeError:
            self.saver.save(self.sess, self.save_path, global_step=self.model.cur_example)

    def _train(self):
        idata = InputPipeline(self.args, self.args.input, self.args.batch_size, self.vocab)
        iter = idata.pipe.make_initializable_iterator()
        iter_ref, _ = self.sess.run([iter.string_handle(), iter.initializer])

        feed_dict = {self.idata.handle: iter_ref}
        prev_time = time.monotonic()
        summ_prev = prev_time
        snap_prev = prev_time
        dev_prev = prev_time

        num_examples = 0
        step = 0

        while True:
            fetches = {
                'opt_op': self.training.optimize_op,
                'step': self.model.cur_example
            }

            start_time = time.monotonic()

            summ_eplaced = start_time - summ_prev

            if summ_eplaced > self.args.summary_freq:
                nex = step - num_examples
                feed_dict[self.summ.num_examples] = float(nex)
                feed_dict[self.summ.time_summaries] = summ_eplaced
                fetches['summary'] = self.summ.train_summaries
                # print(f"summary: processed {nex} examples in {summ_eplaced}: {nex/summ_eplaced} per sec")

            result = self.sess.run(fetches, feed_dict)

            summary = result.get('summary', None)
            if summary is not None:
                self.summ_wr.add_summary(summary, step)
                summ_prev = start_time
                num_examples = step

            step = result['step']

            if (start_time - snap_prev) > self.args.snapshot_freq:
                self.saver.save(self.sess, self.save_path, step)
                snap_prev = start_time

            if self.dev is not None and (start_time - dev_prev) > self.args.dev_freq:
                print("Doing eval")
                self.dev.run(self.sess, self.summ_wr, step)
                dev_prev = start_time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot-dir', required=True)
    p.add_argument('--snapshot-freq', type=float, default=15)
    p.add_argument('--summary-freq', type=float, default=3)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--input', required=True)
    p.add_argument('--shuffle-size', type=int, default=10000)
    p.add_argument('--vocab-file', required=True)
    p.add_argument('--sample-loss', type=int, default=0)
    p.add_argument('--embed-size', type=int, default=32)
    p.add_argument('--lstm-size', type=int, default=40)
    p.add_argument('--lrate', type=float)
    p.add_argument('--warmup', type=int, default=10000)
    p.add_argument('--tf_records', action='store_true')
    p.add_argument('--parallel_map', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=1000)
    p.add_argument('--dev-path')
    p.add_argument('--dev-batch', type=int, default=1000)
    p.add_argument('--dev-freq', type=float, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    runner = Runner(args)
    runner.train()


if __name__ == '__main__':
    main()
