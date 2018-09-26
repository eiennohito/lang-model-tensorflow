# Overview

This is a simple implementation of a Tensorflow RNN-LSTM
langauge model with tf.Dataset input pipeline.
It can process ~10k sentences/s for 
128 embedding size and 256 LSTM size.

# How to run

Have Python 3.6+.
Install tensorflow (or tensorflow-gpu).
The script itself expects the input data be pretokenized,
1 line per training example.

The jpp2line.py script can convert sentences in Juman
format to the expected input format.

`python3 jpp2line.py -o <output> <num_column> <input_files>`,
where num_column is the number of a column to output.
0 will be word surface forms.

Also you need to create a vocabulary file,
lminp2dic.py does that.
The command to run is`python3 lminp2dic.py <output> <max_words> <input_files>`.

You can run the training script itself as:
```
python3 language_model.py \
        --snapshot-freq=60 \
        --summary-freq=3 \
        --epochs=5 \
        --input='input/*.in' \
        --shuffle-size=100000 \
        --vocab-file=input.dic \
        --embed-size=128 \
        --lstm-size=256 \
        --warmup=2000 \
        --batch_size=25000 \
        --sample-loss=1000 \
        --snapshot-dir=snaps
```

Batch size means the maximum number of tokens, 
not examples in a batch.
