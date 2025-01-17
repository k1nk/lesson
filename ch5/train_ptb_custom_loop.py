#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

This code is a custom loop version of train_ptb.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
from __future__ import print_function
import argparse
import copy
import numpy as np

import chainer
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers

import train_ptb
import train as trn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--data_dir', type=str,   default='data/dazai')
    args = parser.parse_args()

    def evaluate(model, iter):
        # Evaluation routine to be used for validation and test.
        model.predictor.train = False
        evaluator = model.copy()  # to use different state
        evaluator.predictor.reset_state()  # initialize state
        evaluator.predictor.train = False  # dropout does nothing
        sum_perp = 0
        data_count = 0
        for batch in copy.copy(iter):
            x, t = convert.concat_examples(batch, args.gpu)
            loss = evaluator(x, t)
            sum_perp += loss.data
            data_count += 1
        model.predictor.train = True
        return np.exp(float(sum_perp) / data_count)

    # Load the Penn Tree Bank long word sequence dataset
    #train, val, test = chainer.datasets.get_ptb_words()

    dataset, words, vocab = trn.load_data(args.data_dir)
    corpus_len = len(words)
    train_len = int(corpus_len*0.9)
    val_len =  int(corpus_len*0.01)
    test_len = corpus_len - train_len - val_len
    train = dataset[:train_len]
    val =  dataset[train_len:train_len+val_len]
    test = dataset[train_len+val_len:]

    #n_vocab = max(train) + 1  # train is just an array of integers
    n_vocab = len(vocab)
    print('#vocab =', n_vocab)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    # Create the dataset iterators
    train_iter = train_ptb.ParallelSequentialIterator(train, args.batchsize)
    val_iter = train_ptb.ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = train_ptb.ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    rnn = train_ptb.RNNForLM(n_vocab, args.unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        # Make the specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    sum_perp = 0
    count = 0
    iteration = 0
    while train_iter.epoch < args.epoch:
        loss = 0
        iteration += 1
        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(args.bproplen):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = convert.concat_examples(batch, args.gpu)
            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))
            count += 1

        sum_perp += loss.data
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

        if iteration % 20 == 0:
            print('epoch: ', train_iter.epoch)
            print('iteration: ', iteration)
            print('training perplexity: ', np.exp(float(sum_perp) / count))
            sum_perp = 0
            count = 0

        if iteration % 1000 == 0:
            print('epoch: ', train_iter.epoch)
            print('iteration: ', iteration)
            print('validation perplexity: ', evaluate(model, val_iter))
            
        if train_iter.is_new_epoch:
            print('epoch: ', train_iter.epoch)
            print('iteration: ', iteration)
            print('validation perplexity: ', evaluate(model, val_iter))

    # Evaluate on test dataset
    print('test')
    test_perp = evaluate(model, test_iter)
    print('test perplexity:', test_perp)

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('rnnlm.model', model)
    print('save the optimizer')
    serializers.save_npz('rnnlm.state', optimizer)


if __name__ == '__main__':
    main()
