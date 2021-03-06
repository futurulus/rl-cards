#!/usr/bin/env python
from stanza.research import config
config.redirect_output()

import datetime

from stanza.monitoring import progress
from stanza.research import evaluate, output

import metrics
import learners
import datasets

parser = config.get_options_parser()
parser.add_argument('--learner', default='Random', choices=learners.LEARNERS.keys(),
                    help='The name of the model to use in the experiment.')
parser.add_argument('--load', metavar='MODEL_FILE', default=None,
                    help='If provided, skip training and instead load a pretrained model '
                         'from the specified path. If None or an empty string, train a '
                         'new model.')
parser.add_argument('--train_size', type=int, default=-1,
                    help='The number of examples to use in training. This number should '
                         '*include* examples held out for validation. If negative, use the '
                         'whole training set.')
parser.add_argument('--validation_size', type=int, default=0,
                    help='The number of examples to hold out from the training set for '
                         'monitoring generalization error.')
parser.add_argument('--test_size', type=int, default=-1,
                    help='The number of examples to use in testing. '
                         'If negative, use the whole dev/test set.')
parser.add_argument('--data_source', default='cards_dev', choices=datasets.SOURCES.keys(),
                    help='The type of data to use.')
parser.add_argument('--metrics', default=['reward'], nargs='+',
                    choices=metrics.METRICS.keys(),
                    help='The evaluation metrics to report for the experiment.')
parser.add_argument('--output_train_data', type=config.boolean, default=False,
                    help='If True, write out the training dataset (after cutting down to '
                         '`train_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--output_test_data', type=config.boolean, default=False,
                    help='If True, write out the evaluation dataset (after cutting down to '
                         '`test_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--output_train_samples', type=config.boolean, default=False,
                    help='If True, write out samples from the trained model on the training '
                         'set after making top-1 predictions.')
parser.add_argument('--output_test_samples', type=config.boolean, default=False,
                    help='If True, write out samples from the trained model on the test '
                         'set after making top-1 predictions.')
parser.add_argument('--progress_tick', type=int, default=10,
                    help='The number of seconds between logging progress updates.')
parser.add_argument('--verbosity', type=int, default=4,
                    help='The amount of logging output (between 0 and 10, inclusive).')


def main():
    options = config.options()

    progress.set_resolution(datetime.timedelta(seconds=options.progress_tick))

    train_size = options.train_size if options.train_size >= 0 else None
    test_size = options.test_size if options.test_size >= 0 else None

    train_data = datasets.SOURCES[options.data_source].train_data()[:train_size]
    if options.validation_size:
        assert options.validation_size < len(train_data), \
            ('No training data after validation split! (%d <= %d)' %
             (len(train_data), options.validation_size))
        validation_data = train_data[-options.validation_size:]
        train_data = train_data[:-options.validation_size]
    else:
        validation_data = None
    test_data = datasets.SOURCES[options.data_source].test_data()[:test_size]

    learner = learners.new(options.learner)

    m = [metrics.METRICS[m] for m in options.metrics]

    if options.load:
        learner.load(options.load)
    else:
        learner.train(train_data, validation_data, metrics=m)
        model_path = config.get_file_path('model')
        if model_path:
            learner.dump(model_path)

        train_results = evaluate.evaluate(learner, train_data, metrics=m, split_id='train',
                                          write_data=options.output_train_data)
        output.output_results(train_results, 'train')

        if options.output_train_samples:
            samples = learner.predict(train_data, random=True)
            config.dump(samples, 'samples.train.jsons', lines=True)

    test_results = evaluate.evaluate(learner, test_data, metrics=m, split_id='eval',
                                     write_data=options.output_test_data)
    output.output_results(test_results, 'eval')

    if options.output_test_samples:
        samples = learner.predict(test_data, random=True)
        config.dump(samples, 'samples.eval.jsons', lines=True)


if __name__ == '__main__':
    main()
