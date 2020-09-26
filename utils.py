import argparse
import json


def parse_filename():
    """
    Parses path/to/data.arff from command line.
    """
    parser = argparse.ArgumentParser(description="Naive Bayes classifier",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("data", type=str, help="path to .arff file")
    args = vars(parser.parse_args())
    return args["data"]


def accuracy(list1, list2):
    """
    Computes the element-wise similarity between lists list1 and list2.

    INPUTS:
    - list1 [list]: list of arbitrary values
    - list2 [list]: list of arbitrary values

    RETURNS:
    - n_correct [int]: total num. of elementwise matches
    - n_total [int]: total num. elements compared
    """
    assert len(list1) == len(list2)
    n_total = len(list1)
    n_correct = sum([list1[i] == list2[i] for i in range(n_total)])
    return n_correct, n_total


def pretty_print(d):
    """ Prints dictionaries in a more readable format. Useful for debuggin. """
    print(json.dumps(d, indent=4))
