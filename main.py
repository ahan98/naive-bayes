from train import train, update_table
from test import test
from utils import accuracy, parse_filename, pretty_print


def hold_one_out(filename):
    """
    Performs hold-one-out testing. Each iteration, one row is held out as the
    test set, and the remaining rows are used as the training set.

    Returns the testing accuracy (# correct / # total examples)
    """
    # compute frequencies using ALL examples first
    table, cumulative, dataset, index_to_name = train(filename)
    n_correct = n_total = 0

    for i, instance in enumerate(dataset):
        # remove example i from dataset and its respective frequencies
        update_table(table, cumulative, instance, index_to_name, remove=True)

        # classify single held-out example
        preds, targets = test(table, cumulative, [dataset[i]], index_to_name)
        correct, total = accuracy(preds, targets)
        n_correct += correct
        n_total += total

        # restore example i and its frequencies
        update_table(table, cumulative, instance, index_to_name, remove=False)

    print("results:", n_correct, "/", n_total)
    return n_correct / n_total


def full_train_and_test(filename):
    """
    Uses the entire dataset for both training and testing.

    Returns the testing accuracy.
    """
    table, cumulative, dataset, index_to_name = train(filename)
    preds, targets = test(table, cumulative, dataset, index_to_name)
    n_correct, n_total = accuracy(preds, targets)
    print("results:", n_correct, "/", n_total)
    return n_correct / n_total


if __name__ == "__main__":
    filename = parse_filename()

    print("\nUsing entire dataset for train and test")
    acc = full_train_and_test(filename)
    print("Accuracy:", acc)

    print("\nUsing hold-one-out test")
    acc = hold_one_out(filename)
    print("Accuracy:", acc)
