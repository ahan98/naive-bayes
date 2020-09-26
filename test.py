# import numpy as np


def test(table, cumulative, dataset, index_to_name):
    """
    Classifies each row of <dataset> given frequency tables <table> and
    <cumulative> (see <train.py>).

    INPUTS:
    - table [dict]: conditional frequency table
    - cumulative [dict]: cumulative[i][j] denotes number of times attribute i
      appeared with class value j
    - dataset [list]: dataset[i][j] denotes the j-th attribute value of
      the i-th example
    - index_to_name [list]: index_to_name[i] maps the name of attribute i to
      original name given by .arff file

    RETURNS:
    - predictions (list): predictions[i] stores the class code with the maximum
      posterior for the i-th test example
    - targets: targets[i] stores the ground truth labels for the
      i-th test example
    """
    predictions = []
    targets = []
    class_name = index_to_name[-1]

    for instance in dataset:
        target = instance[-1]
        class_argmax, max_likelihood = None, float("-inf")
        for class_val in table[class_name]:
            # initialize likelihood to class prior
            likelihood = table[class_name][class_val] / table["TOTAL"]

            # multiply by conditional probabilities
            for i in range(len(instance) - 1):
                attr_name = index_to_name[i]
                attr_val = instance[i]
                if attr_val == "?":
                    continue

                # apply laplace smoothing
                numer = table[attr_name][attr_val][class_val] + 1
                n_attr_vals = len(table[attr_name])
                denom = cumulative[attr_name][class_val] + n_attr_vals
                likelihood *= numer / denom

            if likelihood > max_likelihood:
                max_likelihood = likelihood
                class_argmax = class_val

        predictions.append(class_argmax)
        targets.append(target)

    return predictions, targets
