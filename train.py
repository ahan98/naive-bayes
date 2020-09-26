import re
from utils import pretty_print, parse_filename


def train(filename):
    """
    Parses .arff file and builds frequency tables from data.

    INPUTS:
    - filename [str]: path/to/data.arff

    OUTPUTS:
    - table [dict]: multi-purpose conditional frequency table
      - table[attr_name][attr_val][class_val] stores num. of times <attr_val>
        occurred with <class_val>
      - table[class_name][class_val] stores TOTAL frequency of <class_val>;
        useful to compute priors
      - table["TOTAL"] stores size of dataset
    - cumulative [dict]: cumulative[attr_name][class_val] stores num. of times
      <attr_name> occured with <class_val>
      - equals the sum of table[attr_name][i][class_val] for each attribute
        value i
    - dataset [list]: dataset[i][j] stores value of attribute j for example i
    - index_to_name [list]: index_to_name[i] maps index of attribute to its
      original name
    """
    table = {}
    cumulative = {}
    index_to_name = []
    dataset = []
    reading_data = False  # becomes True after reading line starting w/ "@data"

    lines = None
    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line[0] == "%" or line[0] == "\n":  # line is a comment or blank
            continue

        stripped = re.sub(r"[\n\t\s]*", "", line)  # remove whitespace
        stripped = stripped.lower()

        # parse attribute and class names/values
        if stripped[:10] == "@attribute":
            attr_vals_start = stripped.index("{")

            attr_name = stripped[10: attr_vals_start]
            attr_vals = stripped[attr_vals_start + 1: -1].split(",")
            index_to_name.append(attr_name)

            # initialize table as a dict of (empty) dicts
            table[attr_name] = {k: {} for k in attr_vals}

        # initialize dictionary keys and values
        elif stripped[:5] == "@data":
            reading_data = True
            class_name = index_to_name[-1]

            # initialize conditional frequencies to 0
            for i in range(len(index_to_name) - 1):
                attr_name = index_to_name[i]
                cumulative[attr_name] = {}
                for class_val in table[class_name]:
                    cumulative[attr_name][class_val] = 0
                    for attr_val in table[attr_name]:
                        table[attr_name][attr_val][class_val] = 0

            table[class_name] = {k: 0 for k in table[class_name]}
            table["TOTAL"] = 0

        # update frequencies from each data example
        elif reading_data:
            attr_vals = stripped.split(",")
            dataset.append(attr_vals)
            class_name = index_to_name[-1]
            class_val = attr_vals[-1]
            for i, attr_val in enumerate(attr_vals):
                if attr_val == "?":  # don't count missing attribute values
                    continue

                # increment cumulative and conditional frequencies
                attr_name = index_to_name[i]
                if i == len(attr_vals) - 1:
                    table[attr_name][class_val] += 1
                    table["TOTAL"] += 1
                else:
                    table[attr_name][attr_val][class_val] += 1
                    cumulative[attr_name][class_val] += 1

    return table, cumulative, dataset, index_to_name


def update_table(table, cumulative, instance, index_to_name, remove):
    """
    Helper function for cross-validation. Adds/removs <instance> to/from
    <table>.

    Given data example <instance>, increments/decrements frequencies IN-PLACE
    based on its attribute values.

    INPUTS:
    - table [dict]: multi-purpose conditional frequency table
    - cumulative [dict]: cumulative frequency table
    - dataset [list]: list of training examples
    - index_to_name [list]: maps attribute index to name

    OUTPUTS:
    - None
    """

    class_val = instance[-1]
    sign = -1 if remove else 1  # add or subtract frequencies
    for i, attr_val in enumerate(instance):
        if attr_val == "?":
            continue
        name = index_to_name[i]
        if i == len(instance) - 1:
            table[name][class_val] += sign * 1
            table["TOTAL"] += sign * 1
        else:
            table[name][attr_val][class_val] += sign * 1
            cumulative[name][class_val] += sign * 1


if __name__ == "__main__":
    filename = parse_filename()
    table, cumulative, dataset, index_to_name = train(filename)
    # table, dataset, index_to_name = train("./NominalData/soybean.arff")

    # for d in dataset:
    #     print(d)
    pretty_print(table)
    pretty_print(cumulative)
