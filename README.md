# naive-bayes
A simple Naive Bayes classifier for nominal data. Completed as coursework for Williams College
CS374: Machine Learning.

## How to Run

`python main.py path/to/data.arff` runs the classifier using `path/to/data.arff`
for the dataset.

Alternatively, you may run the classifer on all `.arff` files via
`./arff_script.sh`.

## Notation

Throughout the documentation, we distinguish the "class" from "attribute." Given
`n` attribute variables (as denoted by `@attribute` in the `.arff` data file),
we denote the `n`-th attribute as "class" (e.g., "play"), and the others as
"attributes" (e.g., "outlook", "humidity").
