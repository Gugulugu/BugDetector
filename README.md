Bug Detector a modification of DeepBugs
====================================
This is a Bug Detector model which is build on top of the DeepBugs approach. A transformer model for binary & multi classification was implemented in this project.
DeepBugs Paper:[DeepBugs paper](http://software-lab.org/publications/oopsla2018_DeepBugs.pdf)

Quick Start
--------------
Here is a simple version from DeepBugs, proposed by the authors:
A quick and easy way to play with a simplified version of DeepBugs is a [Jupyter notebook, which you can run on Google's Colaboratory](https://colab.research.google.com/github/michaelpradel/DeepBugs/blob/master/DeepBugs.ipynb). To use the full DeepBugs tool, read on.

Overview
-------------
* All commands are called from the main directory.
* Python code (most of the implementation) and JavaScript code (for extracting data from .js files) are in the `/python` and `/javascript` directories.
* All data to learn from, e.g., .js files are expected to be in the `/data` directory.
* All data that is generated, e.g., intermediate representations, are written into the main directory. It is recommended to move them into separate directories.
* All generated data files have a timestamp as part of the file name. Below, all files are used with `*`. When running commands multiple times, make sure to use the most recent files.


Requirements
------------------

* Node.js
* npm modules (install with `npm install module_name`): acorn, estraverse, walk-sync
* Python 3
* Python packages: keras, scipy, numpy, sklearn
* Further packages used (requirements.txt)


JavaScript Corpus
-----------------------

* The full corpus can be downloaded [here](http://www.srl.inf.ethz.ch/js150.php) and is expected to be stored in `data/js/programs_all`. It consists of 100.000 training files, listed in `data/js/programs_training.txt`, and 50.000 files for validation, listed in `data/js/programs_eval.txt`. 
* This repository contains only a very small subset of the corpus. It is stored in `data/js/programs_50`. Training and validation files for the small corpus are listed in `data/js/programs_50_training.txt` and `data/js/programs_50_eval.txt`.


Learning a Bug Detector
-------------------------------

Creating a bug detector consists of two main steps:
1) Extract positive (i.e., likely correct) and negative (i.e., likely buggy) training examples from code.
2) Train a classifier to distinguish correct from incorrect code examples.

Each bug detector addresses a particular bug pattern, e.g.:

  * The `SwappedArgs` bug detector looks for accidentally swapped arguments of a function call, e.g., calling `setPoint(y,x)` instead of `setPoint(x,y)`.
  * The `BinOperator` bug detector looks for incorrect operators in binary operations, e.g., `i <= len` instead of `i < len`.
  * The `IncorrectBinaryOperand` bug detector looks for incorrect operands in binary operations, e.g., `height - x` instead of `height - y`.

#### Step 1: Extract positive and negative training examples

`node javascript/extractFromJS.js calls --parallel 4 data/js/programs_50_training.txt data/js/programs_50`

  * The `--parallel` argument sets the number of processes to run.
  * `programs_50_training.txt` contains files to include (one file per line). To extract data for validation, run the command with `data/js/programs_50_eval.txt`.
  * The last argument is a directory that gets recursively scanned for .js files, considering only files listed in the file provided as the second argument.
  * The command produces `calls_*.json` files, which is data suitable for the `SwappedArgs` bug detector. For the other bug two detectors, replace `calls` with `binOps` in the above command.

#### Step 2: Train a model to identify Bugs 

A) To run the original model from DeepBugs use:

`python3 python/BugLearnAndValidate.py --pattern SwappedArgs --token_emb token_to_vector.json --type_emb type_to_vector.json --node_emb node_type_to_vector.json --training_data calls_xx*.json --validation_data calls_yy*.json`

  * The first argument selects the bug pattern.
  * The next three arguments are vector representations for tokens (here: identifiers and literals), for types, and for AST node types. These files are provided in the repository.
  * The remaining arguments are two lists of .json files. They contain the training and validation data extracted in Step 1.
  * After learning the bug detector, the command measures accurracy and recall w.r.t. seeded bugs and writes a list of potential bugs in the unmodified validation code (see `poss_anomalies.txt`).

B) To run the transformer model for Binary Classification:

 'python3 python/BugLearnAndValidateTransformer.py --pattern SwappedArgs --token_emb token_to_vector.json --training_data calls_training/calls_*.json --validation_data calls_eval/calls_*.json'

C) To run the transformer model three Bug Patterns:

For binary classification:

  'python3 python/BugLearnAndValidateTransformerMergeBinary.py --token_emb token_to_vector.json --training_data_Swapped merged_buggs_origi/calls_training/calls_*.json --training_data_BinOp merged_buggs_origi/binops_BinOperator_training/binOps_*.json --training_data_IncBinOp merged_buggs_origi/binops_IncBinOperand_training/binOps_*.json  --validation_data_Swapped merged_buggs_origi/calls_eval/calls_*.json  --validation_data_BinOp merged_buggs_origi/binops_BinOperator_eval/binOps_*.json --validation_data_IncBinOp merged_buggs_origi/binops_IncBinOperand_eval/binOps_*.json'

For multi classification:

 'python3 python/BugLearnAndValidateTransformerMergeMulti.py --token_emb token_to_vector.json --training_data_Swapped merged_buggs_origi/calls_training/calls_*.json --training_data_BinOp merged_buggs_origi/binops_BinOperator_training/binOps_*.json --training_data_IncBinOp merged_buggs_origi/binops_IncBinOperand_training/binOps_*.json  --validation_data_Swapped merged_buggs_origi/calls_eval/calls_*.json  --validation_data_BinOp merged_buggs_origi/binops_BinOperator_eval/binOps_*.json --validation_data_IncBinOp merged_buggs_origi/binops_IncBinOperand_eval/binOps_*.json'

#Note the Directorys of the Bug Dectors need do be modified according your data directory

Note that learning a bug detector from the very small corpus of 50 programs will yield a classifier with low accuracy that is unlikely to be useful. To leverage the full power of DeepBugs, you'll need a larger code corpus, e.g., the [JS150 corpus](http://www.srl.inf.ethz.ch/js150.php) mentioned above.

