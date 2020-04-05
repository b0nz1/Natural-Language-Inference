This is a DyNet implementation fo decomposable attention for SNLI partially 

To use the code, run main.py with the next arguments:
- w2v: An embedding file
- train: training file
- dev: dev file
- test: test file

example ( put the SNLI data folder [snli_1.0] and the w2v file deps.words in the same location as the main.py):
python main.py --w2v=deps.words --train=./snli_1.0/snli_1.0_train.jsonl --dev=./snli_1.0/snli_1.0_traindev.jsonl --test=./snli_1.0/snli_1.0_test.jsonl
