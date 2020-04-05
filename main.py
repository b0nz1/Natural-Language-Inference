import codecs as cdc
import argparse
import logging
import json
import random
import numpy as np
import dynet as dy
from pathlib import Path

# create logger
logger = logging.getLogger("mylog")
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

LABELS = {
    u"entailment": 0,
    u"contradiction": 1,
    u"neutral": 2
}
    
#read vectors from w2v file
def vecs_from_file(file):
        words = {}
        with cdc.open(file, "r",encoding="utf8") as lines:
            for l in lines.readlines():
                vec = l.split()
                w = vec.pop(0)
                words[w] = np.array(list(map(float, vec)))
        return words
        
class Embedding:
    def __init__(self, file):
        logger.info("-----")
        logger.info("Read word vectors " + file)
        self.words = vecs_from_file(file)
        logger.info("Create UNK vector")
        self.count_known = 0
        self.count_unknown = 0
        
        vectors = []
        for i, w in enumerate(self.words):
            if i > 100:
                break
            vectors.append(self.words[w])
        self.UNK_vec = np.average(vectors, axis=0)
    
        #embed and count words
    def embedWords(self, words):
        lst = []
        for w in words:
            if w in self.words:
                self.count_known += 1
                lst.append(self.words[w])
            else:
                self.count_unknown += 1
                lst.append(self.UNK_vec)
        return np.array(lst)
    
#Read and process SNLI data
class SNLIData:
    def __init__(self, type, file, embedding):
        logger.info("-----")
        logger.info("Read data " + type + " " + file)

        self.data = []
        
        #self.flag = True
        with open(file) as lines:
            for l in lines:
                datum = json.loads(l)
                gold = datum["gold_label"]
                if gold != u"-":
                    label = LABELS[gold]
                    s1 = embedding.embedWords(datum["sentence1"].lower().rstrip(".").split())
                    s2 = embedding.embedWords(datum["sentence2"].lower().rstrip(".").split())

                    self.data.append((s1, s2, label))

        logger.info(type + " size # sent " + str(len(self.data)))
        logger.info("Num known words " + str(embedding.count_known) + " / Num unknown words " + str(embedding.count_unknown))


class Model:
    #init all the layers
    def __init__(self, embed_size, hidden_size, labels_size):
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.embeddingLinear = self.model.add_parameters((embed_size, hidden_size))
        self.mlpF1 = self.model.add_parameters((hidden_size, hidden_size))
        self.mlpF2 = self.model.add_parameters((hidden_size, hidden_size))
        self.mlpG1 = self.model.add_parameters((2 * hidden_size, hidden_size))
        self.mlpG2 = self.model.add_parameters((hidden_size, hidden_size))
        self.mlpH1 = self.model.add_parameters((2 * hidden_size, hidden_size))
        self.mlpH2 = self.model.add_parameters((hidden_size, hidden_size))
        self.final_layer = self.model.add_parameters((hidden_size, labels_size))

    def accuracy(self, data):
        correct = total = 0.0
        prediction = []
        gold = []
        for (s1, s2, label) in data:
            gold.append(label)
            prediction.append(self.forward(s1, s2))    
            
        for p, g in zip(prediction, gold):
            total += 1
            #the prediction is correct
            if p == g:
                correct += 1

        return correct / total

    def forward(self, s1, s2, label=None):
        eL = dy.parameter(self.embeddingLinear)
        s1 = dy.inputTensor(s1) * eL
        s2 = dy.inputTensor(s2) * eL

        # F step
        Lf1 = dy.parameter(self.mlpF1)
        Fs1 = dy.rectify(dy.dropout(s1, 0.2) * Lf1)
        Fs2 = dy.rectify(dy.dropout(s2, 0.2) * Lf1)
        Lf2 = dy.parameter(self.mlpF2)
        Fs1 = dy.rectify(dy.dropout(Fs1, 0.2) * Lf2)
        Fs2 = dy.rectify(dy.dropout(Fs2, 0.2) * Lf2)

        # Attention scoring
        score1 = Fs1 * dy.transpose(Fs2)
        prob1 = dy.softmax(score1)

        score2 = dy.transpose(score1)
        prob2 = dy.softmax(score2)

        # Align pairs using attention
        s1Pairs = dy.concatenate_cols([s1, prob1 * s2])
        s2Pairs = dy.concatenate_cols([s2, prob2 * s1])

        # G step
        Lg1 = dy.parameter(self.mlpG1)
        Gs1 = dy.rectify(dy.dropout(s1Pairs, 0.2) * Lg1)
        Gs2 = dy.rectify(dy.dropout(s2Pairs, 0.2) * Lg1)
        Lg2 = dy.parameter(self.mlpG2)
        Gs1 = dy.rectify(dy.dropout(Gs1, 0.2) * Lg2)
        Gs2 = dy.rectify(dy.dropout(Gs2, 0.2) * Lg2)

        # Sum
        Ss1 = dy.sum_dim(Gs1, [0])
        Ss2 = dy.sum_dim(Gs2, [0])

        concatS12 = dy.transpose(dy.concatenate([Ss1, Ss2]))

        # H step
        Lh1 = dy.parameter(self.mlpH1)
        Hs = dy.rectify(dy.dropout(concatS12, 0.2) * Lh1)
        Lh2 = dy.parameter(self.mlpH2)
        Hs = dy.rectify(dy.dropout(Hs, 0.2) * Lh2)

        # Final layer
        final_layer = dy.parameter(self.final_layer)
        final = dy.transpose(Hs * final_layer)

        # Label can be 0...
        if label != None:  
            return dy.pickneglogsoftmax(final, label)
        else:
            out = dy.softmax(final)
            return np.argmax(out.npvalue())

    def save(self, modelFile):
        self.model.save(modelFile)

    def load(self, modelFile):
        self.model.populate(modelFile)


if __name__ == '__main__':
    prsr = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    prsr.add_argument('--train', help='training data file (jsonl)',
                        type=str, default='./snli_1.0/snli_1.0_train.jsonl')

    prsr.add_argument('--dev', help='development data file (jsonl)',
                        type=str, default='./snli_1.0/snli_1.0_dev.jsonl')

    prsr.add_argument('--test', help='test data file (jsonl)',
                        type=str, default='./snli_1.0/snli_1.0_test.jsonl')

    prsr.add_argument('--w2v', help='pretrained word vectors file (word tab vector)',
                        type=str, default='deps.words')

    prsr.add_argument('--embedding_size', help='word embedding size',
                        type=int, default=300)

    prsr.add_argument('--epochs', help='training epochs',
                        type=int, default=25)

    prsr.add_argument('--dev_interval', help='interval for development',
                        type=int, default=1)

    prsr.add_argument('--display_interval', help='interval of display by batches',
                        type=int, default=100000)

    prsr.add_argument('--batch', help='size of batch',
                        type=int, default=20000)

    prsr.add_argument('--model', help='path of model file (not include the name suffix',
                        type=str, default='model.save')

    prsr.add_argument('--dynet-autobatch', help='dynet parameter',
                        type=int, default=1)

    prsr.add_argument('--dynet-mem', help='dynet parameter',
                        type=int, default=8192)

    args = prsr.parse_args()
    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    embedding = Embedding(args.w2v)
    # load train/dev/test data
    trainData = SNLIData("train", args.train, embedding)
    devData = SNLIData("dev", args.dev, embedding)
    testData = SNLIData("test", args.test, embedding)

    model = Model(args.embedding_size, 300, len(LABELS))

    #modelFileCache = Path(args.model)
    if Path(args.model).is_file():
        model.load(args.model)

    loss = 0
    tagged = 0
    for EPOCH in range(args.epochs):
        logger.info("-----")
        logger.info("Starting epoch " + str(EPOCH))
        random.shuffle(trainData.data)

        errors = []
        dy.renew_cg()
        for i, (s1, s2, label) in enumerate(trainData.data, 1):
            #display accurecy and loss every 100K sentences
            if i % args.display_interval == 0:
                avgLoss = loss / tagged
                #losses.append(avgLoss)
                logger.info(str(EPOCH) + "/" + str(i) + ": " + str(avgLoss))
                loss = tagged = 0

                accuracy = model.accuracy(devData.data)
                logger.info("Dev Accuracy: " + str(accuracy))

                model.save(args.model)

            if i % args.batch == 0:
                errors_sum = dy.esum(errors)
                loss += errors_sum.value()
                tagged += args.batch

                errors_sum.backward()
                model.trainer.update()

                dy.renew_cg()
                errors = [] 
            errors.append(model.forward(s1, s2, label))