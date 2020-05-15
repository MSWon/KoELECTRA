import sentencepiece as spm
import re
import os

class Tokenizer(object):
    """ Tokenizer class """
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()

    def load(self, bpe_model_path):
        """
        :param bpe_model_path: BPE model path
        :return: None
        """
        self.sp.Load(bpe_model_path)

    def train(self, corpus_path, model_name, vocab_size, character_coverage=0.9995):
        """
        :param corpus_path: corpus path to train BPE
        :param model_name: output model prefix name
        :param vocab_size: size of the vocab (default 32k)
        :return:
        """
        train_sp = '--input={} --pad_id=0 --unk_id=1 \
                    --bos_id=2 --eos_id=3 \
                    --model_prefix={} \
                    --user_defined_symbols=[URL],[MASK],[CLS] \
                    --vocab_size={} \
                    --character_coverage={} \
                    --model_type=bpe'.format(corpus_path, model_name, vocab_size, character_coverage)

        print("training BPE model")
        for config in train_sp.replace("\t","").split("--"):
            print(config)

        spm.SentencePieceTrainer.Train(train_sp)

        uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
        model_path = os.path.join(uppath(corpus_path, 1), model_name)

        f = open("{}.vocab".format(model_path), "r")
        with open("{}.vocab2".format(model_path), "w") as f1:
            for sent in f:
                f1.write(sent.split()[0] + "\n")

        os.system("rm {}.vocab".format(model_path))
        os.system("mv {}.vocab2 {}.vocab".format(model_path, model_path))

    def preprocess(self, sent):
        """
        :param sent: input sentence
        :return: preprocessed sentence
        """
        sent = re.sub("\(.*?\)|\[.*?\]", "", sent)
        sent = re.sub("[^0-9a-zA-Z가-힣_\-@\.:&+!?'/,\s]", "", sent)
        sent = re.sub("\s{2,}", " ", sent)
        return sent.strip()

    def url_replace(self, sent):
        """
        :param sent: input sentence
        :return: url replaced sentence
        """
        url_regex = "(http[s]?:/{1,2}([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])*)|(www.([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)"
        sent = re.sub(url_regex, "[URL]", sent)
        return sent

    def preprocess_corpus(self, corpus_path):
        """
        :param corpus_path: corpus path
        :return: BPE tokenized corpus
        """
        f = open(corpus_path, "r")
        n = 0
        with open(corpus_path +".replaced", "w") as f_out:
            for sent in f:
                if n % 10000 == 0:
                    print("{} sentences processed".format(n))
                replaced_sent = self.url_replace(sent).strip()
                f_out.write(replaced_sent + "\n")
                n += 1        
    
    def tokenize(self, sent):
        """
        :param sent: input sentence
        :return: BPE tokenized list
        """
        return self.sp.EncodeAsPieces(sent)

    def tokenize_corpus(self, bpe_model_path, corpus_path):
        """
        :param corpus_path: corpus path
        :return: BPE tokenized corpus
        """
        f = open(corpus_path, "r")
        self.load(bpe_model_path)
        n = 0
        with open(corpus_path +".bpe", "w") as f_out:
            for sent in f:
                if n % 10000 == 0:
                    print("{} sentences tokenized".format(n))
                bpe_sent = " ".join(self.tokenize(sent)).strip()
                f_out.write(bpe_sent + "\n")
                n += 1
        
