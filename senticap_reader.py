import json
import sys
import argparse


class SenticapSentence(object):
    """
    Stores details about a sentence.

    ivar tokens: A tokenized version of the sentence with punctuation removed and
            words made lower case.
    ivar word_sentiment: Indicates which words are part of an Adjective Noun
            Pair with sentiment; 1 iff the word is part of an ANP with sentiment.
    ivar sentiment_polarity: Does this sentence express positive or negative sentiment.
    ivar raw_sentence: The caption without any processing; taken directly from MTURK.
    """

    NEGATIVE_SENTIMENT = 0
    POSITIVE_SENTIMENT = 1

    def __init__(self):
        self.tokens = []
        self.word_sentiment = []
        self.sentiment_polarity = []
        self.raw_sentence = []

    def setTokens(self, tokens):
        assert isinstance(tokens, list)
        for tok in tokens:
            assert isinstance(tok, str) or isinstance(tok, unicode)

        self.tokens = tokens

    def setWordSentiment(self, word_sentiment):
        assert isinstance(word_sentiment, list)

        self.word_sentiment = [int(s) for s in word_sentiment]

    def setSentimentPolarity(self, sentiment_polarity):
        assert sentiment_polarity in [self.NEGATIVE_SENTIMENT, self.POSITIVE_SENTIMENT]

        self.sentiment_polarity = sentiment_polarity

    def setRawSentence(self, raw_sentence):
        assert isinstance(raw_sentence, str) or isinstance(raw_sentence, unicode)

        self.raw_sentence = raw_sentence

    def getTokens(self):
        return self.tokens

    def getWordSentiment(self):
        return self.word_sentiment

    def getSentimentPolarity(self):
        return self.sentiment_polarity

    def getRawsentence(self):
        return self.raw_sentence


class SenticapImage(object):
    """
    Stores details about a sentence.

    ivar filename: The filename of the image in the MSCOCO dataset
    ivar imgid: A unique but arbritrary number assigned to each image.
    ivar sentences: A list of sentences corresponding to this image of
            type `SenticapSentence`.
    ivar split: Indicates if this is part of the TEST, TRAIN or VAL split.

    """
    TEST_SPLIT = 0
    TRAIN_SPLIT = 1
    VAL_SPLIT = 2

    def __init__(self):
        self.filename = ""
        self.imgid = None
        self.sentences = []
        self.split = None

    def setFilename(self, filename):
        assert isinstance(filename, str) or isinstance(filename, unicode)
        self.filename = filename

    def setImgID(self, imgid):
        self.imgid = imgid

    def addSentence(self, sentence):
        assert isinstance(sentence, SenticapSentence)
        self.sentences.append(sentence)

    def setSplit(self, split):
        assert split in [self.TEST_SPLIT, self.TRAIN_SPLIT, self.VAL_SPLIT]
        self.split = split

    def getFilename(self):
        return self.filename

    def getImgID(self):
        return self.imgid

    def getSentences(self):
        return self.sentences

    def getSplit(self):
        return self.split


class SenticapReader(object):
    """Handles the reading of the senticap dataset.
    Has functions to write examples to a simple csv format,
    and to count the number of examples.
    """

    images = []

    def __init__(self, filename):
        """
        Initializer that reads a senticap json file

        param filename: the file path of the json file
        """
        self.readJson(filename)
        self.filename = filename

    def readJson(self, filename):
        """
        Read a senticap json file and load it into `SenticapImage` and
        `SenticapSentence` classes. The result is saved in `self.images`.

        param filename: the file path of the json file
        """

        data = json.load(open(filename, "r"))
        for image in data["images"]:

            # create the SenticapImage entry
            im = SenticapImage()
            im.setFilename(image["filename"])
            if image["split"] == "train":
                im.setSplit(im.TRAIN_SPLIT)
            elif image["split"] == "test":
                im.setSplit(im.TEST_SPLIT)
            elif image["split"] == "val":
                im.setSplit(im.VAL_SPLIT)
            im.setImgID(image["imgid"])

            # for this image create all the SenticapSentence entries
            for sent in image["sentences"]:
                se = SenticapSentence()
                se.setTokens(sent["tokens"])
                se.setWordSentiment(sent["word_sentiment"])
                if sent["sentiment"] == 0:
                    se.setSentimentPolarity(se.NEGATIVE_SENTIMENT)
                else:
                    se.setSentimentPolarity(se.POSITIVE_SENTIMENT)
                se.setRawSentence(sent["raw"])
                im.addSentence(se)

            self.images.append(im)

    def writeCSV(self, output_filename, train=True, test=True, val=True, pos=True, neg=True):
        """
        Write a CSV file from the examples matching the filter criteria. The
        columns of the csv are (filename, is_positive_sentiment, caption).
        where:
            - B{filename:} is the filename of the MSCOCO image
            - B{is_positive_sentiment:} is 1 if the sentence expresses
                positive sentiment 0 if the sentence expresses
                negative sentiment
            - B{caption:} is the tokenized, lowercase,
                punctuation removed sentence joined with space
                characters

        param output_filename: path of csv to write
        param test: include testing examples
        param val: include validation examples
        param pos: include positive sentiment examples
        param neg: include negative sentiment examples
        """
        fout = open(output_filename, "w")
        fout.write("filename,is_positive_sentiment,caption\n")
        for im in self.images:
            if im.getSplit() == im.TEST_SPLIT and not test:
                continue
            if im.getSplit() == im.TRAIN_SPLIT and not train:
                continue
            if im.getSplit() == im.VAL_SPLIT and not val:
                continue
            sentences = im.getSentences()
            for sent in sentences:
                if sent.getSentimentPolarity() == sent.NEGATIVE_SENTIMENT and not neg:
                    continue
                if sent.getSentimentPolarity() == sent.POSITIVE_SENTIMENT and not pos:
                    continue
                fout.write('%s,%d,"%s"\n' % (im.getFilename(),
                                             sent.getSentimentPolarity() == sent.POSITIVE_SENTIMENT,
                                             ' '.join(sent.getTokens())))
        fout.close()

    def countExamples(self, train=True, test=True, val=True, pos=True, neg=True):
        """
        Count the number of examples matching the filter criteria

        param train: include training examples
        param test: include testing examples
        param val: include validation examples
        param pos: include positive sentiment examples
        param neg: include negative sentiment examples
        return: a tuple giving the number of images with sentences and the
                total number of sentences
        rtype: `tuple(int, int)`
        """
        num_sentence = 0
        num_image_with_sentence = 0
        for im in self.images:
            if im.getSplit() == im.TEST_SPLIT and not test:
                continue
            if im.getSplit() == im.TRAIN_SPLIT and not train:
                continue
            if im.getSplit() == im.VAL_SPLIT and not val:
                continue

            image_has_sentence = False
            sentences = im.getSentences()
            for sent in sentences:
                if sent.getSentimentPolarity() == sent.NEGATIVE_SENTIMENT and not neg:
                    continue
                if sent.getSentimentPolarity() == sent.POSITIVE_SENTIMENT and not pos:
                    continue
                num_sentence += 1
                image_has_sentence = True
            if image_has_sentence:
                num_image_with_sentence += 1

        return (num_image_with_sentence, num_sentence)


def main():
    # handle arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--filename", "-f", default="./data/senticap_dataset.json",
                    help="Path to the senticap json")
    ap.add_argument("--csv_output", "-o", help="Where to write the csv file.")
    ap.add_argument("--train", action="store_true", help="Include the training examples")
    ap.add_argument("--test", action="store_true", help="Include the testing examples")
    ap.add_argument("--val", action="store_true", help="Include the validation examples")
    ap.add_argument("--pos", action="store_true",
                    help="Include the positive sentiment examples")
    ap.add_argument("--neg", action="store_true",
                    help="Include the negative sentiment examples")
    args = ap.parse_args()

    sr = SenticapReader(args.filename)
    if args.csv_output:
        sr.writeCSV(args.csv_output, train=args.train, test=args.test, val=args.val)
    else:
        count = sr.countExamples(train=args.train, test=args.test, val=args.val,
                                 pos=args.pos, neg=args.neg)
        print
        "Input Filename:", args.filename
        print
        "Filters:",
        if args.train:
            print
            "Train",
        if args.test:
            print
            "Test",
        if args.val:
            print
            "Val",
        if args.pos:
            print
            "Positive",
        if args.neg:
            print
            "Negative",
        print
        "\n"
        print
        "Number of images: %d\nNumber of Sentences: %d" % count


if __name__ == "__main__":
    main()
