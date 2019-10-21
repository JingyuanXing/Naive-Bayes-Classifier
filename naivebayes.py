import sys
import io
from collections import Counter
from collections import defaultdict

class NaiveBayes(object):

    def __init__(self, trainingData):
        """
        Initializes a NaiveBayes object and the naive bayes classifier.
        :param trainingData: full text from training file as a single large string
        """

        # make trainingData into a string list list
        trainLL = []
        for line in trainingData.splitlines():
            trainLL.append(line.split())

        # find count(Y=R), count(Y=B), total N, P(Y=R), P(Y=B)
        countY_R = 0
        countY_B = 0
        for line in trainLL:
            if line[0] == "RED":
                countY_R += len(line)-1 # total number of words in this line, minus the dummy "RED"
            else:
                countY_B += len(line)-1 # total number of words in this line, minus the dummy "BLUE"
        N = countY_R + countY_B
        probY_R = countY_R/N
        probY_B = countY_B/N

        # create dictionary of each word count: count(X=xi|Y=R) and count(X=xi|Y=B)
        countX_givenR = Counter()
        countX_givenB = Counter()
        for line in trainLL:
            if line[0] == "RED":
                countX_givenR += Counter(line[1:])
            else:
                countX_givenB += Counter(line[1:])

        print("X given R: ", countX_givenR)
        print("X given B: ", countX_givenB)
        pass

    def estimateLogProbability(self, sentence):
        """
        Using the naive bayes model generated in __init__, calculate the probabilities that this sentence is in each category. Sentence is a single sentence from the test set. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary. 
        :param sentence: the test sentence, as a single string without label
        :return: a dictionary containing log probability for each category
        """
        return {'red': 0, 'blue': 0}

    def testModel(self, testData):
        """
        Using the naive bayes model generated in __init__, test the model using the test data. You should calculate accuracy, precision for each category, and recall for each category. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary.
        :param testData: the test file as a single string
        :return: a dictionary containing each item as identified by the key
        """
        return {'overall accuracy': 0,
                'precision for red': 0,
                'precision for blue': 0,
                'recall for red': 0,
                'recall for blue': 0}

"""
The following code is used only on your local machine. The autograder will only use the functions in the NaiveBayes class.            

You are allowed to modify the code below. But your modifications will not be used on the autograder.
"""
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 naivebayes.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    with io.open(train_txt, 'r', encoding='utf8') as f:
        train_data = f.read()

    with io.open(test_txt, 'r', encoding='utf8') as f:
        test_data = f.read()

    model = NaiveBayes(train_data)
    evaluation = model.testModel(test_data)
    print("overall accuracy: " + str(evaluation['overall accuracy'])
        + "\nprecision for red: " + str(evaluation['precision for red'])
        + "\nprecision for blue: " + str(evaluation['precision for blue'])
        + "\nrecall for red: " + str(evaluation['recall for red'])
        + "\nrecall for blue: " + str(evaluation['recall for blue']))



