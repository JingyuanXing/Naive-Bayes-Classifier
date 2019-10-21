import sys
import io
from collections import Counter
from collections import defaultdict
import math

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
        self.probY_R = countY_R/N
        self.probY_B = countY_B/N

        # create dictionary of each word count: count(X=xi and Y=R) and count(X=xi and Y=B)
        countX_givenR = Counter()
        countX_givenB = Counter()
        for line in trainLL:
            if line[0] == "RED":
                countX_givenR += Counter(line[1:])
            else:
                countX_givenB += Counter(line[1:])

        # create dictionary of each word probability: 
        # prob(X=xi | Y=R) = count(X=xi and Y=R)/ count(Y=R)
        # prob(X=xi | Y=B) = count(X=xi and Y=B)/ count(Y=B)
        self.probX_givenR = Counter()
        self.probX_givenB = Counter()
        for k in countX_givenR:
            self.probX_givenR[k] = countX_givenR[k]/countY_R
        for k in countX_givenB:
            self.probX_givenB[k] = countX_givenB[k]/countY_B
    
        pass

    def estimateLogProbability(self, sentence):
        """
        Using the naive bayes model generated in __init__, calculate the probabilities that this sentence is in each category. Sentence is a single sentence from the test set. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary. 
        :param sentence: the test sentence, as a single string without label
        :return: a dictionary containing log probability for each category
        """

        # make test sentence into a string list
        testL = sentence.split()
        # print(testL[1:])

        # find the product of P(Y=R) * \prod{P(X=xi | Y=R)}
        #                 and P(Y=B) * \prod{P(X=xi | Y=R)}
        test_probX_givenR = self.probY_R
        test_probX_givenB = self.probY_B
        for word in testL:
            if self.probX_givenR[word] == 0:
                test_probX_givenR += math.log(1/len(self.probX_givenR))
            else:
                test_probX_givenR += math.log(self.probX_givenR[word])

            if self.probX_givenB[word] == 0:
                test_probX_givenB += math.log(1/len(self.probX_givenB))
            else:
                test_probX_givenB += math.log(self.probX_givenB[word])
                
        return {'red': test_probX_givenR, 'blue': test_probX_givenB}

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
    model.estimateLogProbability(test_data.splitlines()[0])



