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
        self.countY_R = 0
        self.countY_B = 0
        for line in trainLL:
            if line[0] == "RED":
                self.countY_R += len(line)-1 # total number of words in this line, minus the dummy "RED"
            else:
                self.countY_B += len(line)-1 # total number of words in this line, minus the dummy "BLUE"
        N = self.countY_R + self.countY_B
        self.probY_R = self.countY_R/N
        self.probY_B = self.countY_B/N

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
            self.probX_givenR[k] = countX_givenR[k]/self.countY_R
        for k in countX_givenB:
            self.probX_givenB[k] = countX_givenB[k]/self.countY_B
    
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

        # find the product of P(Y=R) * \prod{P(X=xi | Y=R)}
        #                 and P(Y=B) * \prod{P(X=xi | Y=B)}
        test_probX_givenR = self.probY_R
        test_probX_givenB = self.probY_B
        for word in testL:
            if self.probX_givenR[word] == 0:
                test_probX_givenR += math.log(1/(len(self.probX_givenR)+self.countY_R))
            else:
                test_probX_givenR += math.log(self.probX_givenR[word])

            if self.probX_givenB[word] == 0:
                test_probX_givenB += math.log(1/(len(self.probX_givenB)+self.countY_B))
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

        # make testData into a string list list
        testLL = []
        for line in (testData.splitlines()):
            testLL.append(line)

        # predict and test each line
        TP_R, TN_R, FP_R, FN_R = 0, 0, 0, 0
        TP_B, TN_B, FP_B, FN_B = 0, 0, 0, 0
        for line in testLL:
            actualClass = line.split()[0]
            # print(self.estimateLogProbability(line)['red'], self.estimateLogProbability(line)['blue'])
            if self.estimateLogProbability(line)['red'] > self.estimateLogProbability(line)['blue']:
                predictedClass = "RED"
            else:
                predictedClass = "BLUE"
            # print(actualClass, predictedClass)

            # calculate
            if (actualClass == "RED" and predictedClass == "RED"): 
                TP_R += 1
                TN_B += 1
            elif (actualClass == "RED" and predictedClass == "BLUE"):
                FN_R += 1
                FP_B += 1
            elif (actualClass == "BLUE" and predictedClass == "RED"):
                FP_R += 1
                FN_B += 1
            elif (actualClass == "BLUE" and predictedClass == "BLUE"):
                TN_R += 1
                TP_B += 1

        # overall accuracy, calculate use either red or blue is fine
        accu_overall = (TP_R+TN_R)/(TP_R+TN_R+FN_R+FP_R)
        # precision for red
        preci_red = (TP_R)/(TP_R+FP_R)
        # precison for blue
        preci_blue = TP_B/(TP_B+FP_B)
        # recall for red
        recall_red = (TP_R)/(TP_R+FN_R)
        # recall for blue
        recall_blue = TP_B/(TP_B+FN_B)



        return {'overall accuracy': accu_overall,
                'precision for red': preci_red,
                'precision for blue': preci_blue,
                'recall for red': recall_red,
                'recall for blue': recall_blue}

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



