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


        ######### step 1
        ######### make trainingData into a string list list
        trainLL = []
        for line in trainingData.splitlines():
            trainLL.append(line.split())
        ######### find count(Y=R), count(Y=B), total N, P(Y=R), P(Y=B)
        self.countY_R = 0
        self.countY_B = 0
        for line in trainLL:
            if line[0] == "RED":
                self.countY_R += len(line)-1
            else:
                self.countY_B += len(line)-1
        N = self.countY_R + self.countY_B
        self.probY_R = self.countY_R/N
        self.probY_B = self.countY_B/N


        ######### step 2
        ######### add <s> for bigram purposes
        for line in trainLL:
            line.insert(1,"<s>")
        ######### create dictionary of each word count: count(X=xi and Y=R) 
        #                                           and count(X=xi and Y=B)
        countX_givenR = Counter()
        countX_givenB = Counter()
        for line in trainLL:
            if line[0] == "RED":
                countX_givenR += Counter(line[1:])
            else:
                countX_givenB += Counter(line[1:])

        

        ######### step 3
        ######### make a tuple list list of (xi-1, xi)
        bigramtrainLL = []
        for line in trainLL:
            tempLine = [(line[i-1], line[i]) for i in range(2,len(line))]
            bigramtrainLL.append(tempLine)
        ######### create dictionary of bigaram counts: count(X=xi-1 xi and Y=R) 
        #                                          and count(X=xi-1 xi and Y=B)
        countXprev_X_givenR = Counter()
        countXprev_X_givenB = Counter()
        for i in range(len(bigramtrainLL)):
            if trainLL[i][0] == "RED":
                countXprev_X_givenR += Counter(bigramtrainLL[i])
            else:
                countXprev_X_givenB += Counter(bigramtrainLL[i])

        ######### step 4
        # create dictionary of each bigram probability: 
        # prob(X=xi-1 xi | Y=R) = count(X=xi-1 xi and Y=R)/ count(X=xi and Y=R)
        # prob(X=xi-1 xi | Y=B) = count(X=xi-1 xi and Y=B)/ count(X=xi and Y=B)
        self.prob_bigramX_givenR = Counter()
        self.prob_bigramX_givenB = Counter()
        for (a,b) in countXprev_X_givenR:
            self.prob_bigramX_givenR[(a,b)] = countXprev_X_givenR[(a,b)]/ countX_givenR[a]
        for (a,b) in countXprev_X_givenB:
            self.prob_bigramX_givenB[(a,b)] = countXprev_X_givenB[(a,b)]/ countX_givenB[a]
    
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

        # insert <s> for bigram purposes
        testL.insert(0, "<s>")
        bigramtestL = [(testL[i-1], testL[i]) for i in range(1,len(testL))]

        # find the product of P(Y=R) * \prod{P(X=xi-1 xi | Y=R)}
        #                 and P(Y=B) * \prod{P(X=xi-1 xi | Y=B)}
        test_prob_bigramX_givenR = self.probY_R
        test_prob_bigramX_givenB = self.probY_B
        for currTuple in bigramtestL:
            if self.prob_bigramX_givenR[currTuple] == 0:
                test_prob_bigramX_givenR += math.log(1/(len(self.prob_bigramX_givenR)+self.countY_R))
            else:
                test_prob_bigramX_givenR += math.log(self.prob_bigramX_givenR[currTuple])

            if self.prob_bigramX_givenB[currTuple] == 0:
                test_prob_bigramX_givenB += math.log(1/(len(self.prob_bigramX_givenB)+self.countY_B))
            else:
                test_prob_bigramX_givenB += math.log(self.prob_bigramX_givenB[currTuple])
                
        return {'red': test_prob_bigramX_givenR, 'blue': test_prob_bigramX_givenB}

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



