# CS-131 Artificial Intelligence
# Assignment5 - Naive Bayes Classification
# Yige Sun
import numpy as np
import math

EXTRA_FEATURE = False

class Radar:
    def __init__(self):
        self.pdf = 'pdf.txt'
        self.vel = 'data.txt'

    # load & clean data
    def cleaner(file_name, num):
        f = open(file_name)
        # raw data
        datas = f.readlines()
        pure_data = []
        # clean data
        for line in datas:
            ver1 = line.strip('\n')
            ver2 = ver1.split(',')
            pure_data.append(ver2)
        # str data -> float data, make data manageable
        for line in range(0, len(pure_data)):
            for col in range(num):
                pure_data[line].append(float(pure_data[line][col]))
            del (pure_data[line][0:num])
        return pure_data

    # since the sum of original probability is not equal to 1
    # this function helps us to normalize probabilities
    # this modification will be in-place
    def normalizer(datas):
        sum = 0
        for data in range(len(datas)):
            for col in range(len(datas[data])):
                sum = sum + datas[data][col]
            for col in range(len(datas[data])):
                datas[data][col] = datas[data][col] / sum
            sum = 0
            for col in range(len(datas[data])):
                sum = sum + datas[data][col]
            sum = 0
        return datas

    # transform velocity into valid type
    # eg: NaN -> 0, float -> int
    def transformer(datas):
        for row in range(len(datas)):
            for col in range(len(datas[row])):
                if np.isnan(datas[row][col]):
                    datas[row][col] = 0
                else:
                    datas[row][col] = math.floor(datas[row][col])
        return datas

    # classify objects by calculating their probability to be a bird/airplane
    def classify(pdf1, datas):
        # initial probability(t=0) for object to be bird/airplane
        proba_a = 0.5
        proba_b = 0.5
        final_probability = [0 for m in range(2 * len(datas))]
        for row in range(len(datas)):
            # defualt transition between classes
            transition = 0.1
            for t in range(len(datas[0])):
                if t == 0:
                    # pobb -> probability of being a bird
                    # poba -> probability of being a airplane
                    probb = proba_b
                    proba = proba_a
                if datas[row][t] == 0:
                    continue
                pb = probb
                pa = proba
                # recursive bayesian estimator
                if EXTRA_FEATURE == True:
                    if abs(datas[row][t] - datas[row][t - 1] > 0):
                        probb = pdf1[0][int(2 * datas[row][t])] * (pb * (transition - 0.1) + pa * (1 - transition + 0.1))
                        proba = pdf1[1][int(2 * datas[row][t])] * (pa * transition + pb * (1 - transition))
                else:
                    probb = pdf1[0][int(2 * datas[row][t])] * (pb * transition + pa * (1 - transition))
                    proba = pdf1[1][int(2 * datas[row][t])] * (pa * transition + pb * (1 - transition))
                # normalize
                sump = probb + proba
                probb = probb / sump
                proba = proba / sump
                final_probability[2 * row] = probb
                final_probability[2 * row + 1] = proba
        return final_probability

    def run(self):
        n1 = 400
        n2 = 300

        self.pdf = radar.cleaner(self.pdf, n1)
        self.pdf = radar.normalizer(self.pdf)

        self.vel = radar.cleaner(self.vel, n2)
        self.vel = radar.transformer(self.vel)

        pro1 = radar.classify(self.pdf, self.vel)
        print("Object Number", "|", "Probability of Being Airplane", "|", "Probability of Being Bird", "|",
              "Conclusion")
        for number in range(int(len(pro1) / 2)):
            print("No.", number + 1, end="")
            print("\t\t\t\t\t %.3f" %pro1[2 * number + 1], end="")
            print("\t\t\t\t\t\t\t %.3f" %pro1[2 * number], end="")
            if pro1[2 * number] >= pro1[2 * number + 1]:
                print("\t\t\t\t\tbird")
            else:
                print("\t\t\t\t\tairplane")
if __name__ == '__main__':
    msg = input("Do you want to add extra feature?[y/n]: ")
    if msg == "y":
        EXTRA_FEATURE = True
    else:
        EXTRA_FEATURE = False

    radar = Radar
    radar().run()







