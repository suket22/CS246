def test_accuracy(filename):
    y = []  # Actual values
    yhat = []  # Predicted values
    truep, truen = 0, 0
    falsep, falsen = 0, 0

    with open(filename) as f:  # File we create in test_data()
        for line in f:
            yhat.append(line.split()[2])
    with open('duplicate_sample.out') as f:  # Answers file given to us by Quora
        for line in f:
            y.append(line.split()[2])

    correct = 0
    for i in range(0, len(y)):
        if y[i] == yhat[i] and y[i] == "1":
            truep += 1
        elif y[i] == yhat[i] and y[i] == "0":
            truen += 1
        elif y[i] != yhat[i] and yhat[i] == "1":
            falsep += 1
        else:
            falsen += 1
    print "True Positive - ", truep
    print "True Negative - ", truen
    print "False Positive - ", falsep
    print "False Negative - ", falsen
    precision = truep * 1.0 / (truep + falsep)
    recall = truep * 1.0 / (truep + falsen)
    f_score = (precision * recall * 2) / (precision + recall)
    accuracy = (truep + truen) * 1.0 / len(y)
    print "Precision is", precision
    print "Recall is", recall
    print "F Score is", f_score
    print "Accuracy is ", accuracy
