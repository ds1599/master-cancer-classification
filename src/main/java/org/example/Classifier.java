package org.example;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public interface Classifier {

    String getClassifierInfo();

    void train(Instances trainSet) throws Exception;
    Evaluation evaluate(Instances testSet) throws Exception;
}
