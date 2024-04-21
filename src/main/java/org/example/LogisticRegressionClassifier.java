package org.example;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class LogisticRegressionClassifier implements Classifier {

    private final Logistic logisticRegression;

    public LogisticRegressionClassifier() {
        this.logisticRegression = new Logistic();
    }

    @Override
    public String getClassifierInfo() {
        return "Logistic Regression Classifier";
    }

    @Override
    public void train(Instances trainSet) throws Exception {

        if (trainSet.numInstances() == 0) {
            throw new IllegalArgumentException("Training set is empty.");
        }

        // Calculate the sum of values for the class attribute
        double sum = 0.0;
        for (int i = 0; i < trainSet.numInstances(); i++) {
            sum += trainSet.instance(i).classValue();
        }

        // Check if the sum of values in the dataset is zero
        if (sum == 0.0) {
            System.out.println("Warning: Can't normalize array. Sum is zero. Skipping normalization.");
        } else {
            // Normalize the dataset
            Normalize normalizeFilter = new Normalize();
            normalizeFilter.setInputFormat(trainSet);
            trainSet = Filter.useFilter(trainSet, normalizeFilter);
        }

        // Train the classifier
        logisticRegression.buildClassifier(trainSet);
    }

    @Override
    public Evaluation evaluate(Instances testSet) throws Exception {
        Evaluation evaluation = new Evaluation(testSet);
        evaluation.evaluateModel(logisticRegression, testSet);
        return evaluation;
    }
}
