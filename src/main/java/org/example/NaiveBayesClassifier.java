package org.example;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class NaiveBayesClassifier implements Classifier {

    private final NaiveBayes naiveBayes;

    public NaiveBayesClassifier() {
        this.naiveBayes = new NaiveBayes();
    }

    @Override
    public String getClassifierInfo() {
        return "Naive Bayes Classifier";
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
        naiveBayes.buildClassifier(trainSet);
    }

    @Override
    public Evaluation evaluate(Instances testSet) throws Exception {
        Evaluation evaluation = new Evaluation(testSet);
        evaluation.evaluateModel(naiveBayes, testSet);
        return evaluation;
    }
}
