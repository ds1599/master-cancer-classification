package org.example;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

public class LogisticRegressionClassifierExample {
    public static void main(String[] args) throws Exception {

        // Load the dataset
        BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/breast-cancer-wisconsin.arff"));
        Instances dataset = new Instances(reader);
        reader.close();

        // Set the class attribute as the last attribute
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Split the dataset into training and testing sets (80:20 split)
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        Instances trainSet = new Instances(dataset, 0, trainSize);
        Instances testSet = new Instances(dataset, trainSize, testSize);

        // Build and evaluate the classifier
        Logistic classifier = new Logistic();
        classifier.buildClassifier(trainSet);
        Evaluation evaluation = new Evaluation(trainSet);
        evaluation.evaluateModel(classifier, testSet);

        // Print the evaluation results
        System.out.println("=== # of Instances  ===");
        System.out.println(dataset.numInstances());
        System.out.println("=== # of Train Set  ===");
        System.out.println(trainSize);
        System.out.println("=== # of Test Set  ===");
        System.out.println(testSize);

        System.out.println("=== Logistic Regression Evaluation Results ===");
        System.out.println(evaluation.toSummaryString());

        System.out.println(evaluation.toClassDetailsString());
        System.out.println(evaluation.toMatrixString());
    }
}