package org.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class NaiveBayesClassifierExample {

    public static void main(String[] args) throws Exception {
        System.out.println(new File("iris.arff").getAbsolutePath());

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
        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(trainSet);
        Evaluation evaluation = new Evaluation(trainSet);
        evaluation.evaluateModel(classifier, testSet);

        // Print the evaluation results
        System.out.println("=== Evaluation Results ===");
        System.out.println(evaluation.toSummaryString());

        System.out.println(evaluation.toClassDetailsString());
        System.out.println(evaluation.toMatrixString());
    }
}
