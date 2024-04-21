package org.example;

import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

public class ClassifierMain {

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

        // Instantiate classifiers
        Classifier[] classifiers = {
                new NaiveBayesClassifier(),
                new KNearestNeighborClassifier(),
                new LogisticRegressionClassifier()
        };

        // Train and evaluate each classifier
        for (Classifier classifier : classifiers) {
            classifier.train(trainSet);
            Evaluation evaluation = classifier.evaluate(testSet);

            // Print the evaluation results
            System.out.println("=== Evaluation Results for " + classifier.getClassifierInfo() + " ===");
            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());

        }
    }
}
