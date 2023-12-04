package org.example;

import com.sun.xml.internal.bind.v2.TODO;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.FileInputStream;

public class Main {
    public static void main(String[] args) throws Exception {

        ArffLoader loader = new ArffLoader();
// CSV Loader https://weka.sourceforge.io/doc.dev/weka/core/converters/CSVLoader.html
        CSVLoader csvLoader = new CSVLoader();

        csvLoader.setSource(new File("data.csv"));
        Instances trainingDataSet = csvLoader.getDataSet();
        trainingDataSet.setClassIndex(trainingDataSet.numAttributes() - 1);

        MultilayerPerceptron mlp = new MultilayerPerceptron();
        //Setting Parameters
        mlp.setLearningRate(0.1);
        mlp.setMomentum(0.2);
        mlp.setTrainingTime(2000);
        mlp.setHiddenLayers("a");
        mlp.buildClassifier(trainingDataSet);
        Evaluation eval = new Evaluation(trainingDataSet);
        eval.evaluateModel(mlp, trainingDataSet);
        System.out.println(eval.errorRate()); //Printing Training Mean root squared Error
        System.out.println(eval.toSummaryString()); //Summary of Training
        //TODO: try to fix the warning in the console

    }
}
