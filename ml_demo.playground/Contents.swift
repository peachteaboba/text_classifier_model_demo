import Cocoa
import CreateML

// Import data
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/andyfeng/Desktop/ml_test/test_data.json"))

// Split data into training and testing groups
let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

// Create text classifier using training data
let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "label")

// Training accuracy as a percentage
let trainingAccuracy = (1.0 - sentimentClassifier.trainingMetrics.classificationError) * 100

// Validation accuracy as a percentage
let validationAccuracy = (1.0 - sentimentClassifier.validationMetrics.classificationError) * 100

let evaluationMetrics = sentimentClassifier.evaluation(on: testingData, textColumn: "text", labelColumn: "label")
// Evaluation accuracy as a percentage
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

// Log metrics
print("---------------------------------------------")
print("trainingAccuracy -> \(trainingAccuracy)%")
print("validationAccuracy -> \(validationAccuracy)%")
print("evaluationAccuracy -> \(evaluationAccuracy)%")
print("---------------------------------------------")

if evaluationAccuracy > 99.9 {
    print("OK to save!")
    let metadata = MLModelMetadata(
        author: "Andy Feng",
        shortDescription: "A model trained to classify movie review sentiment",
        license: nil,
        version: "1.0",
        additional: ["name" : "Focus on Science"]
    )
    
    try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/andyfeng/Desktop/ml_test/models/SentimentClassifier.mlmodel"), metadata: metadata)
}


