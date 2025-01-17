using Microsoft.ML;
using MLClassificationDemo;

string? _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";

if (!string.IsNullOrEmpty(_appPath))
    _appPath = Directory.GetParent(_appPath)?.Parent?.Parent?.ToString();
else
    throw new Exception();

string _trainDataPath = Path.Combine(_appPath, "Data", "IRIS_traindata.csv");
string _testDataPath = Path.Combine(_appPath,  "Data", "IRIS_testdata.csv");

MLContext _mlContext;
PredictionEngine<IrisCharacteristics, IssuePrediction> _predEngine;
ITransformer _trainedModel;
IDataView _trainingDataView;

_mlContext = new MLContext(seed: 0);
_trainingDataView = _mlContext.Data.LoadFromTextFile<IrisCharacteristics>(_trainDataPath, ',', hasHeader: true);
Console.WriteLine("Data Loaded");

var _pipeline = ProcessData();
Console.WriteLine("Data Processed");

BuildAndTrainModel(_trainingDataView, _pipeline);
Console.WriteLine("Model trained");

Evaluate();
Console.WriteLine("Model evaluated");

IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "species", outputColumnName: "Label")
        .Append(_mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "sepal_length", outputColumnName: "SepLengthFeaturized"))
        .Append(_mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "sepal_width", outputColumnName: "SepWidthFeaturized"))
        .Append(_mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "petal_length", outputColumnName: "PetLengthFeaturized"))
        .Append(_mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "petal_width", outputColumnName: "PetWidthFeaturized"))
        .Append(_mlContext.Transforms.Concatenate("Features", "SepLengthFeaturized", "SepWidthFeaturized", "PetLengthFeaturized", "PetWidthFeaturized"))
        .AppendCacheCheckpoint(_mlContext);

    return pipeline;
}

IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    _trainedModel = trainingPipeline.Fit(trainingDataView);

    _predEngine = _mlContext.Model.CreatePredictionEngine<IrisCharacteristics, IssuePrediction>(_trainedModel);

    var testIris = new IrisCharacteristics() { petal_length = "1", petal_width = "0.2", sepal_length = "4", sepal_width = "3" };

    var prediction = _predEngine.Predict(testIris);
    Console.WriteLine($"Iris for prediction: sepal_length = 4, sepal_width = 3 petal_length = 1, petal_width = 0.2");
    Console.WriteLine($"== Single Prediction just-trained-model - Result: {prediction.species} ==");

    return trainingPipeline;
}

void Evaluate()
{
    var testDataView = _mlContext.Data.LoadFromTextFile<IrisCharacteristics>(_testDataPath, ',', hasHeader: true);
    var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");
}