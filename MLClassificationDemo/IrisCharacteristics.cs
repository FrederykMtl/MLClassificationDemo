using Microsoft.ML.Data;

namespace MLClassificationDemo
{
    public class IrisCharacteristics
    {
        [LoadColumn(0)]
        public string? sepal_length { get; set; }

        [LoadColumn(1)]
        public string? sepal_width { get; set; }

        [LoadColumn(2)]
        public string? petal_length { get; set; }

        [LoadColumn(3)]
        public string? petal_width { get; set; }

        [LoadColumn(4)]
        public string? species { get; set; }
    }

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string? species;
    }
}
