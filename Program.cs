
using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLTrainingInappContent
{
    public class TextData
    {
	public string Text { get; set; }
	public bool IsInappropriate { get; set; }
    }
    public class Prediction
    {
	[ColumnName("PredictedLabel")]
	public bool IsInappropriate { get; set; }
	public float Probability { get; set; }
	public float Score { get; set; }
    }
    class Program
    {
	static void Main(string[] args)
	{
	    var mlContext = new MLContext();

	    //step 1 : Load the data
	    var trainingDataPath = "content_data.csv";
	    var data = mlContext.Data.LoadFromTextFile<TextData>(trainingDataPath, separatorChar:',', hasHeader:true);

	    //step 2: Define the pipeline
	    var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextData.Text))
		.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
		    labelColumnName: nameof(TextData.IsInappropriate),
		    featureColumnName: "Features"));

	    //step 3 : Train the model
	    var model = pipeline.Fit(data);


	    //Step 4: Save the model
	    var modelPath = "ContentDetectionModel.Zip";
	    mlContext.Model.Save(model, data.Schema, modelPath);

	    Console.WriteLine($"Model trained and saved to {modelPath}");

	}
    }
}