using RedNeuronal_ProyectoFinal.Data;
using RedNeuronal_ProyectoFinal.Utils;
using static RedNeuronal_ProyectoFinal.Data.IrisDataProccessor;

namespace RedNeuronal_ProyectoFinal
{
    public class Program
    {
        static void Main()
        {
            Console.WriteLine("Seleccione el dataset:");
            Console.WriteLine("1. Iris");
            Console.WriteLine("2. Letter Recognition");
            Console.Write("Opción: ");
            string opcion = Console.ReadLine();

            switch (opcion)
            {
                case "1":
                    EjecutarIris();
                    break;
                case "2":
                    EjecutarLetter();
                    break;
                default:
                    Console.WriteLine("Opción inválida.");
                    break;
            }

            Console.WriteLine("Entrenamiento completado. Presiona cualquier tecla para salir.");
            Console.ReadKey();
        }

        static void EjecutarIris()
        {
            string filePath = "iris.csv";

            var data = IrisDataProcessor.LoadData(filePath);
            IrisDataProcessor.NormalizeData(data);
            var (trainingData, testData) = IrisDataProcessor.SplitData(data, 0.8);

            double[][] trainInputs = trainingData.Select(s => s.GetFeatures()).ToArray();
            double[][] trainTargets = trainingData.Select(s => IrisDataProcessor.GetOneHotVector(s.Species)).ToArray();
            double[][] testInputs = testData.Select(s => s.GetFeatures()).ToArray();
            double[][] testTargets = testData.Select(s => IrisDataProcessor.GetOneHotVector(s.Species)).ToArray();

            int inputSize = 4;
            int hiddenSize = 4;
            int outputSize = 3;
            int epochs = 500;

            PerformanceTester.TestNetworks(trainInputs, trainTargets, testInputs, testTargets, epochs, inputSize, hiddenSize, outputSize);
        }

        static void EjecutarLetter()
        {
            string filePath = "letter-recognition.csv";

            var data = LetterDataProcessor.LoadData(filePath);
            LetterDataProcessor.NormalizeData(data);
            var (trainingData, testData) = LetterDataProcessor.SplitData(data, 0.8);

            double[][] trainInputs = trainingData.Select(s => s.GetFeatures()).ToArray();
            double[][] trainTargets = trainingData.Select(s => LetterDataProcessor.GetOneHotVector(s.Letter)).ToArray();
            double[][] testInputs = testData.Select(s => s.GetFeatures()).ToArray();
            double[][] testTargets = testData.Select(s => LetterDataProcessor.GetOneHotVector(s.Letter)).ToArray();

            int inputSize = 16;
            int hiddenSize = 32;
            int outputSize = 26;
            int epochs = 100;

            PerformanceTester.TestNetworks(trainInputs, trainTargets, testInputs, testTargets, epochs, inputSize, hiddenSize, outputSize);
        }
    }
}
