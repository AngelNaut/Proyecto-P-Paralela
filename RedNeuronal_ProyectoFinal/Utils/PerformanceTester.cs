using RedNeuronal_ProyectoFinal.Services;
using System.Diagnostics;
using System.Linq;

namespace RedNeuronal_ProyectoFinal.Utils
{
    public static class PerformanceTester
    {
        public static void TestNetworks(
            double[][] trainInputs, double[][] trainTargets,
            double[][] testInputs, double[][] testTargets,
            int epochs, int inputSize, int hiddenSize, int outputSize)
        {
            Console.WriteLine("\n=== ENTRENANDO RED SECUENCIAL ===\n");

            var sequentialNN = new NeuralNetworkSequential(inputSize, hiddenSize, outputSize);
            var watchSeq = Stopwatch.StartNew();
            sequentialNN.Train(trainInputs, trainTargets, epochs);
            watchSeq.Stop();

            double accSeq = EvaluationHelper.EvaluateAccuracy(sequentialNN.Forward, testInputs, testTargets);
            Console.WriteLine($"[Secuencial] Tiempo total: {watchSeq.ElapsedMilliseconds} ms");
            Console.WriteLine($"[Secuencial] Precisión: {accSeq:P2}");

            Console.WriteLine("\n=== ENTRENANDO RED PARALELA CON MINI-BATCHES ===\n");

            var parallelNN = new NeuralNetworkParallel(inputSize, hiddenSize, outputSize);
            var watchPar = Stopwatch.StartNew();
            parallelNN.TrainWithMiniBatches(trainInputs, trainTargets, epochs, batchSize: 32);
            watchPar.Stop();

            // ✅ Usamos una función anónima para extraer solo la salida final
            double accPar = EvaluationHelper.EvaluateAccuracy(
                input => parallelNN.ForwardWithOutputs(input).Last(),
                testInputs, testTargets
            );

            Console.WriteLine($"[Paralelo] Tiempo total: {watchPar.ElapsedMilliseconds} ms");
            Console.WriteLine($"[Paralelo] Precisión: {accPar:P2}");
        }
    }
}

