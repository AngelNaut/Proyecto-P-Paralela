using RedNeuronal_ProyectoFinal.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RedNeuronal_ProyectoFinal.Services
{
    public class NeuralNetworkParallel
    {
        private Layer[] Layers;
        private double LearningRate;

        public NeuralNetworkParallel(int inputSize, int hiddenSize, int outputSize, double learningRate = 0.1)
        {
            Layers = new Layer[]
            {
                new Layer(hiddenSize, inputSize),
                new Layer(outputSize, hiddenSize)
            };
            LearningRate = learningRate;
        }

        
        // Propagación hacia adelante con salidas por capa.
        
        public List<double[]> ForwardWithOutputs(double[] inputs)
        {
            var outputs = new List<double[]>();

            foreach (var layer in Layers)
            {
                double[] currentOutput = new double[layer.Neurons.Length];

                for (int i = 0; i < layer.Neurons.Length; i++)
                    currentOutput[i] = layer.Neurons[i].Activate(inputs);

                outputs.Add(currentOutput);
                inputs = currentOutput;
            }

            return outputs;
        }

        // Entrenamiento con mini-batches y paralelismo por muestra.
        public void TrainWithMiniBatches(double[][] inputs, double[][] targets, int epochs, int batchSize)
        {
            int sampleCount = inputs.Length;
            int batchCount = (int)Math.Ceiling(sampleCount / (double)batchSize);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;
                object errorLock = new object();

                for (int b = 0; b < batchCount; b++)
                {
                    int start = b * batchSize;
                    int end = Math.Min(start + batchSize, sampleCount);

                    // Inicializar acumuladores de gradientes
                    Gradient[][] gradients = Layers.Select(layer =>
                        layer.Neurons.Select(n => new Gradient(n.Weights.Length)).ToArray()
                    ).ToArray();

                    Parallel.For(start, end, i =>
                    {
                        double[] currentInputs = inputs[i];
                        double[] target = targets[i];

                        var layerOutputs = ForwardWithOutputs(currentInputs);
                        double[] output = layerOutputs[^1];

                        double sampleError = 0;
                        for (int j = 0; j < output.Length; j++)
                            sampleError += Math.Pow(target[j] - output[j], 2);

                        lock (errorLock)
                        {
                            totalError += sampleError;
                        }

                        double[][] deltas = new double[Layers.Length][];
                        for (int l = 0; l < Layers.Length; l++)
                            deltas[l] = new double[Layers[l].Neurons.Length];

                        for (int l = Layers.Length - 1; l >= 0; l--)
                        {
                            double[] currentOutputs = layerOutputs[l];
                            double[] prevOutputs = (l == 0) ? currentInputs : layerOutputs[l - 1];

                            for (int n = 0; n < Layers[l].Neurons.Length; n++)
                            {
                                var neuron = Layers[l].Neurons[n];

                                if (l == Layers.Length - 1)
                                {
                                    deltas[l][n] = (target[n] - currentOutputs[n]) * Neuron.SigmoidDerivative(currentOutputs[n]);
                                }
                                else
                                {
                                    double sum = 0;
                                    for (int k = 0; k < Layers[l + 1].Neurons.Length; k++)
                                        sum += Layers[l + 1].Neurons[k].Weights[n] * deltas[l + 1][k];

                                    deltas[l][n] = sum * Neuron.SigmoidDerivative(currentOutputs[n]);
                                }

                                lock (gradients[l][n])
                                {
                                    for (int w = 0; w < neuron.Weights.Length; w++)
                                        gradients[l][n].WeightGradients[w] += LearningRate * deltas[l][n] * prevOutputs[w];

                                    gradients[l][n].BiasGradient += LearningRate * deltas[l][n];
                                }
                            }
                        }
                    });

                    for (int l = 0; l < Layers.Length; l++)
                    {
                        for (int n = 0; n < Layers[l].Neurons.Length; n++)
                        {
                            gradients[l][n].ApplyTo(Layers[l].Neurons[n]);
                        }
                    }
                }

                Console.WriteLine($"[MiniBatch] Epoch {epoch + 1}, Error: {totalError / sampleCount}");
            }
        }
    }
}
