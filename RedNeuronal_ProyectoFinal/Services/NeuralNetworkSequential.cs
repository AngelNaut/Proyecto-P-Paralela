using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RedNeuronal_ProyectoFinal.Model;

namespace RedNeuronal_ProyectoFinal.Services
{
    public class NeuralNetworkSequential
    {
        private Layer[] Layers;
        private double LearningRate;

        public NeuralNetworkSequential(int inputSize, int hiddenSize, int outputSize, double learningRate = 0.1)
        {
            Layers = new Layer[]
            {
                new Layer(hiddenSize, inputSize),
                new Layer(outputSize, hiddenSize)
            };
            LearningRate = learningRate;
        }

        public double[] Forward(double[] inputs)
        {
            foreach (var layer in Layers)
                inputs = layer.ComputeOutputs(inputs);
            return inputs;
        }

        public void Train(double[][] inputs, double[][] targets, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    var outputsByLayer = ForwardWithOutputs(inputs[i]);
                    var output = outputsByLayer[^1];

                    totalError += output.Zip(targets[i], (o, t) => Math.Pow(t - o, 2)).Sum();

                    Backpropagation(inputs[i], targets[i], outputsByLayer);
                }

                Console.WriteLine($"[Secuencial] Epoch {epoch + 1}, Error: {totalError / inputs.Length}");
            }
        }

        private List<double[]> ForwardWithOutputs(double[] inputs)
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

        private void Backpropagation(double[] inputs, double[] targets, List<double[]> outputsByLayer)
        {
            for (int i = Layers.Length - 1; i >= 0; i--)
            {
                var layer = Layers[i];
                double[] currentOutputs = outputsByLayer[i];
                double[] prevOutputs = (i == 0) ? inputs : outputsByLayer[i - 1];

                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    var neuron = layer.Neurons[j];

                    if (i == Layers.Length - 1)
                        neuron.Delta = (targets[j] - currentOutputs[j]) * Neuron.SigmoidDerivative(currentOutputs[j]);
                    else
                        neuron.Delta = Layers[i + 1].Neurons
                            .Select((nextNeuron, k) => nextNeuron.Weights[j] * nextNeuron.Delta)
                            .Sum() * Neuron.SigmoidDerivative(currentOutputs[j]);

                    for (int k = 0; k < neuron.Weights.Length; k++)
                        neuron.Weights[k] += LearningRate * neuron.Delta * prevOutputs[k];

                    neuron.Bias += LearningRate * neuron.Delta;
                }
            }
        }
    }
}

