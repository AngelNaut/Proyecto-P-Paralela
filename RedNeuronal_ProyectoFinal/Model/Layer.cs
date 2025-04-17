using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RedNeuronal_ProyectoFinal.Model
{
 
    // Representa una capa de la red neuronal compuesta por varias neuronas.

    public class Layer
    {
        public Neuron[] Neurons;

        public Layer(int neuronCount, int inputSize)
        {
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
                Neurons[i] = new Neuron(inputSize);
        }

        
        // Calcula las salidas de la capa para un conjunto dado de entradas.
        // En esta implementación secuencial se usa LINQ; 
        // la paralelización se aplicará en la versión de red.
        
        public virtual double[] ComputeOutputs(double[] inputs)
        {
            return Neurons.Select(neuron => neuron.Activate(inputs)).ToArray();
        }
    }
}
