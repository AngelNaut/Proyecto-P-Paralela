using RedNeuronal_ProyectoFinal.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RedNeuronal_ProyectoFinal.Model
{
    /// <summary>
    /// Representa una neurona de la red neuronal.
    /// Contiene los pesos, bias, salida y delta para la retropropagación.
    /// </summary>
    public class Neuron
    {
        public double[] Weights;  // Pesos de cada entrada
        public double Bias;       // Valor del bias
        public double Output;     // Salida de la neurona tras activación
        public double Delta;      // Delta usado en la retropropagación

        private static Random rand = new Random();

        public Neuron(int inputSize)
        {
            Weights = new double[inputSize];
            for (int i = 0; i < inputSize; i++)
                Weights[i] = rand.NextDouble() * 2 - 1; // Inicializa en el rango [-1,1]
            Bias = rand.NextDouble() * 2 - 1;
        }

        /// <summary>
        /// Calcula la salida de la neurona dada una entrada y aplica la función sigmoide.
        /// </summary>
        public double Activate(double[] inputs)
        {
            double sum = 0;
            for (int i = 0; i < Weights.Length; i++)
                sum += inputs[i] * Weights[i];
            sum += Bias;
            return Sigmoid(sum); // ✅ no toca `Output`
        }

        /// <summary>
        /// Función de activación sigmoide.
        /// </summary>
        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        /// <summary>
        /// Derivada de la función sigmoide para el cálculo del delta.
        /// </summary>
        public static double SigmoidDerivative(double output) => output * (1 - output);
    }
}

