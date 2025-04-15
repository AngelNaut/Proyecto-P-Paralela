using System;
using System.Linq;
using RedNeuronal_ProyectoFinal.Model;

namespace RedNeuronal_ProyectoFinal.Utils
{
    public static class EvaluationHelper
    {
        public static double EvaluateAccuracy(Func<double[], double[]> predictFunc, double[][] inputs, double[][] targets)
        {
            int correct = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = predictFunc(inputs[i]);
                int predictedIndex = Array.IndexOf(prediction, prediction.Max());
                int targetIndex = Array.IndexOf(targets[i], targets[i].Max());

                if (predictedIndex == targetIndex)
                    correct++;
            }

            return (double)correct / inputs.Length;
        }
    }
}
