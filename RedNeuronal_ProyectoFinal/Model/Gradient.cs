namespace RedNeuronal_ProyectoFinal.Model
{
    /// <summary>
    /// Representa los gradientes acumulados para una neurona durante un mini-batch.
    /// </summary>
    public class Gradient
    {
        public double[] WeightGradients;
        public double BiasGradient;

        public Gradient(int size)
        {
            WeightGradients = new double[size];
            BiasGradient = 0;
        }

        public void ApplyTo(Neuron neuron)
        {
            for (int i = 0; i < WeightGradients.Length; i++)
            {
                neuron.Weights[i] += WeightGradients[i];
            }
            neuron.Bias += BiasGradient;
        }
    }
}
