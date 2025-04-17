namespace RedNeuronal_ProyectoFinal.Model
{
    // Representa los gradientes acumulados para una neurona durante un mini-batch.

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
