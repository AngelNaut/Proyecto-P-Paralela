using CsvHelper;
using System.Globalization;

namespace RedNeuronal_ProyectoFinal.Data
{
    public static class LetterDataProcessor
    {
        public static List<LetterSample> LoadData(string filePath)
        {
            using var reader = new StreamReader(filePath);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            return csv.GetRecords<LetterSample>().ToList();
        }

        public static void NormalizeData(List<LetterSample> data)
        {
            int featureCount = 16;
            double[] minValues = new double[featureCount];
            double[] maxValues = new double[featureCount];

            for (int i = 0; i < featureCount; i++)
            {
                var column = data.Select(s => s.GetFeatures()[i]).ToArray();
                minValues[i] = column.Min();
                maxValues[i] = column.Max();
            }

            foreach (var sample in data)
            {
                double[] features = sample.GetFeatures();
                for (int i = 0; i < featureCount; i++)
                {
                    double range = maxValues[i] - minValues[i];
                    if (range != 0)
                        features[i] = (features[i] - minValues[i]) / range;
                }

                // Sobrescribe las propiedades
                var props = typeof(LetterSample).GetProperties().Where(p => p.Name.StartsWith("Feature"));
                int idx = 0;
                foreach (var prop in props)
                {
                    prop.SetValue(sample, features[idx++]);
                }
            }
        }

        public static double[] GetOneHotVector(string letter)
        {
            int index = char.ToUpper(letter[0]) - 'A';
            double[] oneHot = new double[26];
            oneHot[index] = 1;
            return oneHot;
        }

        public static (List<LetterSample> training, List<LetterSample> test) SplitData(List<LetterSample> data, double trainingPercentage)
        {
            var shuffled = data.OrderBy(_ => Guid.NewGuid()).ToList();
            int trainingCount = (int)(shuffled.Count * trainingPercentage);
            return (shuffled.Take(trainingCount).ToList(), shuffled.Skip(trainingCount).ToList());
        }
    }
}
