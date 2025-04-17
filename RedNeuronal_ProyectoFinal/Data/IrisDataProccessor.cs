using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using CsvHelper;  // Asegúrate de instalar CsvHelper mediante NuGet

namespace RedNeuronal_ProyectoFinal.Data
{
    internal class IrisDataProccessor
    {
        
        // Se encarga de cargar, normalizar y transformar el dataset Iris.
        
        public static class IrisDataProcessor
        {
            // Carga el archivo CSV y retorna una lista de IrisSample.
            
            public static List<IrisSample> LoadData(string filePath)
            {
                using (var reader = new StreamReader(filePath))
                using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
                {
                    var records = csv.GetRecords<IrisSample>().ToList();
                    return records;
                }
            }

            // Normaliza las características de las muestras al rango [0,1].
            // Se calcula por columna (característica).
            
            public static void NormalizeData(List<IrisSample> data)
            {
                // Extraer listas por cada característica.
                double[] sepalLengths = data.Select(s => s.SepalLength).ToArray();
                double[] sepalWidths = data.Select(s => s.SepalWidth).ToArray();
                double[] petalLengths = data.Select(s => s.PetalLength).ToArray();
                double[] petalWidths = data.Select(s => s.PetalWidth).ToArray();

                // Calcular min y max para cada columna.
                double minSepalLength = sepalLengths.Min(), maxSepalLength = sepalLengths.Max();
                double minSepalWidth = sepalWidths.Min(), maxSepalWidth = sepalWidths.Max();
                double minPetalLength = petalLengths.Min(), maxPetalLength = petalLengths.Max();
                double minPetalWidth = petalWidths.Min(), maxPetalWidth = petalWidths.Max();

                // Normalizar cada muestra.
                foreach (var sample in data)
                {
                    sample.SepalLength = (sample.SepalLength - minSepalLength) / (maxSepalLength - minSepalLength);
                    sample.SepalWidth = (sample.SepalWidth - minSepalWidth) / (maxSepalWidth - minSepalWidth);
                    sample.PetalLength = (sample.PetalLength - minPetalLength) / (maxPetalLength - minPetalLength);
                    sample.PetalWidth = (sample.PetalWidth - minPetalWidth) / (maxPetalWidth - minPetalWidth);
                }
            }

            // Convierte la especie en un vector one-hot (3 dimensiones).
            // Asume que las especies son: "Iris-setosa", "Iris-versicolor", "Iris-virginica".

            public static double[] GetOneHotVector(string species)
            {
                return species switch
                {
                    "Setosa" => new double[] { 1, 0, 0 },
                    "Versicolor" => new double[] { 0, 1, 0 },
                    "Virginica" => new double[] { 0, 0, 1 },
                    _ => throw new ArgumentException("Especie desconocida")
                };
            }

            // Divide el conjunto de datos en training y test según un porcentaje dado.

            public static (List<IrisSample> training, List<IrisSample> test) SplitData(List<IrisSample> data, double trainingPercentage)
            {
                var shuffled = data.OrderBy(x => Guid.NewGuid()).ToList();
                int trainingCount = (int)(shuffled.Count * trainingPercentage);
                return (shuffled.Take(trainingCount).ToList(), shuffled.Skip(trainingCount).ToList());
            }
        }
    }
}
