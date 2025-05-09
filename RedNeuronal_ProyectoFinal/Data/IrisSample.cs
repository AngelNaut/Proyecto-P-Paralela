﻿using CsvHelper.Configuration.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RedNeuronal_ProyectoFinal.Data
{
 // Representa una muestra del dataset Iris.
 // Incluye las 4 características y la especie.

    public class IrisSample
    {
        [Name("sepal.length")]
        public double SepalLength { get; set; }

        [Name("sepal.width")]
        public double SepalWidth { get; set; }

        [Name("petal.length")]
        public double PetalLength { get; set; }

        [Name("petal.width")]
        public double PetalWidth { get; set; }

        [Name("variety")]
        public string Species { get; set; }

        // Retorna las características en un arreglo de double.
        public double[] GetFeatures() => new double[] { SepalLength, SepalWidth, PetalLength, PetalWidth };
    }
}
