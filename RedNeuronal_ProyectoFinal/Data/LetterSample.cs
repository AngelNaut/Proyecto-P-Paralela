using CsvHelper.Configuration.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RedNeuronal_ProyectoFinal.Data
{
    public class LetterSample
    {
        [Index(0)]
        public string Letter { get; set; }

        [Index(1)] public double Feature1 { get; set; }
        [Index(2)] public double Feature2 { get; set; }
        [Index(3)] public double Feature3 { get; set; }
        [Index(4)] public double Feature4 { get; set; }
        [Index(5)] public double Feature5 { get; set; }
        [Index(6)] public double Feature6 { get; set; }
        [Index(7)] public double Feature7 { get; set; }
        [Index(8)] public double Feature8 { get; set; }
        [Index(9)] public double Feature9 { get; set; }
        [Index(10)] public double Feature10 { get; set; }
        [Index(11)] public double Feature11 { get; set; }
        [Index(12)] public double Feature12 { get; set; }
        [Index(13)] public double Feature13 { get; set; }
        [Index(14)] public double Feature14 { get; set; }
        [Index(15)] public double Feature15 { get; set; }
        [Index(16)] public double Feature16 { get; set; }

        public double[] GetFeatures() => new double[]
        {
            Feature1, Feature2, Feature3, Feature4,
            Feature5, Feature6, Feature7, Feature8,
            Feature9, Feature10, Feature11, Feature12,
            Feature13, Feature14, Feature15, Feature16
        };
    }
}
