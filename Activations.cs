using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal static class Activations
    {
        public static double Linear(double input) { return input; }
        public static double ReLU(double input) { return Math.Max(0, input); }
        public static double Step(double input) { return input > 0 ? 1 : 0; }
        public static double Sigmoid(double input) { return (1 / (1 + Math.Pow(Math.E, -1 * input))); }
    }
}
