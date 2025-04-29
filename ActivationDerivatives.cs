using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal class ActivationDerivatives
    {
        public static double Linear(double input) { return 1; }
        public static double ReLU(double input) { return Activations.Step(input); }
        public static double Step(double input) { return 0; }
        public static double PreSigmoid(double input) { return (Activations.Sigmoid(input) * (1 - Activations.Sigmoid(input))); }
        public static double PostSigmoid(double input) { return (input * (1 - input)); }
    }
}
