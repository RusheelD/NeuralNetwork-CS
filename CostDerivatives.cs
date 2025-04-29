using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal class CostDerivatives
    {
        public static double SquaredDifferenceDerivative(double output, double expected)
        {
            return 2 * (output - expected);
        }
    }
}
