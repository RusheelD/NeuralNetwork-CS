using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal static class Costs
    {
        public static double SquaredDifference(double output, double expected)
        {
            return Math.Pow((output - expected), 2);
        }

        public static double SquaredDifferences(double[] output, double[] expected)
        {
            double cost = 0;
            for(int i = 0; i < output.Length; i++)
            {
                cost += SquaredDifference(output[i], expected[i]);
            }

            return cost;
        }
        
        public static double AvgSquaredDifference(double[][] outputs, double[][] expected)
        {
            double cost = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                cost += SquaredDifferences(outputs[i], expected[i]);
            }

            return cost / outputs.Length;
        }
    }
}
