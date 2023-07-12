using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal class Data
    {
        public int inputsLength;
        public double[] inputs;
        public double[] targetOutput;
        public Data(int length, double[] inputs, double[] targets) 
        {
            this.inputsLength = length;
            this.inputs = inputs;
            this.targetOutput= targets;
        }
    }

    internal class ParityData : Data
    {
        public ParityData(int bitLength, double[] bits, double[] targetOutput) : base(bitLength, bits, targetOutput) { }
    }

    internal class FruitData : Data
    {
        public FruitData(int numAttributes, double[] fruitAttributes, double[] targetOutput) : base(numAttributes, fruitAttributes, targetOutput) { }
    }
}
