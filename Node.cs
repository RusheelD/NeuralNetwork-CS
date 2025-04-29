using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal class Node
    {
        public int numInputs { get; private set; }
        public double value { get; private set; }
        public double[] weights { get; set; }
        public double bias { get; set; }

        public Node(int numInputs) 
        { 
            this.numInputs = numInputs;
            this.value = 0;
            this.weights = new double[numInputs];
            Random random = new Random();

            for (int i = 0; i < numInputs;i++)
            {
                this.weights[i] = 2 * random.NextDouble() - 1;
            }

            this.bias = 2 * random.NextDouble() - 1;
        }

        public double Forward(double[] inputs)
        {
            this.value = this.bias;

            for (int i = 0; i < numInputs; i++)
            {
                this.value += inputs[i] * this.weights[i];
            }

            return this.value;
        }

    }

    internal class InputNode
    {
        public double value { get; set; }

        public InputNode()
        {
            this.value = 0;
        }

        public InputNode(double value)
        {
            this.value = value;
        }
    }
}
