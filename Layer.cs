using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal class Layer
    {
        public int numInputs { get; private set; }
        public Node[] nodes { get; private set; }
        public int size { get; private set; }
        public double[] inputs { get; private set; }
        public double[] activatedOutputs { get; private set; }
        public double[] weightedOutputs { get; private set; }

        public double[,] costGradientW { get; private set; }
        public double[] costGradientB { get; private set; }

        public Layer(int numInputs, int size)
        {
            this.numInputs = numInputs;
            this.size = size;
            this.inputs = new double[this.numInputs];
            this.activatedOutputs = new double[this.size];
            this.weightedOutputs = new double[this.size];
            this.costGradientW = new double[this.numInputs, this.size];
            this.costGradientB = new double[this.size];

            this.nodes = new Node[this.size];
            for (int i = 0; i < size; i++)
            {
                this.nodes[i] = new Node(this.numInputs);
                this.costGradientB[i] = 0;

                for (int j = 0; j < numInputs; j++)
                {
                    this.costGradientW[j, i] = 0;
                }
            }
        }

        public delegate double Activation(double input);
        public delegate double ActivationDerivative(double input);
        public delegate double CostDerivative(double output, double expected);

        public double[] Forward(double[] inputs, Activation ActivationFunction)
        {
            this.inputs = inputs;
            for (int i = 0; i < size; i++)
            {
                this.weightedOutputs[i] = this.nodes[i].Forward(inputs);
                this.activatedOutputs[i] = ActivationFunction(this.weightedOutputs[i]);
            }

            return activatedOutputs;
        }

        public double[] CalculateOutputLayerNodeValues(double[] expectedOutputs, CostDerivative CostDerivativeFunction, ActivationDerivative ActivationDerivativeFunction)
        {
            double[] nodeValues = new double[expectedOutputs.Length];

            for (int i = 0; i < nodeValues.Length; i++)
            {
                double costDerivative = CostDerivativeFunction(this.activatedOutputs[i], expectedOutputs[i]);
                double activationDerivative = ActivationDerivativeFunction(this.activatedOutputs[i]);

                nodeValues[i] = activationDerivative * costDerivative;
            }

            return nodeValues;
        }

        public double[] CalculateHiddenLayerNodeValues(Layer prevLayer, double[] prevNodeValues, ActivationDerivative ActivationDerivativeFunction)
        {
            double[] newNodeValues = new double[this.size];

            for (int curNode = 0; curNode < this.size; curNode++)
            {
                double newNodeValue = 0;
                for (int prevNode = 0; prevNode < prevLayer.size; prevNode++)
                {
                    double weightedInputDerivative = prevLayer.nodes[prevNode].weights[curNode];
                    newNodeValue += weightedInputDerivative * prevNodeValues[prevNode];
                }
                newNodeValue *= ActivationDerivativeFunction(activatedOutputs[curNode]);
                newNodeValues[curNode] = newNodeValue;
            }

            return newNodeValues;
        }

        public void UpdateGradients(double[] nodeValues)
        {
            for (int node = 0; node < this.size; node++)
            {
                for (int input = 0; input < this.numInputs; input++)
                {
                    double weightCostDerivitave = this.inputs[input] * nodeValues[node];
                    this.costGradientW[input, node] += weightCostDerivitave;
                }

                double biasCostDerivative = 1 * nodeValues[node];
                this.costGradientB[node] += biasCostDerivative;
            }
        }

        public void ApplyGradients(double learningRate)
        {
            for (int node = 0; node < this.size; node++)
            {
                this.nodes[node].bias -= costGradientB[node] * learningRate;
                // Console.WriteLine(this.nodes[node].bias + " " + this.nodes[node].weights[0]);
                for (int input = 0; input < this.numInputs; input++)
                {
                    this.nodes[node].weights[input] -= costGradientW[input, node] * learningRate;
                }
            }
        }

        public void ClearGradients()
        {
            for (int node = 0; node < this.size; node++)
            {
                costGradientB[node] = 0;
                for (int input = 0; input < this.numInputs; input++)
                {
                    costGradientW[input, node]  = 0;
                }
            }
        }
    }

    internal class InputLayer
    {
        public int size { get; private set; }
        public double[] values 
        { 
            get
            {
                double[] values = new double[size];
                for (int i = 0; i < this.size; i++) 
                {
                    values[i] = this.nodes[i].value;
                } 
                return values;
            }
            private set
            { 
               for (int i = 0; i < this.size; i++)
                {
                    this.nodes[i].value = value[i];
                }
            }
        }
        public InputNode[] nodes { get; private set; }

        public InputLayer(int size)
        {
            this.size = size;

            this.nodes = new InputNode[this.size];
            for (int i = 0; i < this.size; i++)
            {
                nodes[i] = new InputNode();
            }
        }

        public void SetValues(double[] values)
        {
            for (int i = 0; i < this.size; i++)
            {
                this.nodes[i].value = values[i];
            }
        }
    }
}
