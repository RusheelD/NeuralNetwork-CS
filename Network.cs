using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace MachineLearning
{
    internal class Network
    {
        public double learningRate { get; set; }
        private Layer[]? layers { get; set; }
        public InputLayer inputs { get; private set; }
        private int[] shape { get; set; }

        public Network(int[] shape, double learningRate) 
        { 
            if(shape == null) throw new ArgumentNullException("shape cannot be null");
            if(shape.Length < 1) throw new ArgumentException("shape cannot be an empty array", "shape");

            this.shape = shape;
            this.learningRate = learningRate;

            this.inputs = new InputLayer(shape[0]);

            if (shape.Length > 1)
            {

                this.layers = new Layer[shape.Length - 1];
                for(int i = 1; i < shape.Length; i++)
                {
                    this.layers[i - 1] = new Layer(shape[i - 1], shape[i]);
                }

            } else
            {
                this.layers = null;
            }
        }

        public Network(int[] shape)
        {
            if (shape == null) throw new ArgumentNullException("shape cannot be null");
            if (shape.Length < 1) throw new ArgumentException("shape cannot be an empty array", "shape");

            this.shape = shape;
            this.learningRate = 0;

            this.inputs = new InputLayer(shape[0]);

            if (shape.Length > 1)
            {

                this.layers = new Layer[shape.Length - 1];
                for (int i = 1; i < shape.Length; i++)
                {
                    this.layers[i - 1] = new Layer(shape[i - 1], shape[i]);
                }

            }
            else
            {
                this.layers = null;
            }
        }

        public double[] ForwardPropogate(Layer.Activation ActivationFunction, Layer.Activation OutputActivationFunction)
        {
            double[] outputs = this.inputs.values;
            

            if (this.layers != null)
            {
                for (int i = 0; i < this.layers.Length - 1; i++)
                {
                        outputs = this.layers[i].Forward(outputs, ActivationFunction);
                }
                outputs = this.layers[this.layers.Length - 1].Forward(outputs, OutputActivationFunction);
            }

            return outputs;
        }

        public double[] ForwardPropogate(double[] inputs, Layer.Activation ActivationFunction, Layer.Activation OutputActivationFunction)
        {
            this.inputs.SetValues(inputs);
            return this.ForwardPropogate(ActivationFunction, OutputActivationFunction);
        }

        public void ApplyAllGradients(double learningRate)
        {
            if (this.layers != null)
            {
                for (int i = this.layers.Length - 1; i >= 0; i--)
                {
                    this.layers[i].ApplyGradients(learningRate);
                }
            }
        }

        public void ClearAllGradients()
        {
            if (this.layers != null)
            {
                for (int i = this.layers.Length - 1; i >= 0; i--)
                {
                    this.layers[i].ClearGradients();
                }
            }
        }

        public void BackPropogate(Data[] dataPoints, Layer.Activation ActivationFunction, Layer.Activation OutputActivationFunction,
            Layer.ActivationDerivative ActivationDerivative, Layer.ActivationDerivative OutputActivationDerivative, Layer.CostDerivative CostDerivative)
        {
            foreach(Data dataPoint in dataPoints)
            {
                UpdateAllGradients(dataPoint, ActivationFunction, OutputActivationFunction, ActivationDerivative, OutputActivationDerivative, CostDerivative);
            }

            ApplyAllGradients(this.learningRate / dataPoints.Length);
            ClearAllGradients();
        }

        public void UpdateAllGradients(Data data, Layer.Activation ActivationFunction, Layer.Activation OutputActivationFunction, 
            Layer.ActivationDerivative ActivationDerivative, Layer.ActivationDerivative OutputActivationDerivative, Layer.CostDerivative CostDerivative)
        {
            this.ForwardPropogate(data.inputs, ActivationFunction, OutputActivationFunction);

            Layer? outputLayer = this.layers != null ? this.layers[this.layers.Length - 1] : null;

            if (outputLayer != null && this.layers != null)
            {
                double[] nodeValues = outputLayer.CalculateOutputLayerNodeValues(data.targetOutput, CostDerivative, OutputActivationDerivative);
                outputLayer.UpdateGradients(nodeValues);

                for (int i = this.layers.Length - 2; i >= 0; i--)
                {
                    nodeValues = this.layers[i].CalculateHiddenLayerNodeValues(this.layers[i + 1], nodeValues, ActivationDerivative);
                    this.layers[i].UpdateGradients(nodeValues);
                }
            }
        }
    }
}
