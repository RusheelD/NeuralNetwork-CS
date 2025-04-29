namespace MachineLearning
{
    internal class Program
    {
        static void Main(string[] args)
        {
            /*
            int bitLength = 16;
            ParityData data = DataGenerator.ParityDataGenerator(bitLength, 1)[0];
            double[] inputData = data.bits;
            double[] targetValues = data.targetOutput;

            int[] shape = { bitLength, 8, 4, 2 };
            Network network = new Network(shape);
            network.inputs.SetValues(inputData);
            double[] outputs = network.ForwardPropogate(Activations.Sigmoid, Activations.Sigmoid);
            double cost = Costs.SquaredDifference(outputs, targetValues);

            Console.WriteLine(
                "With inputData of " + ArrayToString(inputData) +
                "\nAnd a shape of " + ArrayToString(shape) +
                "\nThe target values were " + ArrayToString(targetValues) +
                "\nAnd the output was " + ArrayToString(outputs) +
                "\nSo the cost was " + cost.ToString()
            );
            

            int bitLength = 2;
            int batchSize = 1000;
            int sampleSize = 100;
            int numBatches = 10;

            int[] shape = { bitLength, 16, 16, 2 };
            Network network = new Network(shape, 0.0005);


            for (int i = 0; i < numBatches; i++)
            {
                Data[] batchData = DataGenerator.ParityDataGenerator(bitLength, batchSize);
                network.BackPropogate(batchData, Activations.Sigmoid, Activations.Sigmoid);

                Data testData = DataGenerator.ParityDataGenerator(bitLength, 1)[0];
                double[] testOutputs = network.ForwardPropogate(testData.inputs, Activations.Sigmoid, Activations.Sigmoid);
                Console.WriteLine("After batch " + i +
                ", the cost of a single iteration is " + 
                Costs.SquaredDifferences(testOutputs, testData.targetOutput));
            }

            double[][] targetValues = new double[sampleSize][];
            double[][] outputs = new double[sampleSize][];
            ParityData[] data = DataGenerator.ParityDataGenerator(bitLength, sampleSize);
            for (int i = 0; i < sampleSize; i++)
            {
                network.inputs.SetValues(data[i].inputs);
                outputs[i] = network.ForwardPropogate(Activations.Sigmoid, Activations.Sigmoid);
                targetValues[i] = data[i].targetOutput;
            }

            Console.WriteLine("The average cost across " + sampleSize + 
                " samples, with a bit-length of " + bitLength + 
                ", was: " + Costs.AvgSquaredDifference(outputs, targetValues));
            */

            int numAttributes = 10;
            int batchSize = 1000;
            int sampleSize = 1000;
            int numBatches = 10000;

            int[] shape = { numAttributes, 2 * numAttributes, numAttributes / 2, 2 };
            Network network = new Network(shape, 0.001);


            double[][] targetValues = new double[sampleSize][];
            double[][] outputs = new double[sampleSize][];
            FruitData[] data = DataGenerator.FruitDataGenerator(numAttributes, sampleSize);
            for (int i = 0; i < sampleSize; i++)
            {
                network.inputs.SetValues(data[i].inputs);
                outputs[i] = network.ForwardPropogate(Activations.Sigmoid, Activations.Sigmoid);
                targetValues[i] = data[i].targetOutput;
            }

            Console.WriteLine("BEFORE TRAINING: ");
            Console.WriteLine("The average cost across " + sampleSize +
                " samples, with " + numAttributes +
                " attributes, was: " + Costs.AvgSquaredDifference(outputs, targetValues));
            Console.WriteLine("The accuracy across " + sampleSize +
                " samples, was: " + CalculateAccuracyPercentage(outputs, targetValues) +
                "%\n");
             

            for (int i = 0; i < numBatches; i++)
            {
                Data[] batchData = DataGenerator.FruitDataGenerator(numAttributes, batchSize);
                network.BackPropogate(batchData, Activations.ReLU, Activations.Sigmoid, 
                    ActivationDerivatives.ReLU, ActivationDerivatives.PostSigmoid, CostDerivatives.SquaredDifferenceDerivative);
            }

            data = DataGenerator.FruitDataGenerator(numAttributes, sampleSize);
            for (int i = 0; i < sampleSize; i++)
            {
                network.inputs.SetValues(data[i].inputs);
                outputs[i] = network.ForwardPropogate(Activations.Sigmoid, Activations.Sigmoid);
                targetValues[i] = data[i].targetOutput;
            }

            Console.WriteLine("AFTER TRAINING: ");
            Console.WriteLine("The average cost across " + sampleSize +
                " samples, with " + numAttributes +
                " attributes, was: " + Costs.AvgSquaredDifference(outputs, targetValues));
            Console.WriteLine("The accuracy across " + sampleSize +
                " samples, was: " + CalculateAccuracyPercentage(outputs, targetValues) + 
                "%");

        }

        public static double CalculateAccuracyPercentage(double[][] outputs, double[][] expected)
        {
            double numAccurate = 0;

            for (int i = 0; i < outputs.Length; i++)
            {
                double maxOutput = outputs[i].Max();
                double maxTarget = expected[i].Max();

                double outputIndexMax = Array.IndexOf(outputs[i], maxOutput);
                double targetIndexMax = Array.IndexOf(expected[i], maxTarget);

                numAccurate += outputIndexMax == targetIndexMax ? 1 : 0;
            }

            return numAccurate / outputs.Length * 100;
        }

        public static int XorArray(double[] inputs)
        {
            int ret = (int)inputs[0];

            for (int i = 1; i < inputs.Length; i++)
            {
                ret ^= (int)inputs[i];
            }

            return ret;
        }

        public static string ArrayToString(double[] data)
        {
            string output = "{ ";

            for (int i = 0; i < data.Length - 1; i++)
            {
                output += data[i].ToString();
                output += ", ";
            }

            if (data.Length > 0)
            {
                output += data[data.Length - 1];
                output += " ";
            }

            output += "}";

            return output;
        }

        public static string ArrayToString(int[] data)
        {
            string output = "{ ";

            for (int i = 0; i < data.Length - 1; i++)
            {
                output += data[i].ToString();
                output += ", ";
            }

            if (data.Length > 0)
            {
                output += data[data.Length - 1];
                output += " ";
            }

            output += "}";

            return output;
        }
    }
}