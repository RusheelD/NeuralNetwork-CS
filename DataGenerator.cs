using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal static class DataGenerator
    {

        public static ParityData[] ParityDataGenerator(int bitLength, int numSamples)
        {
            Random random = new Random();
            ParityData[] data = new ParityData[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                double[] bits = new double[bitLength];
                for (int j = 0; j < bitLength; j++)
                {
                    bits[j] = random.Next(0, 2);
                }

                double[] targetValues = { Math.Abs(Program.XorArray(bits) - 1), Program.XorArray(bits) };

                data[i] = new ParityData(bitLength, bits, targetValues); 
            }

            return data;
        }

        public static FruitData[] FruitDataGenerator(int numAttributes, int numSamples)
        {
            Random random = new Random();
            FruitData[] data = new FruitData[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                double[] attributes = new double[numAttributes];
                for (int j = 0; j < numAttributes; j++)
                {
                    attributes[j] = random.Next(0, 15);
                }

                double[] targetValues = { (attributes.Sum() / numAttributes) <= 5 ? 1 : 0, (attributes.Sum() / numAttributes) > 5 ? 1 : 0 };

                data[i] = new FruitData(numAttributes, attributes, targetValues);
            }

            return data;
        }

    }
}
