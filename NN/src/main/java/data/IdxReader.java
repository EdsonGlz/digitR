package data;
/*
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
*/
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

public class IdxReader {

    //private final static Logger LOGGER = LoggerFactory.getLogger(IdxReader.class);

    public static final String INPUT_IMAGE_PATH = "C:/Users/gonza/OneDrive/Documentos/Edson/Digitrecognizer/NN/src/main/resources/train-images.idx3-ubyte";
    public static final String INPUT_LABEL_PATH = "C:/Users/gonza/OneDrive/Documentos/Edson/Digitrecognizer/NN/src/main/resources/train-labels.idx1-ubyte";

    public static final String INPUT_IMAGE_PATH_TEST_DATA = "C:/Users/gonza/OneDrive/Documentos/Edson/Digitrecognizer/NN/src/main/resources/t10k-images.idx3-ubyte";
    public static final String INPUT_LABEL_PATH_TEST_DATA = "C:/Users/gonza/OneDrive/Documentos/Edson/Digitrecognizer/NN/src/main/resources/t10k-labels.idx1-ubyte";


    public static final int VECTOR_DIMENSION = 784; //square 28*28 as from data set -> array 784 items

    public static List<LabeledImage> loadData(final int size) {
        return getLabeledImages(INPUT_IMAGE_PATH, INPUT_LABEL_PATH, size);
    }
    public static List<LabeledImage> loadTestData(final int size) {
        return getLabeledImages(INPUT_IMAGE_PATH_TEST_DATA, INPUT_LABEL_PATH_TEST_DATA, size);
    }

    private static List<LabeledImage> getLabeledImages(final String inputImagePath,
                                                       final String inputLabelPath,
                                                       final int amountOfDataSet) {

        final List<LabeledImage> labeledImageArrayList = new ArrayList<LabeledImage>(amountOfDataSet);

        try {

            FileInputStream inImage = new FileInputStream(inputImagePath);
            FileInputStream inLabel = new FileInputStream(inputLabelPath);
            // just skip the amount of a data
            // see the test and description for dataset
            inImage.skip(16);
            inLabel.skip(8);
            System.out.println("Available bytes in inputImage stream after read: " + inImage.available());
            System.out.println("Available bytes in inputLabel stream after read: " + inLabel.available());

            //empty array for 784 pixels - the image from 28x28 pixels in a single row
            double[] imgPixels = new double[VECTOR_DIMENSION];

            System.out.println("Creating ADT filed with Labeled Images ...");
            long start = System.currentTimeMillis();
            for (int i = 0; i < amountOfDataSet; i++) {

                if (i % 1000 == 0) {
                    System.out.println("Number of images extracted: " + i);
                }
                //it fills the array of pixels
                for (int index = 0; index < VECTOR_DIMENSION; index++) {
                    imgPixels[index] = inImage.read();
                }
                //it creates a label for that
                int label = inLabel.read();
                //it creates a compound object and adds them to a list
                labeledImageArrayList.add(new LabeledImage(label, imgPixels));
            }
            System.out.println("Time to load LabeledImages in seconds: " + ((System.currentTimeMillis() - start) / 1000d));
        } catch (Exception e) {
            System.out.println("Smth went wrong: \n" + e);
            throw new RuntimeException(e);
        }

        return labeledImageArrayList;
    }

}
