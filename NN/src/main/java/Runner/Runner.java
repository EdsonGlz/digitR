package Runner;

import NN.NN;
import sun.awt.windows.WPrinterJob;

import java.util.Scanner;

public class Runner {
    public static void main(String [] args){

        NN neuralNetwork = new NN();
        System.out.println("Iniciando red neuronal");
        System.setProperty("hadoop.home.dir", "C:/Users/gonza/OneDrive/Documentos/Software/hadoop-2.7.2/");
        neuralNetwork.train(10000, 10000);
        System.out.println("Ya termine");
        Scanner s = new Scanner(System.in);
        s.nextLine();
    }
}
