import cern.colt.matrix.impl.SparseDoubleMatrix1D;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

/**
 * Created with IntelliJ IDEA.
 * User: tpeng
 * Date: 8/18/12
 * Time: 10:53 PM
 * To change this template use File | Settings | File Templates.
 */
public class DataSet {
    public static Instance create(File file, int size, int target) throws IOException {

        Instance instance = new Instance();
        SparseDoubleMatrix1D vector = new SparseDoubleMatrix1D(size);
        Scanner scanner = new Scanner(file);

        int index = 0;
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            for (int i=0;i<32;i++) {
                int value = line.charAt(i) - '0';
                vector.set(index, value);
                index += 1;
            }
        }
        String name = file.getName();
        int label = Integer.parseInt(name.split("_")[0]);
        if (label == target) {
            instance.setLabel(1);
        } else {
            instance.setLabel(-1);
        }
        instance.setFeatures(vector);
        return instance;
    }
}
