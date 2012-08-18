import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.jet.math.Functions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: tpeng
 * Date: 8/18/12
 * Time: 10:56 PM
 * To change this template use File | Settings | File Templates.
 */
public class TinyOCR {

    public final static int MAX_FEATURE = 1024;

    private double lambda = 0.0001;

    // the count of instance being trained
    private int t = 0;

    // the count of errors
    private int errors = 0;

    // the weight to learn
    private DenseDoubleMatrix1D weight = new DenseDoubleMatrix1D(MAX_FEATURE);

    private String dataPath;

    public TinyOCR(String data) {
        this.dataPath = data;
    }

    public double inner(DoubleMatrix1D a, DoubleMatrix1D b) {
        return a.zDotProduct(b);
    }

    // a[i] = a[i] * b;
    public void scaleInPlace(DoubleMatrix1D a, double b) {
        a.assign(Functions.mult(b));
    }

    public DoubleMatrix1D scale(DoubleMatrix1D a, double b) {
        return a.copy().assign(Functions.mult(b));
    }

    // a[i] = a[i] + b[i]
    public void addInPlace(DoubleMatrix1D a, DoubleMatrix1D b) {
        a.assign(b, Functions.plus);
    }

    public double norm(DoubleMatrix1D a) {
        return Math.sqrt(inner(a, a));
    }

    /**
     * get the hinge loss with current model
     */
    public double hinge(Instance instance) {
        int label = instance.getLabel();
        SparseDoubleMatrix1D v = instance.getFeatures();
        return Math.max(0, 1 - label * inner(weight, v));
    }

    /**
     * correct the model. see Pegasos algorithm.
     */
    private void correct(Instance instance) {
        int y = instance.getLabel();
        SparseDoubleMatrix1D xs = instance.getFeatures();
        scaleInPlace(weight, 1 - 1.0 / t);
        addInPlace(weight, scale(xs, y * 1.0 / (lambda * t)));
        double norm = norm(weight);
        double scale = 1.0 / (Math.sqrt(lambda) * norm);
        if (scale < 1.0) {
            scaleInPlace(weight, scale);
        }
    }

    /**
     * update the model base on current model and predication
     */
    private void update(Instance instance) {
        double error = hinge(instance);
        t += 1;
        if (error > 0) {
            errors += 1;
            correct(instance);
        }
//        status(100);
    }

    private void status(int interval) {
        if (t % interval == 0) {
            System.out.print("step: " + t);
            System.out.print("\terrors: " + errors);
            System.out.print("\taccuracy: " + (1 - (1.0 * errors / t)));
            System.out.println();
        }
    }

    public void train(int target) throws IOException {
        File dir = new File(dataPath);
        List<File> files = Arrays.asList(dir.listFiles());
        for (File file : files) {
            Instance instance = DataSet.create(file, MAX_FEATURE, target);
            update(instance);
        }
    }

    public int classify(Instance instance) {
        return inner(instance.getFeatures(), weight) > 0 ? 1 : -1;
    }

    public void reset() {
        weight.assign(0);
        t = 0;
        errors = 0;
    }

    public static void main(String... args) throws IOException {

        TinyOCR ocr = new TinyOCR("data/trainingDigits/");

        for (int i = 0; i < 10; i++) {

            System.out.println("train svm model for: " + i);
            ocr.train(i);

            File testDir = new File("data/testDigits/");
            List<File> files = Arrays.asList(testDir.listFiles());

            int pass = 0;
            List<Instance> testInstances = new ArrayList<Instance>();
            for (File file : files) {
                Instance testInstance = DataSet.create(file, MAX_FEATURE, i);
                testInstances.add(testInstance);
            }

            for (Instance instance : testInstances) {
                int predicted = ocr.classify(instance);
                if (instance.getLabel() == predicted) {
                    pass += 1;
                }
            }

            System.out.println("accuracy on test data: " + 1.0 * pass / testInstances.size());
            System.out.println();
            ocr.reset();
        }
    }
}
