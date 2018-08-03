import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;


public class Mnist{
    private SameDiff sd;

    public static void loadModel(String file){
        sd = TFGraphMapper.getInstance().importGraph(new File(file));
        if (sd == null){
            throw new Exception("Error loading model : " + file);
        }
    }

    public static INDArray predict(INDArray arr){
        //INDArray batchedArr = Nd4j.expandDims(arr, 0);
        arr = Nd4j.pile(arr, arr);

        sd.associateArrayWithVariable(batchedArr, sd.variables().get(0));
        INDArray out = sd.execAndEndResult();
        return Nd4j.get(NDArrayIndex.point(0);
    }

    public static INDArray predictBatch(INDArray arr){
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        return sd.execAndEndResult();
    }

    public static INDArray predict(String file){
        File file = new File(filepath);
        BufferedImage img = ImageIO.read(file);
        double data[] = new double[28 * 28];
        for(int i = ; i < 28; i++){
            for(int j = 0; j < 28; j++){
                data[i * 28 + j] = (double)img.getRGB(i, j) / 255.0;
            }
        }
        INDArray arr = Nd4j.create(data).reshape(1, 28, 28);
        arr = Nd4j.pile(arr, arr);
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        output = sd.execAndEndResult().get(NDArrayIndex.point(0));
        return output;

    }

    public static int predictionToLabel(INDArray prediction){
        return Nd4j.argMax(prediction.reshape(10)).getInt(0);
    }
    public static void main(){
        loadModel("mnist.pb");
        for(int i = 1; i < 11; i++){
            String file = "images/img (%d).jpg";
            file = String.format(file, i);
            INDArray prediction = predict(file);
            String label = predictionToLabel(prediction);
            System.out.println(file + "  ====  " + label);
        }

    }
}
