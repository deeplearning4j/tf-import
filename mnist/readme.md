# MNIST MLP Example


### 1. Train model using tensorflow

Run `mnist_tf.py` to train an MLP on the MNIST dataset. The model config and weights will be saved to `mnist.pb`

### 2. Inference from Java

 Load the `mnist.pb` file to get a `SameDiff` instance

```java
SameDiff sd = TFGraphMapper.getInstance().importGraph(new File("mnist.pb"));
```

Load image from file:

```java
File file = new File(filepath);
BufferedImage img = ImageIO.read(file);
```

Create array from image:

```java
double data[] = new double[28 * 28]; // 28 x 28 is the size MNIST images
for(int i = ; i < 28; i++){
    for(int j = 0; j < 28; j++){
        data[i * 28 + j] = (double)img.getRGB(i, j) / 255.0;
    }
}
INDArray arr = Nd4j.create(data).reshape(1, 28, 28);
```
> Note: You can use Datavec to efficiently load images from disk.

Set input variable:

```java
// The first variable is the input variable
SDVariable input = sd.variables().get(0);
sd.associateArrayWithVariable(arr, input);
```

Get output from model:
```java
INDArray prediction = sd.execAndEndResult();
```

Perform `argMax` on the output probabilities to get the label:

```java
int label = Nd4j.argMax(prediction.reshape(10)).getInt(0);
```

### 3. Inference on the JVM from Python

We use jumpy to access Nd4j functionality from python

See [mnist_jumpy.py](mnist_jumpy.py)
```
