## Getting Started (using Python virtualenv)

[![Open tutorial in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imadtoubal/3D-U-Net-implementation-in-TensorFlow-and-Keras/blob/master/1.GettingStarted.ipynb)

You need to have Python installed in your computer.

1. Install `virtualenv`:
   ```
   pip install virtualenv
   ```
2. Create a Python virtual environment:
   ```
   virtualenv venv
   ```
3. Activate virtual environment:
   1. Windows:
   ```
   cd venv\Scripts
   activate
   cd ..\..
   ```
   2. Lunix / Mac:
   ```
   source venv/bin/activate
   ```
4. Install libraries:

   ```
   pip install -r requirements.txt
   ```

### Training and testing

- Run the training:
  ```
  python train.py
  ```
- Run testing (you need to have trained a model):
  ```
  python test.py
  ```

## Choosing models

In order to choose the models to train, set up the dictionary of `{key: value}` pairs where the `key` is the name you want to give the model and `value` is the model function from the table below:

| Model            | Function     | Description                                                                                                                                                                   |
| ---------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| U-Net            | `unet`       | 3D U-Net architecture with kernel sizes of 3x3x3                                                                                                                              |
| U-Net 2D         | `unet2d`     | 3D U-Net architecture with kernel sizes of 3x3x1                                                                                                                              |
| U-Net++          | `unetpp`     | 3D U-Net++ architecture with kernel sizes of 3x3x3                                                                                                                            |
| U-Net w/ scSE    | `scSEunet`   | 3D U-Net architecture with kernel sizes of 3x3x3, and <br>Spatial and Channel-wise Squeeze and Excitation (scSE)<br>[[View paper for scSE](https://arxiv.org/abs/1709.01507)] |
| U-Net 2D w/ scSE | `scSEunet2d` | 3D U-Net architecture with kernel sizes of 3x3x1, and <br>Spatial and Channel-wise Squeeze and Excitation (scSE)                                                              |
| U-Net++ w/ scSE  | `scSEunetpp` | 3D U-Net++ architecture with kernel sizes of 3x3x3, and <br>Spatial and Channel-wise Squeeze and Excitation (scSE)                                                            |

In order to use the functions listed in the table above, make sure you have imported them from `nets`. Below is an example of importing U-Net++.

```Python
from nets import unetpp
```

## Initializing models

As can be seen in `train.py`, all models takeas parameters `(W, H, D, C)` for width, height, depth, and number of input channels respectively. To complete the example of U-Net++, below is a code for initializing the network:

```Python
model = unetpp(128, 128, 64, 1)
```

## Fitting a model

In order to fit the model, we use `model.fit` function. The most basic way to do so is as follows:

```Python
model.fit(X, Y, batch_size=1, validation_data=(Xv, Yv))
```

- `X` and `Y` are the training data and ground truth of shapes `(B, W, H, W, C_in)`, and `(B, W, H, W, C_out)`. Where `C_in` and `C_out` are the number of channels for the input and output of the network.
- `Xv` and `Yv` are the validation data and ground truth of the same shapes as `X` and `Y`.

## Checkpoints during training (saving the model)

In order to save the model, this code uses a Keras callback during training to save the model with the least loss value for validation data. The callback is defined as follows:

```Python
checkpointer = tf.keras.callbacks.ModelCheckpoint(<nodel_path>, save_best_only=True)
```

Additionally, you can have an **early stopping** condition in order to stop the training if the validation loss does not improve in `n` epochs:

```Python
earlystopper = tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss')
```

with `n` being 20 in the above case.

## Loading a model

Below is example code of loading weights of a traiened model.

```Python
model = untpp(128, 128, 64, 1)
model.load_weights(<model_path>)
```

Where `<model_path>` is the path in which the checkpoint has been saved during training U-Net++ in this example.

## Testing the model

First, we need to generate an output of a testing set `Xt`. `batch_size` is set to 1 for GPU memory limitation reasons.

```Python
out = model.predict(Xt, batch_size=1)
dice_coeff_summary(out, Yt, class_names=<list_of_classes>)
```

## Authors

- **Imad Eddine Toubal** - _Initial work_ - [imadtoubal](https://github.com/imadtoubal)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

Happy coding!
