import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from tensorflow.keras import backend as K
from pathlib import Path


def readmat(filename, var_name):
    img = sio.loadmat(filename)
    img = img.get(var_name)
    img = img.astype(np.float32)

    # unsqueeze for channel size of 1
    # return np.expand_dims(img, 0)
    return img


def ind2onehot(indimg):
    indimg = indimg.astype('int')
    classes = indimg.max() + 1
    Y = np.stack([indimg == i for i in range(classes)], axis=4)
    return Y



def dice_coef(y_true, y_pred, smooth=1, numpy=False):
    """
    Dice = (2*|X & Y|)/ (|X| + |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    if numpy:
        intersection = np.sum(np.abs(y_true * y_pred))
        return (2. * intersection + smooth) / (np.sum(np.square(y_true)) + np.sum(np.square(y_pred)) + smooth)
    else:
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def export_outs(X, Y, out, out_path, paths=None):
    # saving probs
    print('Exporting...')
    Path(out_path).mkdir(parents=True, exist_ok=True)
    for i in range(out.shape[0]):
        img = X[i, :, :, :, 0]
        prob = out[i]
        seg = np.argmax(prob, axis=3)
        grt = np.argmax(Y[i], axis=3)

        output = np.stack((img, seg, grt), axis=3)
        if paths == None:
            sio.savemat('{}out{}.mat'.format(out_path, i), {'data': output})
            sio.savemat('{}prob_out{}.mat'.format(out_path, i), {'data': prob})
        else:
            sio.savemat('{}{}'.format(out_path, paths[i]), {'data': output})
            sio.savemat('{}prob_{}'.format(out_path, paths[i]), {'data': prob})


def dice_coeff_summary(outs, Y, class_names=['Liver', 'Kidney', 'Stomach', 'Duodenum', 'Largebowel']):
    tabledata = []
    num_classes = len(class_names)
    means = [0] * num_classes
    for i in range(outs.shape[0]):
        row = [paths[i]]
        mean = 0
        for j in range(1, num_classes + 1):
            yt = Y[i, :, :, :, j]
            yp = outs[i, :, :, :, j]

            dice = dice_coef(yt, yp, numpy=True) * 100
            row.append('%.2f' % dice)

            mean += (dice / 5)
            means[j-1] += (dice / outs.shape[0])

        row.append('%.2f' % mean)
        tabledata.append(row)
    tabledata.append(['Mean'] + ['%.2f' % m for m in means] +
                     ['%.2f' % np.mean(means)])
    df = pd.DataFrame(tabledata, columns=['Name', *class_names, 'Mean'])
    df = df.set_index('Name')
    return df

# def probs_to_mask(probs):


if __name__ == '__main__':
    print('Utils work perfectly')
