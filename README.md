# S2VT
## Tensorflow implement of paper: Sequence to Sequence: Video to Text

### extract video features
First, download MSVD dataset, and extract video features:
```bash
$ python extract_feats.py
```
After this operation, you should split the features into two parts:
 - `train_features`
 - 'test_features'

Second, train the model:
```bash
$ CUDA_VISIBLE_DEVICES=0 ipython
```
When in the ipython environment, then:
```bash
>>> import model_rgb
>>> model_rgb.train()
```
You should change the training parameters and directory path in the `model_rgb.py`

Third, test the model, choose a trained model, then:
```bash
>>> import model_rgb
>>> model_rgb.test()
```
After testing, a text file, "S2VT_results.txt" will generated.
