# S2VT: Sequence to Sequence: Video to Text
## Acknowledgement
I modified the code from [jazzsaxmafia](https://github.com/jazzsaxmafia/video_to_sequence), and I have fixed [some problems](https://github.com/jazzsaxmafia/video_to_sequence/issues/9) in his code.

## Requirement
 - Tensorflow 0.12
 - Keras

## How to use my code

### First, download MSVD dataset, and extract video features:
```bash
$ python extract_feats.py
```
After this operation, you should split the features into two parts:
 - `train_features`
 - `test_features`

###Second, train the model:
```bash
$ CUDA_VISIBLE_DEVICES=0 ipython
```
When in the ipython environment, then:
```bash
>>> import model_rgb
>>> model_rgb.train()
```
You should change the training parameters and directory path in the `model_rgb.py`

### Third, test the model, choose a trained model, then:
```bash
>>> import model_rgb
>>> model_rgb.test()
```
After testing, a text file, "S2VT_results.txt" will generated.

### Last, evaluate results with COCO
We evaluate the generation results with [coco-caption tools](https://github.com/tylin/coco-caption).

You can run the shell `get_coco_tools.sh` get download the coco tools:
```bash
$ ./get_coco_tools.sh
```
After this, generate the reference json file from ground truth CSV file:
```bash
$ python create_reference.py 
```
Then, generate the results json file from `S2VT_results.txt` file:
```bash
$ python create_result_json.py
```
Finally, you can evaluate the generation results:
```bash
$ python eval.py
```

## Results
|Model|METEOR|
|---|:---:|
|S2VT(ICCV 2015)||
|-RGB(VGG)|29.2|
|-Optical Flow(AlexNet)|24.3|
|**Our model**||
|-RGB(VGG)|**28.1**|
|-Optical Flow(AlexNet)|**23.3**|

## Attention
1. Please feel free to ask me if you have questions.
2. I only commit the RGB parts of all my code, you can modify the code to use optical flow features.
