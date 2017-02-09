# S2VT
## Tensorflow implement of paper: Sequence to Sequence: Video to Text

###First, download MSVD dataset, and extract video features:
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

###Third, test the model, choose a trained model, then:
```bash
>>> import model_rgb
>>> model_rgb.test()
```
After testing, a text file, "S2VT_results.txt" will generated.

###Last, evaluate results with COCO
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
