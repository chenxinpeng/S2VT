# S2VT
## Tensorflow implement of paper: Sequence to Sequence: Video to Text

### extract video features
1. First, download MSVD dataset, and extract video features:
```bash
$ python extract_feats.py
```

2. Second, train the model:
```bash
$ CUDA_VISIBLE_DEVICES=0 ipython
```
When in the ipython environment, then:
```bash
>>> import model_rgb
>>> model_rgb.train()
```

