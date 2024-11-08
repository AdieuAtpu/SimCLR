## Installation

```
$ conda create -n [envname] python==3.10.14
$ conda activate [envname]
$ conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
$ pip install -r requirements.txt
```

## Training

Here i use cifar-10, and use fp16 to accelerate

```python
$ python run.py -data ./datasets -dataset-name cifar10 -j 4 --log-every-n-steps 100 --epochs 100 --batch-size 256 --fp16-precision

```

If you want to run it on CPU (for debugging purposes) use the ```--disable-cuda``` option.