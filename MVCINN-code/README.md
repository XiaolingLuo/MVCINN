This code is for MVCINN.

The dataset can be download from [here](https://github.com/mfiddr/MFIDDR)

The pre-train weight is in [here](https://drive.google.com/file/d/1EKsx7UEEIl4Mc0mbp9ZjsqHTgzIHsHai/view?usp=sharing) 


First, you should set folds as follows (Attention: move the csv files to dataset fold):
```
|--
    |--EYData_BaseEye_newdata
        |--test
            |--rgb
                |--3392_19491104_left_05.jpg
                |--3392_19491104_left_06.jpg
                |--...
        |--train
            |--rgb
                |--1_19970611_left_05.jpg
                |--1_19970611_left_06.jpg
                |--...
        |--test_rgb_label_newname.csv
        |--train_rgb_label_newname.csv
    |--MVCINN
        |--imgs
        |--models
        |--weights
        |--train.py
        |--test.py
        |--dataset.py
        |--README.md
        |--...
```
Second, you can download pre-train weight to weights as:
```
    |--weights
        |--final_0.8010.pth
```

Then, you can run test.py to test a pre-trained model or run train.py to train the MVCINN.