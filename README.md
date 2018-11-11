# Fast_Image_Style_Transfer
based on https://github.com/hzy46/fast-neural-style-tensorflow

| Content Image | Style Image | Result |
| :------------ |:---------------:| -----:|
|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/test3.jpg" width="300" />|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/CyberPunk4.jpg" width="300" />|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/CyberPunk4-test3.jpg" width="300" />|
|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/SG3.jpg" width="300" />|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/fantasy.jpg" width="300" />|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/fantasy-SG3.jpg" width="300" />|
|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/test2.jpg" width="300" />|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/starry.jpg" width="300" />|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/starry-test2.jpg" width="300" />|
|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/SG1.jpg" width="300" />|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/mgs.jpg" width="300" />|<img src="https://github.com/cruzsoma/Fast_Image_Style_Transfer/blob/master/images/mgs-SG1.jpg" width="300" />|

"generate.py" can transfer style for images.And Build a pb model to deployment.

"train.py" can train a stylization network with specified style image. using [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip).
