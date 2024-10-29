## DL-FORESTS-IVORY-COAST

<div align="justify">
This repository contains the code used to produce the results of the article "Deep Learning tools to support deforestation monitoring in the Ivory Coast using SAR and Optical satellite imagery" (https://arxiv.org/abs/2409.11186).
Our study is focused on the central part of Ivory Coast (see figure below) and aims at detecting forest/non-forest pixels to detect then a potential deforestation events.
<br>
<br>
<div align="center">
<img src="https://github.com/gabrielesartor/dl-forests-ivory-coast/blob/main/imgs/roi.png" width="600">
</div>
<br>
<br>
The repository presents the scripts to perform training and test of the models, and the dataset creation.
To collect the dataset, it is necessary to have a Google Earth Engine account and enough space on disk to download the amount of images you want to use for training and test.
The general pipeline of our experiments is depicted in the image below.
</div>
<br>
<br>
<div align="center">
<img src="https://github.com/gabrielesartor/dl-forests-ivory-coast/blob/main/imgs/pipeline.png" width="800">
</div>
<br>
<br>

> [!IMPORTANT]
> This work is based on a modified version of the repository https://github.com/davej23/attention-mechanism-unet
