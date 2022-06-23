# Multimodal emotion recognition

Simple multimodal recognition on the RAVDESS dataset.

## How to install

Install required dependencies with 

`pip install -r requirements.txt`

You also need pyAudioAnalysis. We recommend to follow the steps provided in the official repo.
Download the library and place it in a folder called pyAudioAnalysis inside this directory.

## How to run demo

Execute the following:

`python demo.py 'path_to_video.mp4'`

There are some example videos to try the trained model in dataset/irl folder.

## How to train

Since we couldn't upload the whole training dataset to github, you'll need to provide
your own dataset to train the model. After placing it to your local folder, follow the steps mentioned in the 
demo_multi.ipynb file.