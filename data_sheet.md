# Datasheet Template
 

## Motivation

The dataset is AudioMNIST. It is widely used to train deep and/or convolutional neural networks for speech-recognition 
tasks. In this project the dataset is used to train a CNN to classify audio input as either male or female depending on 
the speaker.

The dataset was created by Becker et al and is available here: https://github.com/soerenab/AudioMNIST.
The creators of the dataset request that projects using their work cite the following paper:

| Title         | Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals                       |
|---------------|------------------------------------------------------------------------------------------------------------|
| Journal       | CoRR                                                                                                       |
| Volume        | abs/1807.03418                                                                                             |
| Year          | 2018                                                                                                       |
| ArchivePrefix | arXiv                                                                                                      |
| Eprint        | 1807.03418                                                                                                 |
| Authors       | Becker, Sören and Ackermann, Marcel and Lapuschkin, Sebastian and Müller, Klaus-Robert and Samek, Wojciech |
 
## Composition

Data instances are audio recordings of digits 0-9 in the wav format. 
There are 30'000 instances.
The dataset does not contain any private or confidential information.

## Collection process

The data was obtained from the above-cited ```github``` repository.

## Preprocessing/cleaning/labelling

Firstly, the audio data was preprocessed to produce a visual rendering (```jpg```) of each recording. The images represent 
Mel-frequency cepstrum renderings of the original audio. It is these images, which are fed into the CNN for training.

The original AudioMNIST labels, amongst other things, the gender of the speaker. The gender has been incorporated 
into the filename.

## Uses

This project has focussed on gender-recognition. Other than the dataset's original intent of training a NN to recognise
which digit is being spoken, it could potentially be used to identify other characteristics of the speaker. The file
```AudioMNIST/audioMNIST_meta.json``` reveals the labels that have been associated with each audio file:

| Label           | Description                                               | Example                  |
|-----------------|-----------------------------------------------------------|--------------------------|
| accent          | Country or region from where the accent originates        | German                   | 
| age             | Speak age                                                 | 25                       | 
| gender          | Speaker gender                                            | male                     | 
| native speaker  | Whether or not the speaker is using their native language | no                       | 
| origin          | From which country or region the speaker originates       | Europe, Germany, Hamburg | 

It should therefore be possible to train a neural network to identify these characteristics.                                                                          

As the dataset is already widely used and established, this author does not forsee any issues using it for other 
purposes provided credit is given to the set's originators. 

## Distribution

See the above-cited ```github``` repository for all details.
  

## Maintenance

See the above-cited ```github``` repository for all details.
