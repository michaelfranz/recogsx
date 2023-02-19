# AUDIO GENDER RECOGNITION 


## NON-TECHNICAL DESCRIPTION

This project aims to create a software program, based on a Convolutional Neural Network, which can 
distinguish male from female voices based on a short audio recording of a human voice. Upon receiving
an audio input, the software will return a classification 'male' or 'female'. 

The software has been trained using a large number of recordings of human adults saying a single digit i.e. 0-9. 

## DATA

The data used to train and validate this solution is AudioMNIST. It is widely used to train deep and/or convolutional neural 
networks for speech-recognition tasks. In this project the dataset is used to train a CNN to classify audio input as 
either male or female depending on the speaker.

The dataset was created by Becker et al and is available here: https://github.com/soerenab/AudioMNIST.
The creators of the dataset request that projects using their work cite the following paper:

| Paper Title   | Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals                       |
|---------------|------------------------------------------------------------------------------------------------------------|
| Journal       | CoRR                                                                                                       |
| Volume        | abs/1807.03418                                                                                             |
| Year          | 2018                                                                                                       |
| ArchivePrefix | arXiv                                                                                                      |
| Eprint        | 1807.03418                                                                                                 |
| Authors       | Becker, Sören and Ackermann, Marcel and Lapuschkin, Sebastian and Müller, Klaus-Robert and Samek, Wojciech |

These data must be preprocessed in order to make them suitable for processing by a convolutional neural network.
Preprocessing includes adding a specific form of emphasis, as well as padding. Padding ensures that the resulting input 
has the same length irrespective of audio sample. 
The AudioMNIST preprocessing code in included in this project in the Jupyter notebook 
```preprocess_audiomnist.ipynb```.  

Running this script will write MFC images to the folder mfc_dataset.
By running the shell script ```pre_dataset.sh``` a meaningful subset of these 
images are copied to ```train``` and ```val``` subdirectories of the folder ```mfc_dataset_train_test```. 

Test audio was obtained from the author's work environment. These audio samples are
comparable to the AudioNIST samples, in that they are approximately 1s in length and
are recordings of adults verbalising the digits 0 through 9. One difference is that
speakers provided samples in different languages. This was a quick way to obtain a 
larger test set. If the model is working well, and as long as speakers do not 
fundamentally change their voices when speaking the foreign language, these samples
should provide a good test of the model's capability. Test audio is included in the
project in the folder ```my_audio_samples```.

Running the script ```preprocess_my_audio_samples.ipynb``` will write MFC images 
of these samples into the ```test``` subfolder of ```mfc_dataset_train_test```.

Once the above scripts have been run successfully the folder ```mfc_dataset_train_test``` will contain training, validation and test data for the main program. 

## MODEL 

The model chosen to perform this task is a Convolutional Neural Network (CNN). The CNN does not process audio samples (in the ```wav``` format) directly, but instead processes an image representation (in the ```jpg``` format) of the original audio. The image represents a so-called Mel-frequency cepstrum (MFC). For example, the following MFC image represents a female saying the digit "zero": 

![img_2.png](img_2.png)
   
These RGB images have shape 98x12x3, and the CNN is specifically designed to process shapes of this kind.                                

## HYPERPARAMETER OPTIMSATION

I performed an exhaustive grid search of the following hyperparameters and values:

| Parameter     | Value 1 | Value 2 | Value 3 |
|---------------|---------|---------|---------|
| Epochs        | 5       | 10      | 20      |
| Learning rate | 0.1     | 0.01    | 0.001   |
| Gamma         | 0.1     | 0.3     | 0.7     |

Gamma is the multiplicative factor of learning rate decay. The learning rate was adjusted every 5 epochs.

## RESULTS

TODO

A summary of your results and what you can learn from your model 

## NOTES ON HARDWARE AND PROCESSING SPEED

The project tries to exploit specialised hardware, if present. Training was performed on a Mac M1 Max with both CPU and MPS (Mac M1 GPU). Switching from CPU to GPU results in a 3-fold increase in training speed. The project will automatically detect cuda or mps hardware if present. Program arguments can be used to disable these options of required.  

## CONTACT DETAILS

Mike Mannion B.Sc. (hons) MBA
michaelmannion@me.com

 

