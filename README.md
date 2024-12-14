# ADS Project - Implementing CNN for Bird Sound Detection with Bioacoustic Techniques

This is the final project for the Advanced Topics in Machine Learning course at NYU. This specific project was for the purpose of detecting bird sounds using CNNs, though we also later expanded this to also include RNNs and a CRNN model (ResNet-LSTM). Birds are an integral part of our ecosystems across the globe. The presence, or lack thereof, of birds are an indicator of the healthiness of an area. This project carries a two-fold benefit: not only it can help in conservation efforts without a need for people to put in laborious and time consuming ground work, it also acts as an early indication that something is wrong, which can help people act earlier and help prevent an issue before it becomes worse. 

This is an overview of what we did in this project initially for our CNN model: 
- Data Preprocessing: Process dataset of bird sounds from Kaggle (23.4 GB) by creating melspectrograms, and then normalize spectrograms   
- Data Pipeline: Retrieve data in shuffled batches, then perform caching, prefetching, and autotuning to increase efficiency in loading data
  - Since there was only a training set, we split the set into a train/validation set
- Data Augmentation: Use MixUp and RandomCutOut to diversify inputs and artificially increase size of dataset to make for more robust training 
- Modeling and Training: Use pre-trained weights for both EfficientNet and ResNet, use a flexible input layer with a GlobalAveragePooling layer, and stack a dropout and fully connected softmax output layer at the end 

We also created a separate, but similar pipeline for the ResNet-LSTM model: 
- Data Preprocessing: Same as before
- Data Pipeline: Same as before, but also split each spectrogram into equally sized slices based on user set time steps, and build dataset using these chronologically ordered slices
- Data Augmentation: Same as before
- Modeling and Training: Similar to model before, but instead use time distributed input and pooling layer to apply model to each slice in spectrogram
  - Use previous model to output a sequence of feature maps that preserves the time dimension, and then input this sequence of feature maps into an LSTM layer that's then connected to the previously mentioned dropout and fully connnected softmax output layer 

Here is a link to the dataset we used: https://www.kaggle.com/competitions/birdclef-2024/data

We have 5 (relevant) files in this (Note: a lot of the files include some similar code due to us splitting the workload and working on separate things at the same time): 
- birdclef24-initial-implementation.ipynb: File that includes most of the data preprocessing and processing steps as well as the code for the training of each of our 4 models (ResNet, EfficientNet, ResNet-LSTM, EfficientNet-LSTM)
- birdclef24-LSTM.ipynb: File that actually has the output for the CRNN model
- resnet_lstm.py: Code for the architecture of the ResNet-LSTM model
- efficientnet_lstm.py: Code for the architecture of the EfficientNet-LSTM model
- tSNE.ipynb: Visualization of our basic EfficientNet-based model and how well it can extract features from the data

If you want to run this yourself, you only really need to use the birdclef-initial-implementation.ipynb and run the code from top to bottom (also download the dataset from Kaggle and place it into a data folder). In the middle, you are able to choose which of our 4 models you want to use, so choose one and run that. Your model will be saved to the models folder for later use if you wish. 

-----
Regarding our results, we found that despite the varying complexities of the models are their number of parameters, the results were extremely similar (with the hybrid models performing worse): 

Validation AUC
- ResNet: 0.9572
- EfficientNet: 0.9616
- ResNet-LSTM: 0.9043
- EfficientNet-LSTM: TBA 
