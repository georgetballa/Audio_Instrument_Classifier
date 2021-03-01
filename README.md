# Instrument Classifier
### Predict Instrument Family from Audio Recording

The goal of this project was to use Transfer Learning in Tensorflow to train a Neural Network to predict instrument family based on audio recordings of the instrument playing a single note. 

### Pipeline
Data was dowloaded from [NSynth](https://magenta.tensorflow.org/nsynth). This model was trained on a a balanced subset (5000 per instrument class, split into 80% training 20% test sets) of the available training set found on the linked site. Validation was performed after training by downloading the validation set.

Using Librosa, WAV files were converted to spectrograms and separated into instrument class folders and saved as PNG images.
Images were not transformed due to the graphical nature of spectrograms. Modifying the images by fitlering, skewing, or blurring as in a traditional image classification network would result in the transformation of the data provided and could inhibit the ability to predict. Having many files to train on made up for the inability to transform.

### Training

Transfer Learning provided a way to load the structure and weights from VGG19 and apply the pre-trained model to the 11 categories in my classification. Initially, traning was performed on top layers only, and layers were incrementally unfrozen until accuracy and loss did not see an improvement.

### Validation

Best weights were saved with 92% accuracy on the test split from the initial dataset. The unseen, unbalanced validation set scored 72% accuracy. As expected, overfitting caused the accuracy to drop on unseen data, and the inability to transform limited the amount of training data available.

A confusion matrix shows further insight into the miscategorization of instruments.

### Conclusion

Classifing audio is a complex, but achievable task with the help of transfer learning. Using spectrograms to describe audio was a successful method with this dataset, but could prove difficult in some other applications. Some data about the audio is lost when converting to spectrograms. Using graphical representation of audio allows for the use of well studied image classification techniques. Transfer learning allows time and resource savings, allowing this project to be completed in a week instead of much longer training times required to train from scratch.

### Future Exploration

Attempt to retain audio data by training on raw audio instead of converting to spectrograms. Assess the prediction accuracy improvement against the additional training time.

Look at more audio focused networks like WaveNet to predict on more than a fixed length time audio sample.

Deep dive on confusion matrix to target the errors in the model and continue to train on more data to assist with the overfitting on the training set.

Research transformation possibilites on spectrograms that would not alter the data.
