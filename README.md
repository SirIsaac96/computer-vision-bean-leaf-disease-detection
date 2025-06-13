# Bean Leaf Disease Classification using CNN

## Project Overview
This project focuses on building a Convolutional Neural Network (CNN) model to detect and classify diseases in bean leaves. The model is trained to distinguish between healthy leaves and two common disease classes: bean rust and angular leaf spot. The dataset, provided by the course professor, includes labeled images for training, validation, and testing.

### Key Features
        - **Dataset**: The dataset consists of 1,295 bean leaf images, divided into:
        - **Training set**: 1,034 images
        - **Validation set**: 133 images
        - **Test set**: 128 images

### Workflow
        - **Data Processing**: Images are loaded and preprocessed (resized, normalized) using TensorFlow's image_dataset_from_directory.
        - **Model Training**: The CNN is trained on the training set, with validation performance monitored to avoid overfitting.
        - **Evaluation**: The model's performance is assessed on the test set using metrics like accuracy and loss.
        - **Prediction**: The trained model can classify new bean leaf images into one of the three categories (healthy, bean rust, angular leaf spot).

## Model
- **Architecture**: The CNN follows a standard architecture with convolutional layers, activation functions, pooling layers, and fully connected layers for classification.
- **Goal**: Classify bean leaf images into three classes based on visual features.
- **Input**: Images of size 180x180 pixels, processed in batches of 32.
- **Evaluation**: The model predicts classes for a test dataset, with results visualized using Matplotlib.

#### Performance
        - **Training Accuracy**: 98.55% (Loss: 0.0965)
        - **Test Accuracy**: 81.25% (Loss: 0.5613)
        - The test accuracy of 81% indicates robust performance on unseen data, while the high training accuracy suggests effective learning of the dataset's features.

### Tools & Libraries:
        - TensorFlow and Keras for model building and training.
        - OpenCV and PIL for image processing.
        - Matplotlib and NumPy for visualization and numerical operations.

## Results
The output includes a visualization of predictions and a count of predicted classes in the test set. The model achieved a training accuracy of 98.55% and a test accuracy of 81.25%, demonstrating good generalization to unseen data. The model evaluates new bean leaf images, predicting their class (healthy, angular leaf spot, or bean rust).

## License
This project is for educational purposes.
