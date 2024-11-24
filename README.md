
# Toxic Comment Classification Model

This repository contains a deep learning model to classify toxic comments into multiple categories such as toxic, severe toxic, obscene, threat, insult, and identity hate. The implementation leverages TensorFlow and Gradio for deployment.

## Dataset

The model uses the Jigsaw Toxic Comment Classification Challenge dataset. The dataset includes comments labeled across six categories indicating toxicity.

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- Pandas
- Numpy
- Matplotlib
- Gradio

Install the required libraries:

```bash
pip install tensorflow pandas numpy matplotlib gradio
```

## File Structure

- `toxicity_model.py`: The script containing the complete model pipeline.
- `toxicity.h5`: The trained model file.
- `README.md`: Documentation for the project.

## Model Architecture

The model is built using a sequential architecture with the following layers:

- **Embedding Layer**: Converts words to dense vector representations.
- **Bidirectional LSTM Layer**: Captures long-term dependencies.
- **Dense Layers**: Includes multiple fully connected layers with ReLU activation.
- **Output Layer**: Six sigmoid-activated neurons for multi-class classification.

The model is compiled with the Adam optimizer and binary crossentropy loss for multi-label classification.

## Workflow

1. **Data Preprocessing**
   - Load the dataset using Pandas.
   - Vectorize text data using TextVectorization from TensorFlow.
   - Split the data into training, validation, and test sets.
   - Optimize data input pipeline using TensorFlow's tf.data API.

2. **Model Training**
   - Define a sequential model.
   - Train the model for one epoch using the training and validation data.

3. **Evaluation**
   - Evaluate the model using metrics like Precision, Recall, and Accuracy.
   - Generate predictions on test data.

4. **Deployment**
   - Save the model as `toxicity.h5`.
   - Create an interactive Gradio interface for real-time comment toxicity scoring.

## Usage

### Training

Run the script to train the model:

```bash
python toxicity_model.py
```

### Prediction

The trained model can score the toxicity of a comment:

```python
score_comment("You are awesome!")
```

### Deploy Gradio App

Launch the Gradio interface:

```bash
python toxicity_model.py
```

## Gradio Interface

The Gradio interface provides an interactive platform for testing the model. Enter a comment, and the interface outputs a score for each toxicity category.

### Sample Results

**Input Comment:**
"I am going to kill you"

**Model Output:**

```vbnet
toxic: True
severe_toxic: False
obscene: True
threat: True
insult: True
identity_hate: False
```

## Evaluation Metrics

The model achieves the following evaluation metrics:

- **Precision**: value
- **Recall**: value
- **Accuracy**: value

## Next Steps

- Fine-tune the model with more epochs for better performance.
- Test with additional datasets for generalization.
- Deploy the model on a web service.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

- Jigsaw Toxic Comment Classification Challenge
- TensorFlow Documentation
- Gradio Documentation
