# CRNNCrossEntropy-pytorch

State-of-the-art OCR for handwritten digit recognition, based on custom cross-entropy CRNN.

```python
94.66 % accuracy on provided dataset vs 93 % human validation inter-agreement accuracy.
```

## Consulting

Our open source models are provided without support or warranty. 

If you need help building, managing or training specialized state of the art models or machine learning pipelines for your own business needs in **[vision, NLP, tabular, time series]**, shoot me an email **(weixiluo [dot] gmail [dot] com)** and I will get back to you < 24 hours.

Past and current clients include top hedge funds, leading startups and biotech companies. 

## Model architecture

Our model is a custom CRNN-like model built from scratch in PyTorch, but using cross-entropy + auxiliary stop prediction loss instead of CTC loss (10%+ performance boost).

The model uses a ResNet50 as a backbone and feature extractor, which feeds a 512-dimension latent vector to a bidirectional LSTM. Can easily swap backbone for lighter (ResNet18, MobileNet etc.) or heavier model.

## Installation

Requires Python 3+, PyTorch, and standard scientific Python libraries.

Install missing dependencies using pip or conda.

## Usage

```bash
jupyter notebook
[navigate to model.ipynb or model_pretrain.ipynb]
```

To train your own model, reshape your data in the same structure as the provided zip datasets and load it to the model. 

To predict on your own data, run the notebook where there is a section where you can upload your own image.

## Model training

For best accuracy, the model uses a per-timestep cross-entropy loss alongside an auxiliary loss to improve stoppage prediction. 

We trained the model using SGD with Nesterov momentum, alongside cyclic learning rates for 42 epochs. 

The data was augmented with standard affine transformations, and we converted the full MNIST dataset in .jpg format similar to that of the OCR challenge in order to pretrain both the ResNet50 head and the LSTM.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)