# Open-AI gymnasium 6x6 Empty-Grid solver

This project aims to develop an autonomous agent capable of navigating through the Empty-Grid 6x6 environment. It encompasses recording observations and actions, building and training a Convolutional Neural Network (CNN), and evaluating the model's performance.

## Overview

The project is structured into three main components:

1. **Recording Observations and Actions**: Utilizing the [`record.py`](https://github.com/mradovic38/empty-grid-conv/blob/main/record.py) script, we manually navigate the agent through the MiniGrid environment to collect data. This data consists of observations (environment states) and the corresponding actions taken by the agent.

2. **Building and Training the Convolutional Neural Network**: The [`training.py`](https://github.com/mradovic38/empty-grid-conv/blob/main/training.py) script outlines the process of constructing a CNN model. This model is trained on the recorded observations and actions, learning to predict the best action based on the current state of the environment.

3. **Model Evaluation**: After training, the model's performance is assessed using the [`test.py`](https://github.com/mradovic38/empty-grid-conv/blob/main/test.py) script. This involves running the trained model in the MiniGrid environment to evaluate its ability to navigate autonomously.

## Getting Started

### Dependencies

Ensure you have the following dependencies installed:

- gymnasium==0.29.1
- minigrid==2.3.1
- numpy==1.26.4
- pygame==2.5.2
- scikit_learn==1.5.0
- tensorflow==2.16.1

You can install all required dependencies by running:

```bash
pip install -r requirements.txt
```

### Recording Data

To start recording observations and actions, run:

```bash
python record.py
```

Navigate the agent through the environment using keyboard controls. The observations and actions will be saved for training.

### Training the Model

To train the CNN model with the recorded data, execute:

```bash
python training.py
```

This script will preprocess the data, build the CNN model, and train it.

### Evaluating the Model

To evaluate the trained model's performance, run:

```bash
python test.py
```

This will use the trained model to autonomously navigate the MiniGrid environment, and output the performance metrics.

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
