# Run Google Gemma In Local

This project is a web application built with Gradio in Python.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- Python

### Installing

A step by step series of examples that tell you how to get a development environment running:

1. Clone the repository
2. Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python web_gradio.py
```

## Command Line Arguments

The script accepts the following command line arguments:

### `--model`

This argument specifies the model name or the path of model to be used. If not provided, it defaults to 'google/gemma-2b-it'.

Example usage:

```bash
python web_gradio.py --model your_model_name
```

### `--device`

This argument specifies the device to be used for computations. It can be one of "cpu", "cuda", or "mps". If not provided, it defaults to "cpu".

Example usage:

```bash
python web_gradio.py --device cuda
```
