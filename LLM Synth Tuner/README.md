# LLM Synth Tuner

LLM Synth Tuner is a comprehensive pipeline for fine-tuning language models using OpenAI's API. It includes several scripts for different stages of the process, from data preparation to model fine-tuning and monitoring.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/LLM-Synth-Tuner.git
    cd LLM-Synth-Tuner
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory and add your OpenAI API key:
      ```
      OPENAI_API_KEY=your_openai_api_key
      ```

## Usage

### Fine-Tuning

1. **Prepare your data**:
    - Use `jsonl_converter.py` to convert and split your data into training, validation, and test sets.

    ```sh
    python src/jsonl_converter.py
    ```

2. **Fine-tune the model**:
    - Run `fine_tune_model.py` to start the fine-tuning process.

    ```sh
    python src/fine_tune_model.py
    ```

### Q&A Generation

1. **Generate Q&A pairs**:
    - Use `qna_generation.py` to generate Q&A pairs and multi-turn conversations from text segments.

    ```sh
    python src/qna_generation.py
    ```

### PDF Segmentation

1. **Extract TOC from PDF**:
    - Use `toc_extraction2.py` to extract the Table of Contents from a PDF.

    ```sh
    python src/toc_extraction2.py
    ```

2. **Segment PDF using TOC**:
    - Use `segmentation2.py` to segment the PDF based on its TOC.

    ```sh
    python src/segmentation2.py
    ```

## Scripts Overview

### `fine_tune_model.py`

Handles the fine-tuning process, including cost estimation, file upload, job creation, and monitoring.

### `jsonl_converter.py`

Converts and splits JSONL data into training, validation, and test sets.

### `qna_generation.py`

Generates Q&A pairs and multi-turn conversations from text segments.

### `segmentation2.py`

Segments a PDF document based on its Table of Contents (TOC).

### `toc_extraction2.py`

Extracts the TOC from a PDF using OpenAI's API.

## Configuration

- Configuration files are in YAML format and should be placed in the `src` directory.
- Update the paths and parameters in the configuration files as needed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.