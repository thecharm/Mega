# Source Code of MEGA Model for Multimodal Relation Extraction
Implementation of Our Paper "Multimodal Relation Extraction with Efficient Graph Alignment" in ACM Multimedia 2021. This Implementation is based on the [OpenNRE](https://github.com/thunlp/OpenNRE).

## Model Architecture
![model](model.png)

The Overall Framework of Our Proposed MEGA Model. Our Model Introduces Visual Information into Predicting Textual Relations. Besides, We leverages the Graph Structural Alignment and Semantic Alignment to Help Model Find the Mapping From Visual Relations to Textual Contents.


## Requirements
* `torch==1.6.0`
* `transformers==3.4.0`
* `pytest==5.3.2`
* `scikit-learn==0.22.1`
* `scipy==1.4.1`
* `nltk==3.4.5`

## Data Format
The dataset used in our paper can be downloaded [here](https://github.com/thecharm/MNRE)

>Each sentence is split into several instances (depending on the number of relations).
>Each line contains
>```
>'token': Texts preprocessed by a tokenizer
>'h': Head entities and their positions in a sentence
>'t': Tail entities and their positions in a sentence
>'image_id': You can find the corresponding images using the link above
>'relation': The relations and entity categories
>```

## Usage
