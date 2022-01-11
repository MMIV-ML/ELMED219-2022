# Lab 2: Natural language processing

In this lab we'll learn about _natural language processing_. That is, the analysis and synthesis of natural language using computers. This is one of the major focus areas for machine learning, and an area that has seen many striking results over the past few years.

## Slides and video recordings

[ELMED219-2022-Lab1-Lab2-slides](https://docs.google.com/presentation/d/e/2PACX-1vQOKLmNXIEZIFCewE6DBaW-zwEunEjZUfc-1SFOi_hXlIhxcOV66L1E9sVSvGJkIusaFrghF2RuTV62/pub?start=false&loop=false&delayms=3000)

[ELMED219-2022-Lab1-Lab2-PDF-slides](../assets/PDF-slides/7-ELMED219-2022-Lab1-Lab2-EHR_and_NLP.pdf)

ELMED219-2022-Lab2-video


## Jupyter Notebooks

| Notebook    |      1-Click Notebook      |
|:----------|------|
|  [ELMED219-2022-Lab2-NLP-1-medical_tweets.ipynb](https://nbviewer.org/github/MMIV-ML/ELMED219-2022/blob/main/Lab2-NLP/ELMED219-2021-Lab2-NLP-1-medical_tweets.ipynb)  <br>introduces natural language processing and text classification. | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/ELMED219-2022/blob/main/Lab2-NLP/ELMED219-2021-Lab2-NLP-1-medical_tweets.ipynb)|
|  [Extra-ELMED219-2022-Lab2-NLP-2-transformers.ipynb](https://nbviewer.org/github/MMIV-ML/ELMED219-2022/blob/main/Lab2-NLP/Extra-ELMED219-2022-Lab2-NLP-2-transformers.ipynb)  <br>gives an example of using Transformers for NLP. Extra material if you're interested in more recent NLP developments. | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/ELMED219-2022/blob/main/Lab2-NLP/Extra-ELMED219-2022-Lab2-NLP-2-transformers.ipynb)|


## Extra material: NLP using Transformers

In later years, the model class known as Transformers have been shown to be very powerful across all the major tasks in natural language processing (and are also increasingly entering computer vision). 

Here's a quick [survey of some relatively recent developments in deep learning for NLP](https://towardsdatascience.com/a-2021-nlp-retrospective-b6f51e60026a) that can give you a flavor of the field. 

Famous examples of Transformers include [BERT](https://arxiv.org/abs/1810.04805), [GPT-2](https://blog.openai.com/better-language-models/), [GPT-3](https://arxiv.org/abs/2005.14165), [RoBERTa](https://arxiv.org/abs/1907.11692), [XLM](https://arxiv.org/abs/1901.07291), [DistilBERT](https://arxiv.org/abs/1910.01108) and many more. 


(I recommend that you start out by playing with various Transformer models before reading up on the mathematical details. Here's a technical intro to transformers: https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html)


[HuggingFace](https://huggingface.co/)  :hugs: is an open-source provider of large libraries of pre-trained transformers and datasets, and the best starting point for all kinds of deep learning-based NLP these days. 

If you want to get started, have a look at the [Hugging Face course](https://huggingface.co/course/chapter1/1) and their [documentation](https://huggingface.co/transformers/).
