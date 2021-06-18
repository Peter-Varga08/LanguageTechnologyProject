# LanguageTechnologyProject
Repository of Group10. Language Technology Project MSc course at the University of Groningen.

## How to run
# LSTM model:
- download pretrained embeddings (glove.6B.zip) into the 'data' folder from: https://nlp.stanford.edu/projects/glove/

- running genre classification (data generation is automatic, there are no files saved for it in the data folder):
* with parameter sweep: python3 lstm_text_classification.py -mode genre -t
* without parameter sweep: python3 lstm_text_classification.py -mode genre

- running cast classification (plots are saved in the data folder):
* To regenerate the data with masking: python3 cast_pre_processing.py -masking
* To regenerate the data without masking: python3 cast_pre_processing.py
* with parameter sweep: python3 lstm_text_classification.py -mode cast -t
* without parameter sweep: python3 lstm_text_classification.py -mode cast
