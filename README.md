# LanguageTechnologyProject
Repository of Group10. Language Technology Project MSc course at the University of Groningen.

# How to run
## LSTM model:
- download pretrained embeddings (glove.6B.zip) into the 'data' folder from: https://nlp.stanford.edu/projects/glove/

- running genre classification (data generation is automatic, there are no files saved for it in the data folder):
	* with parameter sweep: python3 lstm_text_classification.py -mode genre -t
	* without parameter sweep: python3 lstm_text_classification.py -mode genre

- running cast classification (plots are saved in the data folder):
	* To regenerate the data with masking: python3 cast_pre_processing.py -masking
	* To regenerate the data without masking: python3 cast_pre_processing.py
	* with parameter sweep: python3 lstm_text_classification.py -mode cast -t
	* without parameter sweep: python3 lstm_text_classification.py -mode cast

## Transformer models:
- LTP_movie_cast is a google colab notebook. Different models are commented and uncommenting a particular model will result in that model training on the dataset to predict the main actor for a movie plot. As a dataset you need plots_masked_test, plots_masked_valid, plots_masked_train

- transformers_genre is a google colab notebook where different models are commented  and uncommenting a particular model will result in that model training on the dataset For this you need the wiki_movie_plots_deduped datafile.
