# CNN_Project

CNN for Sentence Classification (Naver Movie sentiment analysis)
* language : python
* OS environment : macOS Catalina
* IDE : Jupyter Notebook

**How to use**<br>
**CNN modeling with mecab word2vec model**
1. clone repository to your local computer
2. recommend to change file extension to .ipynb
3. recommend to execute in jupyter notebook or colaboratory environment
4. you can use raw ratings.txt or ratings_test.txt and ratings_test.txt in 'prep for modeling' folder
5. make word2vec model first with tokenize_mecab_300dim_w_sw.py file
6. make cnn model with word2vec model
7. you can use 'stopwords.xlsx' in 'prep for modeling' folder for removing stop words (in my case, there was no big difference in val_accuracy(0.7984))

8. if you want to see my other trials for make model with some changes in word2vec dimension/base_tokenizer/stopwords removal, see files in 'okt models files' folder or 'mecab model files' folder
