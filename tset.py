import numpy as np
from nltk.stem import PorterStemmer , WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
def tokenizer(sentences):
    words = [word_tokenize(words.lower()) for words in sentences]
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    clean_words = [[lemmatizer.lemmatize(word, pos="v") for word in sentence] for sentence in words]
    tokenize_words = [[porter.stem(word) for word in sentence] for sentence in clean_words]
    stop_words = set(stopwords.words("english"))
    clean_words = [[word for word in sentence if word not in stop_words] for sentence in tokenize_words]
    return clean_words
#with open("topics1.data") as f:
#        text_words = tokenizer(f.readlines())
#with open("words.txt" , "w") as file:
#    for word in text_words:
#        for w in word:
#            file.write(w + " ")
#        file.write("\n")
with open("words.txt") as f:
        text_words = f.read().splitlines()
words = []
for data in text_words:
    li = data.split()
    words.append(li)

tags= ['gun', 'basebal', 'window', 'car', 'medicin', 'religion']            
number_words_in_tag = np.zeros(6)

wr = []
number_words = 0
for sentence in words :
    for word in sentence:
        number_words_in_tag[tags.index(sentence[0])] += 1
        wr.append(word)
        number_words+=1

unique_words = list(set(wr))
B = np.zeros((6,len(unique_words)))

for sentence in words :
    for word in sentence:
        B[tags.index(sentence[0])][unique_words.index(word)] += 1


A = number_words_in_tag / number_words
A = np.diag(A)

#number_words_in_tag = number_words_in_tag[: , np.newaxis]
#B = (B + 1) / (number_words_in_tag + 6)

#em = np.load("Emission.py")
#print(em.shape)

print("sucsess")



    

        
        