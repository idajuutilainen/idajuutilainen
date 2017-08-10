
# import all modules in beginning of text
import glob, io, os, re
from collections import defaultdict
from operator import itemgetter

wd = '/Users/idadamjuutilainen/Desktop/tmgu17/ADL'# absolute path to session working directory
os.chdir(wd)
data_path = 'SAK/data_txt/' # relative path to data
filenames = os.listdir('SAK/data_txt')
print filenames
from gensim import corpora, models


# functions for reading text(s) from a folder
def read_txt(filepath):
    """
    Read txt file from filepath and returns char content in string
    parameters:
        - filepath including filename to file
    """
    with io.open(filepath, 'r', encoding = 'utf-8') as f:
        content = f.read()
    return content




def read_dir(filenames):
    text_ls = [ ]
    for filename in filenames:
        text_ls.append(read_txt(filename))
    return text_ls

#########################################

filenames = glob.glob (data_path +'/*.txt')
print filenames

len(filenames)

Alltexts = read_dir(filenames)
#print Alltexts
print Alltexts[213]

##### identify specific text through index
# BI.txt: Om Begrebet Ironi 29 sep 1841
# EE1.txt: Enten - Eller 20 feb 1843 (del1)
# EE2.txt: Enten - Eller 20 feb 1843 (del2)
# FB.txt: Frygt og Bæven 16 okt 1843
# BA.txt: Begrebet Angest 17 juni 1844
# EOT.txt: En opbyggelig tale 20 dec 1850

target = ['BI','EE1','EE2','FB','BA','EOT']
filenames_clean = []
target_i = []
i = 0
for filename in filenames:
    tmp = filename.split('/')[1].split('.')[0]
    # filenames_clean.append(filename.split('/')[1].split('.')[0])
    if tmp in target:
        target_i.append(i)
    i += 1

target = [] 
for i in target_i:
    target.append(filenames[i].split('/')[1].split('.')[0])

# probe filenames (files of special interest)
print target
# model index for files of special interest
print target_i

####################################


def tokenize(text, lentoken = 0):
    """
    string tokenization for characters only
    - case-fold: lower
    -
    """
    tokenizer = re.compile(r'\W+', re.UNICODE)####it should be told to include æøå
    tokens = [token.lower() for token in tokenizer.split(text)
        if len(token) > lentoken]
    return tokens


# tokenize all documents
Alltokens = []
for text in Alltexts:
    tokens = tokenize(text.lower(), lentoken = 1)
    Alltokens.append(tokens)


for word in Alltokens[0]:
    print word
## stopword filtering
# import stopwords (very restrictive list)

# tokenize and remove external stopwords
#stopword = read_txt('stopword_da.txt').split()
# tokenize and ignore tokens (i.e., words) in stopword list
#tokens_nostop = [token for token in tokenize(text,1) if token not in stopword]

#Alltokens_nostop = []
#for tokens in Alltokens:
    #tokens = [token for token in tokens if token not in stopword]
    #Alltokens_nostop.append(tokens)

# tokenize and remove stopwords from corpus
def gen_ls_stoplist(input, n = 100):
    """
    generate stopword list from list of tokenized text strings
    """
    t_f_total = defaultdict(int)
    #n = 100
    for text in input:
        for token in text:
            t_f_total[token] += 1
    nmax = sorted(t_f_total.iteritems(), key = itemgetter(1), reverse = True)[:n]
    return [elem[0] for elem in nmax]

stopword_2 = gen_ls_stoplist(Alltokens, 150)# try change number of stopwords
Alltokens_nostop_2 = []
for tokens in Alltokens:
    tokens = [token for token in tokens if token not in stopword_2]
    Alltokens_nostop_2.append(tokens)

#data = Alltokens# med stopord
#data_nostop = Alltokens_nostop
data_nostop_2 = Alltokens_nostop_2


#dictionary = corpora.Dictionary(data)
#dictionary_nostop = corpora.Dictionary(data_nostop)
dictionary_nostop_2 = corpora.Dictionary(data_nostop_2)

#text_bow = [dictionary.doc2bow(text) for text in data]
#text_bow_nostop = [dictionary_nostop.doc2bow(text) for text in data_nostop]
text_bow_nostop_2 = [dictionary_nostop_2.doc2bow(text) for text in data_nostop_2]


#change the amount of topics to zoom in and out on text
k = 15
#mdl = models.LdaModel(text_bow, id2word = dictionary, 
                      #num_topics = k, random_state = 1234)
#mdl_nostop = models.LdaModel(text_bow_nostop, id2word = dictionary_nostop, 
                      #num_topics = k, random_state = 1234)
mdl_nostop_2 = models.LdaModel(text_bow_nostop_2, id2word = dictionary_nostop_2, 
                      num_topics = k, random_state = 1234)

#i = -1500
#print data[i]
#inspect bag-of-words
#paragraph = paragraph_bow[i]
#print paragraph

#for i in range(k):
    #print 'Topic',i
    #print[t[0] for t in mdl.show_topic(i,10)]
    #print '-----'

#for i in range(k):
    #print 'Topic',i
    #print[t[0] for t in mdl_nostop.show_topic(i,10)]
    #print '-----'

# print word in topic
for i in range(k):
    print 'Topic',i
    print[t[0] for t in mdl_nostop_2.show_topic(i,15)]
    print '-----'

## inspect files of special interest as the model represents topic in them
for i, j in enumerate(target_i):
    print 'text', target[i], 'has the following topics:'
    file_bow = text_bow_nostop_2[target_i[i]]
    print mdl_nostop_2[file_bow] 
    print
    print
    print '----------------'

#print data
#see what topic is particularly prevalent in doc(later)
#print mdl[data]
#print data


