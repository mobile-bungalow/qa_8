from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk.tree import Tree
from nltk.stem.wordnet import WordNetLemmatizer
import operator
import nltk
import re
import csv
import math
import itertools
import collections

def consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)


def get_flags(question):
    '''
    returns a dict full of flags for programs usage
    '''
    q = nltk.word_tokenize(question['text'])[0].lower()
    qdict = {'what':False,'why':False,'who':False,'when':False,
            'which':False,'where':False,'how':False,'did':False}
    qdict[q] = True
    return qdict


def get_dep_words(dep_graph, list_=False):

    ret = ''

    if list_:

        l = 0
    
    else:

        l = 0


    return ret


def get_similarity(qkw,skw):
    l = 0
    return l

prn_map = {'her':'female','hers':'female','she':'female','herself':'female',
            'him':'male','he':'male','his':'male','himself':'male','hisself':'male',
            'i':'neutral_or_plural','me':'neutral_or_plural','you':'neutral_or_plural','them':'neutral_or_plural','they':'neutral_or_plural',
            'it':'neutral_or_plural','their':'neutral_or_plural','yourself':'neutral_or_plural','yourselves':'neutral_or_plural','itself':'neutral_or_plural',
            'its':'neutral_or_plural','us':'neutral_or_plural','whom':'neutral_or_plural',
            'those':'neutral_or_plural','these':'neutral_or_plural'}

def check_agreement(words,pronoun,prev_dict):
    # check if the word is even in the dict
    # if not enter it under the current pronoun and return current word
    # else go to next item in the word lsit and repeat until there is either a hit
    # or no valid item and then return none
    
    category = prn_map[pronoun]
    miss = False
    if not words:
        return None

    for word in words:
        for key in prev_dict.keys():
                if word.lower() in prev_dict[key] and key == category:
                    return word
                elif word.lower() in prev_dict[key] and key != category:
                    break
                elif key == 'neutral_or_plural': #the last key
                    prev_dict[category] += [word.lower()]
                    return word
        


    return None 

grammar =   """
            NP: {<DT>?<JJ>*<NNP|NNS>}
            GROUP: {(<NP><CC|,>+<NP>(<,|CC>+<NP>)*|<DT>?<NNS>)}
            """

chunker = nltk.RegexpParser(grammar)

def check_for_group(slice_):
    #takes tagges slive of a sentence
    #chunks it looking for groups of Nouns
    # ie: jim, john, jillian or jim and john and jillian etc...
    # calls it a group , returns it, and a number of words (length of the substring)
    # to skip upon further parsing
    ret = chunker.parse(slice_)
    locations = []
    for subtree in ret.subtrees():
            if subtree.label() == 'GROUP':
                locations.append(subtree)
    if len(locations) > 0:
        ret = locations[0]
    else:
        return None , None
    length = len(ret.leaves())
    ret = ' '.join(i[0] for i in ret.leaves())
    return length , ret


def most_freq(lst,meta_lst):
    '''
    returns most frequent item in a list,
    if there are two equal frequency items,
    it returns the lates
    '''
    meta_lst_set = set(meta_lst)
    lst_set = set(lst)
    tup_counts = []
    meta_tup_counts = []

    for i in meta_lst_set:
        meta_tup_counts.append((meta_lst.count(i)/len(meta_lst),i))

    meta_dict = {key: value for (value, key) in meta_tup_counts}

    for i in lst_set:
        tup_counts.append((lst.count(i)*meta_dict[i],i))


    sort = sorted(tup_counts, key=operator.itemgetter(0), reverse=True)
    ret = []
    if len(sort) > 0:
        top_val = sort[0][0]
        valid_items = []
        if top_val < 0.1:
            return None
        for i in sort:
            if i[0] >= top_val - 0.9:
                valid_items += [i[1]]
            else:
                break

        for i in valid_items:
            ret += [i]
        
        return ret    
    else:
        return None          
    
def resolve_pronouns(story, type_='text',index=None):
    '''
    the mc500 test data has unresolved anaphora ,  this attempts to replace all anaphora with

    type = the return type, either text or dependency graph
    index = the sentence index, if it is asked for.
    '''
    
    NNS = [] #they, them

    NNP = [] #he , she, the, him , her

    NN = []#it

    prev_dict = {'male':[],'female':[],'neutral_or_plural':[]}


    if type_ == 'text':
        answer = story['text']
        answer_sents = nltk.sent_tokenize(answer)
        answer_sents = [nltk.word_tokenize(i) for i in answer_sents]
        answer_pos_sents = [nltk.pos_tag(i) for i in answer_sents]
        
        doc_meta_NNS = [] #they, them
        doc_meta_NNP = [] #he , she, the, him , her , 
        doc_meta_NN = [] #it

        #POS correction, some Nouns are not properly labeled
        for j in range(len(answer_pos_sents)):
            for i, word in enumerate(answer_pos_sents[j]):
                if word[0].istitle() and word[1] in ['MD','JJ','VB']:
                    answer_pos_sents[j][i] = (word[0],'NNP')
                
        for i in range(len(answer_pos_sents)):
            backtrack = i - 2 if i >= 2 else 0
            NNS = [] #they, them
            NNP = [] #he , she, the, him , her ,
            NN = [] #it
            for j in (range(backtrack,i) if backtrack != i else [0] ):
                for ndx , tuple_ in enumerate(answer_pos_sents[j]):
                    
                    skip, group = check_for_group(answer_pos_sents[j][ndx-1:])

                    if group:
                        NNS.append(group)
                        doc_meta_NNS.append(group)
                        #print(group)
                        consume(enumerate(answer_pos_sents[j]),skip+i)


                    if tuple_[1] == 'NNS':
                        NNS.append(tuple_[0])
                        doc_meta_NNS.append(tuple_[0])

                    elif tuple_[1] == 'NNP':
                        NNP.append(tuple_[0])
                        #print(tuple_[0])
                        doc_meta_NNP.append(tuple_[0])

                    elif tuple_[1] == 'NN':
                        NN.append(tuple_[0])
                        doc_meta_NN.append(tuple_[0])

            
            NNS_mf = most_freq(NNS,doc_meta_NNS)
            NN_mf = most_freq(NN,doc_meta_NN)
            NNP_mf = most_freq(NNP,doc_meta_NNP)
            


            
            for index in range(len(answer_pos_sents[i])):
                if  answer_pos_sents[i][index][1] in ['PRP$','PRP']:
                    if answer_pos_sents[i][index][1] == 'PRP$':
                        pssv = "'s"
                    else:
                        pssv = ''
                    prn = answer_pos_sents[i][index][0].lower()
                    if prn in ['he','she','his','her','hers']:

                        slct1 = check_agreement(NNP_mf,prn,prev_dict)

                        answer_pos_sents[i][index] = (slct1+pssv if slct1 else answer_pos_sents[i][index][0],\
                                                     'NNP' if slct1 else  answer_pos_sents[i][index][1])


                    elif prn in ['they','their','theirs','them']:

                        slct2 = check_agreement(NNS_mf,prn,prev_dict)

                        answer_pos_sents[i][index] = (slct2+pssv if slct2 else answer_pos_sents[i][index][0],\
                                                    'NNS' if slct2 else  answer_pos_sents[i][index][1])

                    elif prn in ['it','one']:

                        slct3 = check_agreement(NN_mf,prn,prev_dict)

                        answer_pos_sents[i][index] = (slct3+pssv if slct3 else answer_pos_sents[i][index][0],\
                                                     'NN' if slct3 else  answer_pos_sents[i][index][1])
                    else:
                        answer_pos_sents[i][index] = (answer_pos_sents[i][index][0],\
                                                      answer_pos_sents[i][index][1])
            
    ans = ''
    for i in answer_pos_sents:
        ans += ' '.join( word[0] for word in i) + ' '
            
    return ans