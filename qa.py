
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
import utils

################ COLLAPSE ME I AM SETUP SCRIPT ###################

def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type_ = "noun"
    else: type_ = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], \
                                        'story_'+type_: line['story_'+type_], \
                                        'stories': line['stories']}
    return word_ids

####################### Global VARIABLES #########################

stop_words = nltk.corpus.stopwords.words("english") 

lmtzr = WordNetLemmatizer()

noun_ids = load_wordnet_ids("Wordnet_nouns.csv")

verb_ids = load_wordnet_ids("Wordnet_verbs.csv")

######################## HELPER FUNCTIONS ########################

def sentence_selection(question,story):
    
    eligible_sents = []

    sents = get_sents(story['text'])

    dep_sents = story['story_dep']

    dep_quest = question['dep']

    keywords , pattern = get_keywords_pattern_tuple(question['text'],question['par'])

    for i in range(len(sents)):
        words = nltk.word_tokenize(sents[i])
        words_pos = nltk.pos_tag(words)
        words = list(filter(lambda x: x not in (stop_words + [':','`','’',',','.','!',"'",'"','?']), words))
        words = list(map(lambda x: lmtzr.lemmatize(x[0], pos=penn2wn(x[1])), words_pos))
        
        
        quant = len(set(words) & set(keywords))
        
        #chunk the data , if no match eliminate question. if there is a match +2 to quant

        eligible_sents.append((quant,sents[i],i))


    eligible_sents = sorted(eligible_sents, key=operator.itemgetter(0), reverse=True)

    best = []

    best += [eligible_sents[0][1]]

    best_dep = wn_extract(question,story,eligible_sents[0][2])

    best = (best_dep if best_dep else best[0])

    return best

def wn_extract(question, sentence, sent_index):

    qgraph = question['dep']

    quest_type = [nltk.word_tokenize(question['text'])[0].lower()]

    qnode = find_node(quest_type, qgraph)

    #generalized way to extract plausible answer types from  the graph
    #currently probably a problem

    if qnode:
        answer_type = [qnode['rel']] + [i for i in qnode['deps']]
        if 'dep' in answer_type:
            answer_type += ['dobj','iobj']
    else:
        answer_type = ["nsubj"]

    qmain = find_main(qgraph)

    qword = qmain["word"]
    qpos = penn2wn(qmain["tag"])
    #qword = [lmtzr.lemmatize(qword,qpos).lower()]

    filename = question['sid'] + '.vgl'
    snode = find_similar(qword,qmain['tag'],sentence['story_dep'][sent_index],filename)
    
    sent_list = nltk.sent_tokenize(sentence['text'])

    if snode:
        sgraph = sentence['story_dep'][sent_index]
    else:
        #print('Stage 1 FALLBBACK , question' , question['text'] , 'For word :' , qword , ' missed in ', sent_list[sent_index]) 
        for i in range(len(sentence['story_dep'])):
            snode = find_similar(qword,qmain['tag'], sentence['story_dep'][i],filename)
            if snode:
                sgraph = sentence['story_dep'][i]
                break

    if snode: 
        #print(question['text'],' : ')
        for node in sgraph.nodes.values():
            if node.get('head', None) == snode["address"]:
                #print('word :', node['word'],' pointed to by snode : ' , snode['word'])
                #print(node['rel'] , answer_type ," : ", node['rel'] in answer_type)
                if node['rel'] in answer_type:
                    deps = get_dependents(node, sgraph)
                    deps = sorted(deps+[node], key=operator.itemgetter("address"))
                    #print( 'ANSWER : ', question['text']," : ", " ".join(dep["word"] for dep in deps))
                    return " ".join(dep["word"] for dep in deps)       

    return None

def find_node(word, graph):
    ## replace with is , similar or equal.
    for node in graph.nodes.values():
        if node["lemma"] in word:
            return node
    return None

def find_similar(word, word_pos, graph, filename):
    if word.lower() in ['why','when','what','who','where','did','which','how']:
        #apply some heuristic
        #until that work is done return none
        return find_main(graph)

    wn_word = None

    if word_pos.startswith('V'):
        for key in wn.synsets(word):
            if key.name() in verb_ids:
                wn_word = wn.synset(key.name())
                break

   
    if word_pos.startswith('N'):
        for key in wn.synsets(word):
            if key.name() in noun_ids:
                wn_word = wn.synset(key.name())
                break
    
    if len(wn.synsets(word)) == 1:
        wn_word = wn.synsets(word)[0]
    
    if wn_word:
        wn_to_check = None       
        for node in graph.nodes.values():
            if node['word']:
                if node['word'] in wn_word.lemma_names():
                    return node
                #print(' SYNSETS FOR WORD : ', node['word'], 'as compared to:', wn_word)
                sets = wn.synsets(node['word'])
                for item in sets:
                    path_sim = wn.path_similarity(item,wn_word)
                    if path_sim and path_sim > 0.6:
                        return node
                node_lemma = lmtzr.lemmatize(node['word'],penn2wn(node['tag'])).lower()
                word_lemma = lmtzr.lemmatize(word,penn2wn(word_pos)).lower()
                if node_lemma == word_lemma or word == node_lemma:
                    return node 

    else:

        ##just find the fucking node anyways. if its not a noun or a verb
        ## then they must be being pretty exact with their language
        return find_node([word],graph)

def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None

def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
        
    return results

def penn2wn(treebank_tag):

    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN
        #erroneus but wn corrects as necessary lol.


#converts penn tree bank to WN style pos tags

def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = chunk_matches(pattern, subtree)
        if node is not None:
            return node
    return None


def chunk_matches(pattern,root):
    
    if root is None and pattern is None: 
        return root
    
    elif pattern is None:                
        return root
    
    elif root is None:                   
        return None

    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    if plabel == "*":
        return root

    elif plabel == rlabel:
        for pchild, rchild in zip(pattern, root):
            match = chunk_matches(pchild, rchild) 
            if match is None:
                return None 
        return root
    
    return None

#returns matches based on tree pairs from sherehezade data.

def get_pos(text):
    return nltk.pos_tag(nltk.word_tokenize(nltk.sent_tokenize(text)))

def get_words(text):
    return nltk.word_tokenize(nltk.sent_tokenize(text))

def get_sents(text):
    return nltk.sent_tokenize(text)


############### GRAMMARS AND PATTERNS ###############

GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <PRP>? <NP>* <V> <PP>?}
            """

chunker = nltk.RegexpParser(GRAMMAR)

pp_filter = set(["in", "on", "at",'by','from','to','after','until'])
ppl_filter = set(["in", "on", "at",'near','infront','by','from','to','of','the','a'])
why_filter = set(['so','to', 'because', 'in', 'due', 'for'])

#######################################################


## further specify the grammar of each language ##

def Who(question,question_sch):

    def Who_chunk(text):
        locations = []
        for subtree in text.subtrees():
            if subtree.label() == 'NP':
                locations.append(subtree)
        return locations

    return {'subject_pos':['NNP','NNS'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))") ,'chunk_func':Who_chunk}

def What(question,question_sch):

    def What_chunk(text):
        locations = []
        for subtree in text.subtrees():
            if subtree.label() in ['NP','VP']:
                    locations.append(subtree)
        return locations

        
    return {'subject_pos':['NN','VBD'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))") ,'chunk_func':What_chunk}

def When(question,question_sch):

    def When_chunk(text):
        locations = []
        for subtree in text.subtrees():
            if subtree.label() == 'PP':
                if subtree[0][0] in pp_filter:
                    locations.append(subtree)
        return locations


    return {'subject_pos':['NN'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))")  , 'chunk_func':When_chunk}

def Why(question,question_sch):
    
    def Why_chunk(text):
        locations = []
        for subtree in text.subtrees():
            #if subtree[0] in why_filter:
            locations.append(subtree)
        return locations

    return {'subject_pos':['VB'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))") , 'chunk_func':Why_chunk}

def Where(question,question_sch):

    
    def Where_chunk(text):
        locations = []
        for subtree in text.subtrees():
            if subtree.label() == 'PP':
                if subtree[0][0] in ppl_filter:
                    locations.append(subtree)
        return locations

        
    return {'subject_pos':['NN'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))")  ,'chunk_func':Where_chunk}

def How(question,question_sch):

    def How_chunk(text):
        locations = []
        for subtree in text.subtrees():
            if subtree.label() == 'PP':
                if subtree[0][0] in ppl_filter:
                    locations.append(subtree)
        return locations
    
    return {'subject_pos':['NN'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))")  ,'chunk_func':How_chunk}

def Did(question, question_sch):

    def Did_chunk(text):
        locations = []
        for subtree in text.subtrees():
            if subtree.label() in ['PP','VP']:
                if subtree[0][0] in ppl_filter:
                    locations.append(subtree)
        return locations

    return  {'subject_pos':['NN'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))")  ,'chunk_func':Did_chunk}

def Had(question, question_sch):

    def Had_chunk(text):
        locations = []
        for subtree in text.subtrees():
            if subtree.label() == 'PP':
                if subtree[0][0] in ppl_filter:
                    locations.append(subtree)
        return locations

    return  {'subject_pos':['NN'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))")  ,'chunk_func':Had_chunk}

def Which(question, question_sch):

    def Which_chunk(text):
        locations = []
        for subtree in text.subtrees():
            if subtree.label() == 'PP':
                if subtree[0][0] in ppl_filter:
                    locations.append(subtree)
        return locations

    return  {'subject_pos':['NN'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))")  ,'chunk_func':Which_chunk}


def get_pattern(question,question_sch):

    case = {'who':Who,'what':What,'when':When,'where':Where,'why':Why,\
    #Q_Set from ASG 7
    'how':How, 'did':Did , 'had':Had, 'which':Which}

    wh_question = nltk.word_tokenize(question)[0].lower()

    pattern_dict = case[wh_question](question,question_sch)

    return pattern_dict

def get_keywords_pattern_tuple(question,question_sch):
    
    q_words = nltk.word_tokenize(question)
    q_words = q_words[1:]
    pattern = get_pattern(question,question_sch)
    keywords = list(filter(lambda x: x not in (stop_words + ['’',',','.','!',"'",'"','?']), q_words))
    keywords = nltk.pos_tag(keywords)
    keywords = list(map(lambda x: lmtzr.lemmatize( x[0], pos = penn2wn(x[1]) )  , keywords))

    q_wh = nltk.word_tokenize(question)[0]

    if(q_wh.lower() == 'where'):
        keywords += ppl_filter
    elif(q_wh.lower() == 'when'):
        keywords += pp_filter
    elif(q_wh.lower() == 'why'):
        keywords += why_filter

    return keywords , pattern

def select_best(chunk):
    return chunk[0]

def who_baseline(question,story):

    story_sents = nltk.sent_tokenize(story['text'])
    
    q_dep_words = utils.get_dep_words(question['dep'],list_=False)
    '''
    returns just a list of keywords based on the dependancy graph
    '''
    
    s_dep_words_list = utils.get_dep_words(story['story_dep'],list_=True)
    '''
    returns a list of list of keywords based on the dependancy graph
    '''

    sorter = []

    for i in range(len(s_dep_words_list)):
        similarity_score = utils.get_similarity(q_dep_words, s_dep_words_list[i])

        #zip the score into a list with teh sentence index
        tup = (similarity_score,i)
        sorter.append(tup)
    
    sorter = sorted(sorter, key=operator.itemgetter(0), reverse=True)

    best_index = sorter[0][1]


    return story_sents[best_index]






##################################################################

def get_eligible_chunks(question,story):

    ##populate this function with other test for  chunk retrieval using different methods

    ## constituency tree chunk retrieval metho

    best = sentence_selection(question,story)
    
    return best

def get_answer(question,story):

    flag_500 = story['sid'].startswith('mc500') # mctrain500 missing the sch data, changes pattern
    
    qflags = utils.get_flags(question)

    sch_flag = question['type'] == 'Sch' or question['type'] == 'Story | Sch'


    if qflags['who']:
        
        #sentence selection:
        #resolve anaphora if necesary
        #similarity overlap , fallback to word overlap

        story_text = utils.resolve_pronouns(story['text'])
        answer = story_text

    elif qflags['what']:

        # distinguish between verb and noun return type
        # select sentence with similarity overlap as a first choice 
        # failin onto word overlap of sch if possible

        answer = 'next code'

    elif qflags['when']:
        answer = 'next code'
    elif qflags['why']:
        answer = 'next code'
    elif qflags['where']:
        #
        answer = 'next code'
    elif qflags['which']:
        #question reformation
        answer = 'next code'
    elif qflags['did']:
        #question reformation
        answer = 'next code'
    elif qflags['how']:
        #resovle whether adj or verb gerund return type
        answer = 'next code'
    else:
        #dialogues questions
        #seriously word overlap
        #just get the sentence then question reformation
        print(question['text'])
        answer = 'next code'
    
    return answer



if __name__ == '__main__':

    class QAEngine(QABase):
        @staticmethod
        def answer_question(question, story):
            answer = get_answer(question, story)
            return answer

def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()