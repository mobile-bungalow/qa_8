
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

def sentence_selection(question,story,sch_flag=False):
    
    eligible_sents = []

    if sch_flag:
        text = story['sch']
    else:
        text = utils.resolve_pronouns(story['text'])

    sents = get_sents(text)

    keywords , pattern = get_keywords_pattern_tuple(question['text'],question['par'])

    keywords +=['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','at','during','while','from','to']

    for i in range(len(sents)):
        words = nltk.word_tokenize(sents[i])
        words_pos = nltk.pos_tag(words)
        words = list(filter(lambda x: x not in (stop_words + [':','`','’',',','.','!',"'",'"','?']), words))
        words = list(map(lambda x: lmtzr.lemmatize(x[0], pos=penn2wn(x[1])), words_pos))
        
        
        quant = len(set(words) & set(keywords))
        
        #chunk the data , if no match eliminate question. if there is a match +2 to quant

        eligible_sents.append((quant,sents[i],i))


    eligible_sents = sorted(eligible_sents, key=operator.itemgetter(0), reverse=True)

    #best = []

    index = eligible_sents[0][2]

    best = eligible_sents[0][1]

    #best_dep = wn_extract(question,story,eligible_sents[0][2])

    #best = (best_dep if best_dep else best[0])

    return best , index

def wn_extract(question, sentence, sent_index, sch_flag = False):

    qgraph = question['dep']

    quest_type = [nltk.word_tokenize(question['text'])[0].lower()]

    qnode = find_node(quest_type, qgraph)

    if not qnode:
        qnode = find_main(qgraph)

    if sch_flag:
        dep = sentence['sch_dep']
    else:
        dep = sentence['story_dep']

    dep_set = set(i for i in find_main(dep[sent_index])['deps'].keys())
    q_dep_set = set([i for i in qnode['deps']] + [qnode['rel']]  )
    union = q_dep_set & dep_set
    #generalized way to extract plausible answer types from  the graph
    #currently probably a problem

    if qnode:
        answer_type = [qnode['rel']] + [i for i in qnode['deps']]
        if 'dep' in answer_type:
            answer_type += ['dobj','iobj']
    else:
        answer_type = ["nsubj"]

    if 'punct' in answer_type:
        answer_type.remove('punct')

    #answer_type = ['advmod','iobj','dobj']
    qmain = find_main(qgraph)

    qword = qmain["word"]
    qpos = penn2wn(qmain["tag"])
    #qword = [lmtzr.lemmatize(qword,qpos).lower()]

    filename = question['sid'] + '.vgl'
    snode = find_similar(qword, qmain['tag'], dep[sent_index], filename)
    
    sent_list = nltk.sent_tokenize(sentence['text'])

    if snode:
        sgraph = dep[sent_index]
    else:
        #print('Stage 1 FALLBBACK , question' , question['text'] , 'For word :' , qword , ' missed in ', sent_list[sent_index]) 
        for i in range(len(dep)):
            snode = find_similar(qword,qmain['tag'], dep[i],filename)
            if snode:
                sgraph = dep[i]
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
                    return " ".join(depo["word"] for depo in deps)       

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
                    if path_sim and path_sim > 0.32:
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
## thsi model is deprecated, i was goign to fiddle with the grammar for each
## question until they all returned soemthign new and good

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

def default_chunk(text):
    locations = []
    for subtree in text.subtrees():
        if subtree.label() == 'PP':
            if subtree[0][0] in ppl_filter:
                locations.append(subtree)
    return locations

def get_pattern(question,question_sch):

    case = {'who':Who,'what':What,'when':When,'where':Where,'why':Why,\
    #Q_Set from ASG 7
    'how':How, 'did':Did , 'had':Had, 'which':Which}

    wh_question = nltk.word_tokenize(question)[0].lower()

    try:
        pattern_dict = case[wh_question](question,question_sch)
    except KeyError:
        pattern_dict = {'subject_pos':['NN'] , 'tree':nltk.ParentedTree.fromstring("(VP (*) (PP))")  ,'chunk_func':default_chunk}

    return pattern_dict

def get_keywords_pattern_tuple(question,question_sch):
    
    q_words = nltk.word_tokenize(question)
    rep_list = []
    for i in range(len(q_words)):
        if '/' in q_words[i]:
            rep_list += [q_words[i]]
    for i in rep_list:
        q_words[q_words.index(i):q_words.index(i)+1] =  (z for z in i.split('/'))

            
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

#88%
def who_baseline(question,story,sch_flag=False):
    
    eligible_sents = []

    if sch_flag:
        text = utils.resolve_pronouns(story['sch'])
        text_actual = nltk.sent_tokenize(story['sch'])
    else:
        text = utils.resolve_pronouns(story['text'])
        text_actual = nltk.sent_tokenize(story['text'])


    sents = get_sents(text)

    keywords , pattern = get_keywords_pattern_tuple(question['text'],question['par'])

    for i in range(len(sents)):
        words = nltk.word_tokenize(sents[i])
        words_pos = nltk.pos_tag(words)
        words = list(filter(lambda x: x not in (stop_words + [':','`','’',',','.','!',"'",'"','?']), words))
        sim_score = utils.get_similarity(list(set(words)),list(set(keywords)),story['sid'],question['qid'])
        words = list(map(lambda x: lmtzr.lemmatize(x[0], pos=penn2wn(x[1])), words_pos))

        quant = len(set(words) & set(keywords))
        
        if sim_score:
            quant += sim

        eligible_sents.append((quant,text_actual[i],i))

    eligible_sents = sorted(eligible_sents, key=operator.itemgetter(0), reverse=True)


    best = eligible_sents[0][1]

    index = eligible_sents[0][2]

    return best , index


#77 21 30
def what_baseline(question,story,return_type,sch_flag=False):
    eligible_sents = []


    if sch_flag:
        text_actual = get_sents(story['sch'])
        text = utils.resolve_pronouns(story['sch'])
    else:
        text_actual = get_sents(story['text'])
        text = utils.resolve_pronouns(story['text'])

    sents = get_sents(text)

    keywords , pattern = get_keywords_pattern_tuple(question['text'],question['par'])

    kw_mod = []
    
    if return_type == 'quotation':
        kw_mod = ['said']
    if return_type == 'verb':
        kw_mod = []
    if return_type == 'noun':
        if 'day' in question['text']:
            kw_mod += ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']

    keywords += kw_mod

    rem_list = ['in','he','she','him','her','before','after']
    stop_words_cust = [i if i not in rem_list else '' for i in stop_words]

    for i in range(len(sents)):
        words = nltk.word_tokenize(sents[i])
        words_pos = nltk.pos_tag(words)
        words = list(filter(lambda x: x not in (stop_words_cust + [':','’',',','.','!',"'",'"','?']), words))
        sim_score = utils.get_similarity(list(set(words)),list(set(keywords)),story['sid'],question['qid'])
        words = list(map(lambda x: lmtzr.lemmatize(x[0], pos=penn2wn(x[1])), words_pos))
                
        
        quant = len(set(words) & set(keywords))
        if sim_score:
            quant += sim_score
        eligible_sents.append((quant,text_actual[i],i))

    eligible_sents = sorted(eligible_sents, key=operator.itemgetter(0), reverse=True)


    best = eligible_sents[0][1]

    index = eligible_sents[0][2]

    return best, index

#90% up from 66 baseline , also used for why , at 79.5%
def when_baseline(question,story,kw_adds,sch_flag=False):
    eligible_sents = []

    if sch_flag:
        text = story['sch']
        text_actual = get_sents(story['sch'])
    else:
        text = utils.resolve_pronouns(story['text'])
        text_actual = get_sents(story['text'])

    sents = get_sents(text)

    if len(sents) != len(text_actual):
        print(len(sents),len(text_actual))
        print(sents)
        print(text_actual)

    keywords , pattern = get_keywords_pattern_tuple(question['text'],question['par'])

    keywords += kw_adds

    for i in range(len(sents)):
        words = nltk.word_tokenize(sents[i])
        words_pos = nltk.pos_tag(words)
        words = list(filter(lambda x: x not in (stop_words + [':','`','’',',','.','!',"'",'"','?']), words))
        sim_score = utils.get_similarity(list(set(words)),list(set(keywords)),story['sid'],question['qid'])   
        
        words = list(map(lambda x: lmtzr.lemmatize(x[0], pos=penn2wn(x[1])), words_pos))
             
        quant = len(set(words) & set(keywords))

        if sim_score:
            quant += sim_score
            #joint = (set(words) & set(keywords))
            #disjointq = (set(words)-joint) 
            #disjoints =  (set(keywords)- joint)
            #print('\n Question DJ Set : ',disjointq,'\n Sent DJ Set : ',disjoints)


        eligible_sents.append((quant,text_actual[i],i))

    eligible_sents = sorted(eligible_sents, key=operator.itemgetter(0), reverse=True)
    best = eligible_sents[0][1]
    index = eligible_sents[0][2]
    return best , index

#where baseline 68%

##################################################################


def get_answer(question,story,sch_flag=False):

    flag_500 = story['sid'].startswith('mc500') # mctrain500 missing the sch data, changes pattern
    
    qflags = utils.get_flags(question)

    sch_flag = 'Sch' in question['type']

    whole = nltk.word_tokenize(question['text'])

    if not any(qflags[key] for key in qflags):
        for i in whole:
            qflags = utils.get_flags(i)

    if qflags['who']:
        
        #sentence selection:
        #resolve anaphora if necesary
        #similarity overlap , fallback to word overlap

        answer , i = who_baseline(question,story,sch_flag=sch_flag)
        best_dep = wn_extract(question,story,i,sch_flag=sch_flag)
        answer = (best_dep if best_dep else answer)#'next code'
        #answer = 'next code'

    elif qflags['what']:

        # distinguish between verb and noun and quote return type
        # select sentence with similarity overlap as a first choice 
        # failing onto word overlap of sch if possible

        return_type = utils.return_type(question)

        answer , i = what_baseline(question,story,return_type,sch_flag=sch_flag)
        best_dep = wn_extract(question,story,i,sch_flag=sch_flag)
        answer =(best_dep if best_dep else answer)
        #answer = 'next code'

    elif qflags['when']:

        kw_adds = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
        kw_adds += pp_filter
        answer , i = when_baseline(question,story,kw_adds,sch_flag,)
        best_dep = wn_extract(question,story,i,sch_flag=sch_flag)
        answer = (best_dep if best_dep else answer)#'next code'
        #answer = 'next code'
    elif qflags['why']:

        #add why answer triggers to the question when looking for overlap

        kw_adds = why_filter
        answer , i = when_baseline(question,story,kw_adds,sch_flag)
        best_dep = wn_extract(question,story,i,sch_flag=sch_flag)
        answer = (best_dep if best_dep else answer)#'next code'
        #answer = 'next code'

    elif qflags['where']:

        kw_adds = ['in','where','at','front','back','outside','inside']
        answer , i = when_baseline(question,story,kw_adds,sch_flag=sch_flag)
        best_dep = wn_extract(question,story,i,sch_flag=sch_flag)
        answer = (best_dep if best_dep else answer)
        
    elif qflags['which']:
        #question reformation
        kw_adds = []
        answer , i = when_baseline(question,story,kw_adds,sch_flag)
        best_dep = wn_extract(question,story,i)
        answer = (best_dep if best_dep else answer)
        #answer = 'next code'
    elif qflags['did']:
        #question reformation
        #simple overlap , look for 'nt in answer, make a score threshold
        #based on the number of key words

        kw_adds = []
        answer , i = when_baseline(question,story,kw_adds,sch_flag=sch_flag)
        best_dep = wn_extract(question,story,i)
        answer = (best_dep if best_dep else answer)
        answer = 'yes' if "'nt" not in answer else 'no' 
        
        #answer = 'next code'
    elif qflags['how']:
        #resovle whether adj or verb gerund return type

        kw_adds = []
        answer , i = when_baseline(question,story,kw_adds,sch_flag=sch_flag)
        best_dep = wn_extract(question,story,i,sch_flag=sch_flag)
        answer = (best_dep if best_dep else answer)
    else:
        #dialogues questions
        #seriously word overlap
        #just get the sentence then question reformation       
        kw_adds = []
        answer , i = when_baseline(question,story,kw_adds,sch_flag=sch_flag)
        best_dep = wn_extract(question,story,i)
        answer = (best_dep if best_dep else answer)
        #answer = 'next code'
    
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