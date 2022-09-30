# Test texts in test_data/ generated randomly from https://randomtextgenerator.com

import pytest

from glob import glob
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from gensim.models import Word2Vec
from tqdm import tqdm
import timeit
import re
import yaml

from utils import smart_procrustes_align_gensim

@pytest.fixture
def test_configs():
    with open("tests/test_config.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

@pytest.fixture
def test_validation_function1(test_configs):
    return {
        'namethetest': test_configs['namethetest'],
        'inputfiles': test_configs['inputs']['directory'],
        'skip_preprocessing': test_configs['preprocessing']['skip'],
        'skip_training': test_configs['training']['skip'],
        'skip_alignment': test_configs['alignment']['skip'],

        ## -- Preprocessing
        'savepreprocessed': test_configs['preprocessing']['savepreprocessed'],
        'minwordlength': test_configs['preprocessing']['pipelines']['minwordlength'],
        'lowercase': test_configs['preprocessing']['pipelines']['lowercase'],
        'remove_punctuation': test_configs['preprocessing']['pipelines']['remove_punctuation'],
        'remove_stopwords': test_configs['preprocessing']['pipelines']['remove_stopwords'],

    ## -- Training
        'epochs': test_configs['training']['options']['epochs'],
        'vector_size': test_configs['training']['options']['vector_size'],
        'sg': test_configs['training']['options']['sg'],
        'min_count': test_configs['training']['options']['min_count'],
        'window': test_configs['training']['options']['window'],
        'start_alpha': test_configs['training']['options']['start_alpha'],
        'end_alpha': test_configs['training']['options']['end_alpha'],
        'workers': test_configs['training']['options']['workers']
        }


# ------------------- Start timing the whole process --------------------
def test_preprocess(test_validation_function1):
    startall = timeit.default_timer()

    # ------------------- Import configs --------------------
    # --- Relevant vars
    ## -- General
    namethetest = test_validation_function1['namethetest']
    inputfiles = test_validation_function1['inputfiles']
    skip_preprocessing = test_validation_function1['skip_preprocessing']
    skip_training = test_validation_function1['skip_training']
    skip_alignment = test_validation_function1['skip_alignment']

    ## -- Preprocessing
    if skip_preprocessing == False:
        savepreprocessed = test_validation_function1['savepreprocessed']
        minwordlength = test_validation_function1['minwordlength']
        lowercase = test_validation_function1['lowercase']
        remove_punctuation = test_validation_function1['remove_punctuation']
        remove_stopwords = test_validation_function1['remove_stopwords']

    ## -- Training
    if skip_training == False:
        epochs = test_validation_function1['epochs']
        vector_size = test_validation_function1['vector_size']
        sg = test_validation_function1['sg']
        min_count = test_validation_function1['min_count']
        window = test_validation_function1['window']
        start_alpha = test_validation_function1['start_alpha']
        end_alpha = test_validation_function1['end_alpha']
        workers = test_validation_function1['workers']

    # ------------------- Create dirs if needed --------------------

    @pytest.fixture(scope="session")
    def needed_dirs(tmp_path_factory):
    # --- General outputs directory
        outdir = tmp_path_factory.mktemp("test_outputs")
        outdir_namethetest = tmp_path_factory.mktemp(f"test_outputs/{namethetest}")
        rawdir = tmp_path_factory.mktemp(f"test_outputs/{namethetest}/raw")
        aligneddir = tmp_path_factory.mktemp(f"test_outputs/{namethetest}/aligned")
        preprocesseddir = tmp_path_factory.mktemp(f"test_outputs/{namethetest}/preprocessed_corpus")
        return {
            'outdir': outdir,
            'outdir_namethetest': outdir_namethetest,
            'rawdir': rawdir,
            'aligneddir': aligneddir,
            'preprocesseddir': preprocesseddir
        }

    # ------------------- Preprocessing & training --------------------

    # --- List the paths for all the texts inside the input dir
    alltexts = glob(f'{inputfiles}/*txt')

    # --- Define stopwords
    cachedStopWords = stopwords.words("english")

    ## -- To expand the default NLTK stopwords for English, uncomment the following three lines and add your own:
    #newstopwords = ['mrs','mr','say','says','said','tell','told','seem','seems','seemed','ask','asks','asked','upon','aboard','about','above','account','across','after','against','ago','ahead','along','alongside','amid','among','around','aside','at','atop','away','because','before','behalf','behind','below','beneath','beside','besides','between','beyond','but','by','circa','considering','depending','despite','down','due','during','for','from','further','given','in','including','inside','instead','into','less','like','near','notwithstanding','of','off','on','onto','opposite','other','out','outside','over','owing','per','plus','regarding','regardless','round','since','than','thanks','through','throughout','till','to','together','toward','towards','under','underneath','unlike','until','up','upon','versus','via','with','within','without']
    #cachedStopWords.extend(newstopwords)
    #cachedStopWords = list(dict.fromkeys(cachedStopWords))

    for subcorpus in alltexts:
        # --- Start timing for the specific time slice
        startslice = timeit.default_timer()

        # --- Get name of the time slice (i.e. the last bit of the file path without the ext)
        nameoftimeslice = subcorpus.split('/')[-1].split('.')[0]

        print(f'Now preprocessing {nameoftimeslice}...')

        # --- Open new file in w mode if option to save the preprocessed text is chosen
        
        if savepreprocessed == True:
            newfile = open(f"{needed_dirs['preprocesseddir']}/{nameoftimeslice}.txt", 'w')
        
        # --- Start list of sentences (i.e. articles, text chunks, etc., depending on input), this will be input of w2v
        sentences = []

# ------------------- stopwdsrm --------------------

        with open(subcorpus) as infile:
            # --- Read line by line and preprocess
            for line in tqdm(infile):
                ## -- Preprocess the texts according to configs
                text = re.sub('-\n', '', line) # Remove OCR'd linebreaks within words if they exist
                text = re.sub('\n', ' ', text) # Remove ordinary linebreaks (there shouldn't be, so this might be redundant)
                if remove_punctuation == True:
                    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text)) # Remove anything that is not a space, a letter, or a number
                if lowercase == True:
                    text = str(text).lower()
                ## -- Remove stopwords if chosen to
                if remove_stopwords == True:
                    text = ' '.join([word for word in text.split() if word not in cachedStopWords and len(word) >= minwordlength])
                ## -- Write the line to the newfile if chosen to (without tokenizing)
                if savepreprocessed == True:
                    if text != ' ' and text != '' and text != '\n':
                        newfile.write(text + '\n')
                ## -- Tokenize if what's left after preprocessing is not just whitespace
                if skip_training == False:
                    if text != ' ' and text != '' and text != '\n':
                        text = regexp_tokenize(text, pattern='\s+', gaps =True)
                        sentences.append(text)
        
        print('All preprocessing finished. Altogether it took {} minutes.'.format((timeit.default_timer() - startslice) / 60))
        
        if skip_training == False:
            # --- Initialize w2v model with selected parameters
            model = Word2Vec(min_count=min_count,
                                window=window, 
                                sg=sg,
                                vector_size=vector_size,
                                workers=workers)

            # --- Build vocabulary from the sentences based on the initialized model
            print('Building the vocab...')
            model.build_vocab(sentences)
            print('Vocab built!')
            
            # --- Start the training
            print('Now training the {}s model... This might take a very long time. Have a coffee. Or two.'.format(nameoftimeslice))

            # --- Start timing for training process
            traintime = timeit.default_timer()
            
            model.train(sentences, 
                            total_examples=model.corpus_count, 
                            start_alpha=start_alpha, 
                            end_alpha=end_alpha, 
                            epochs=epochs)
            
            # --- Save model
            model.save(f"{needed_dirs['rawdir']}/{nameoftimeslice}.model")

            stopslice = timeit.default_timer()
            print('{} done! It took {} minutes'.format(nameoftimeslice, (stopslice - startslice)/60))

    # ------------------- Alignment --------------------

        if skip_alignment == False:
            print('Now starting alignement with Orthogonal Procrustes...')
            # --- Start timing alignment process
            startalign = timeit.default_timer()

            # --- List all raw models and order them by name
            allmodels = sorted(glob(f"{needed_dirs['rawdir']}/*model"))

            # --- Load last model (fixed time slice)
            model1 = Word2Vec.load(allmodels[-1])

            # --- Just get the model name and make a copy in the aligned folder
            model1_name = allmodels[-1].split('/')[-1]
            model1.save(f"{needed_dirs['aligneddir']}/{model1_name}")

            # --- Aligned each of the models to the model for the last time slice
            for model in allmodels[:-2]:

                model2_name = model.split('/')[-1]
                print('Now aligning {} to {}...'.format(model1_name,model2_name))
                
                ## -- Load model
                model2 = Word2Vec.load(model)

                ## -- Temp create a third model and save it in the aligned folder
                model3 = smart_procrustes_align_gensim(model1,model2)

                model3.save(f"{needed_dirs['aligneddir']}/{model2_name}")

            print('Alignment complete!')
            print('Time to finish alignment: {} mins'.format(round((timeit.default_timer() - startalign) / 60, 2)))

        print('Time to finish the whole process: {} mins'.format(round((timeit.default_timer() - startall) / 60, 2)))