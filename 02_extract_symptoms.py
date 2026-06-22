#Extract Symptoms - use Metamap to create representations of whether the top 200 symptoms are negated, present, or not-observed for both the lookback and the current encounter

#Necessary Packages
import pandas as pd
import numpy as np
import re, spacy, os, shutil, time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from numpy import loadtxt
from pathlib import Path 
import pandasql as ps
from glob import glob
from pymetamap import MetaMap 
import networkx as nx
from networkx.algorithms import bipartite
import hashlib
import janitor as jn
from janitor import col

DATA_PATH = os.path.join('data/')
TEMP_SOSY_DIR = os.path.join('sosy_raw/')
if not os.path.exists(TEMP_SOSY_DIR):
    os.makedirs(TEMP_SOSY_DIR)

#Set Up Servers
metamap_base_dir = '/your_folder_with_metamap/umls/public_mm/' #set to the location of the server
metamap_bin_dir = 'bin/metamap20'
metamap_pos_server_dir = 'bin/skrmedpostctl'
metamap_wsd_server_dir = 'bin/wsdserverctl'

#Start Servers
os.system(metamap_base_dir + metamap_pos_server_dir + ' start') #part of speech tagger
os.system(metamap_base_dir + metamap_wsd_server_dir + ' start') #word sense disambiguation 
time.sleep(60) #give time for warmup

#Helper Functions to Extract Metamap Concepts and Convert to Easy Data Structures
keys_of_interest = ['preferred_name', 'cui', 'semtypes', 'trigger', 'pos_info'] #need the trigger for negated status (last field 0/1)

def get_keys_from_mm(concept, klist):
    conc_dict = concept._asdict()
    conc_list = [conc_dict.get(kk) for kk in klist]
    return(tuple(conc_list))

def get_concepts(text, note_id, keep_semtypes = ['sosy']):
    hashed_noteid = hashlib.md5(str(note_id).encode()).hexdigest()
    hashed_filename = f"{TEMP_SOSY_DIR}{hashed_noteid}.csv.gz"
    if os.path.exists(hashed_filename):
        print(f"Already processed note {hashed_noteid} so skipping.", flush = True)
        return(None)
    else:
        metam = MetaMap.get_instance(metamap_base_dir + metamap_bin_dir)    
        try:
            cons, errs = metam.extract_concepts([text],
                                                word_sense_disambiguation = True,
                                                restrict_to_sts = keep_semtypes,
                                                composite_phrase = 2, # for memory issues
                                                prune = 25) # for memory issues (smaller N is stricter and more memory efficient)
            # Get columns of interest
            cols = [get_keys_from_mm(cc, keys_of_interest) for cc in cons]
            tmp_df = pd.DataFrame(cols, columns = keys_of_interest)
            if (len(tmp_df) > 0):
                tmp_df = tmp_df[tmp_df['semtypes'].str.contains('|'.join(keep_semtypes), na = False)]
                tmp_df = tmp_df[keys_of_interest]
                tmp_df.to_csv(hashed_filename, index = False)
                return(None)
            else:
                return(None)
        except:
            return(None)

#Read Data and Drop Duplicates
raw_notes_df = pd.read_csv('data/notes.csv')
raw_notes_df = raw_notes_df.drop_duplicates()

#Get Symptom Profile for Each Note
_ = Parallel(n_jobs = cpu_count(), verbose = 5)(delayed(get_concepts)(txt, noteid) for txt, noteid in zip(raw_notes_df.clean_note_text, raw_notes_df.note_id))

#Stop Servers
os.system(metamap_base_dir + metamap_pos_server_dir + ' stop') 
os.system(metamap_base_dir + metamap_wsd_server_dir + ' stop')
time.sleep(60)

finding_list = []
for nid in raw_notes_df.note_id:
    hashed_noteid = hashlib.md5(str(nid).encode()).hexdigest()
    fpath = f'{TEMP_SOSY_DIR}{hashed_noteid}.csv.gz'
    if os.path.isfile(fpath):
        res = pd.read_csv(fpath)
    else:
        res = pd.DataFrame()
    res['note_id'] = str(nid)
    finding_list.append(res)

findings_df = pd.concat(finding_list)

#Identify & Gather Negations: 1= negated, 0 = not_negated
def extract_negation_flags(tr):
    tt = re.sub(']', '', tr)
    ttsplit = tt.split(',')
    res = [x[-1] for x in ttsplit]
    return (''.join(res))

findings_df['all_negation_flags'] = Parallel(n_jobs = cpu_count(), verbose = 5)(delayed(extract_negation_flags)(tr) for tr in findings_df.trigger)
findings_df['any_negated'] = ['1' in flags for flags in findings_df.trigger]
findings_df['pref_name_clean'] = [re.sub(' ', '_', x) for x in findings_df.preferred_name]
findings_df['sosy_name_cui'] = 'snc_' + findings_df['pref_name_clean'] + findings_df['cui']

#Use the 200 Most Commonly Mentioned Symptoms
top200sxslist = findings_df.sosy_name_cui.value_counts().reset_index()[:200]
top200sxslist['sosy_name_cui'] = top200sxslist['sosy_name_cui'].astype(str)
findings_df['sosy_name_cui'] = findings_df['sosy_name_cui'].astype(str)
keep_findings_df = pd.merge(findings_df, top200sxslist, on = ['sosy_name_cui'], how = 'inner')
keep_findings_df['present'] = keep_findings_df.any_negated.map(lambda x: True if x is False else False)
keep_findings_df['absent'] = keep_findings_df.any_negated.map(lambda x: x)
findings_df_wide = keep_findings_df.pivot_table(index = ['note_id'], columns = ['sosy_name_cui'], values = ['present', 'absent'], dropna  = False).reset_index()
findings_df_wide.columns = findings_df_wide.columns.map(lambda x: '_'.join(list(reversed([z for z in x if len(z)>0]))))
findings_df_wide['note_id'] = findings_df_wide['note_id'].astype(np.int64) #clean up types in preparation for merge

#Merge Back in Identifier and Save
findings_df_wide = pd.merge(findings_df_wide, raw_notes_df[['note_id', 'pat_enc_csn_id']], on = 'note_id', how = 'left')
snc_cols = [re.match('^snc_.+', cc) for cc in findings_df_wide.columns]
snc_cols = [ss.group(0) for ss in snc_cols if ss is not None]
findings_df_wide = findings_df_wide.groupby(['pat_enc_csn_id'])[snc_cols].apply(lambda x : x.sum())
findings_df_wide = findings_df_wide.reset_index()
findings_df_wide.to_csv(f"{FINAL_TEXT_OUTPUT_PATH}findings_index.csv.gz",index = False)

#Aggregate Over Lookback for findings_previous
