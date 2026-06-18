#Vectorize Docs - use BioClinicalBERT to convert notes to vectors

#Import Packages
import numpy as np
import pandas as pd
import os, re, torch
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

#Establish Paths and Models
DATA_PATH = os.path.join('data/')
tokenizer = BertTokenizer.from_pretrained("/share/projects/resources/Bio_ClinicalBERT") #set to location of BERT tokenizer
model = BertModel.from_pretrained("/share/projects/resources/Bio_ClinicalBERT") #set to location of BERT model

#Set # of threads for PyTorch
torch.set_num_threads(32) #set to num cpus

#Read in and Merge Data
notes_df = pd.read_csv(f"{DATA_PATH}notes.csv") #read in data
notes_df = notes_df.drop_duplicates() #remove duplicates just in case
notes_df = notes_df.dropna(subset=['clean_note_text']) #drop empty notes (if any)

#Aggregate Notes in Lookback Windows
aggregated_df = notes_df.groupby(['pat_enc_csn_id'])['clean_note_text'].agg(lambda x: ' '.join(x.dropna())).reset_index()

#Define Helper Functions
def text_to_vector(txt, tok, emb, max_length): #convert token to vector
	if txt != "" and txt is not None and txt is not np.nan:
		inputs = tok(txt, max_length = max_length, truncation = True, return_tensors='pt')
		with torch.no_grad():
			outputs = emb(**inputs)
		embeddings = outputs.last_hidden_state
		res = embeddings.mean(dim=1).squeeze().numpy()
	else:
		res = np.zeros(emb.config.hidden_size)
	return (res)

def split_into_chunks(txt, max_length = 512): #splits words into lists of length max_length tokens
	txt = str(txt)
	chunks = []
	splitter = re.compile(r'(\w+)')
	current_chunk = []
	if (txt is not None) and (txt is not np.nan) and (txt != ''):
		word_list = splitter.findall(txt)
		for idx, tok in enumerate(word_list):
			current_chunk.append(tok)    		
			if len(current_chunk) == max_length or idx+1 == len(word_list):
				chunks.append(' '.join(current_chunk))
				current_chunk = []
	else:
		current_chunk = []
	return(chunks)

def vectorize_doc(txt, tok, emb): #converts doc into vector
	chunks = split_into_chunks(txt)
	if len(chunks) > 0:
		vec_list = [text_to_vector(ch, tok, emb, 512) for ch in chunks]
	else:
		vec_list = [np.zeros(emb.config.hidden_size)]
	if len(vec_list) < 1:
		vec_list = [np.zeros(emb.config.hidden_size)]
	d_mat = np.stack(vec_list, axis=0)
	ew_mean = d_mat.mean(axis=0)
	return(ew_mean)

#Run BERT in Parallel
vec_doc_list = Parallel(n_jobs=32, prefer='processes', verbose=1)(delayed(vectorize_doc)(txt, tokenizer, model) for txt in aggregated_df['clean_note_text'])

#Reformat and Save
reshaped_vec_doc_list = [pd.DataFrame(np.reshape(v, [1, model.config.hidden_size])) for v in vec_doc_list]
vec_df2 = pd.concat(reshaped_vec_doc_list)
vec_df2.columns = ['docvec_' + str(x) for x in range(model.config.hidden_size)] #append docvec_ to column names
vec_df2['pat_enc_csn_id'] = aggregated_df['pat_enc_csn_id'].values #sort
vec_df2.to_csv(f"{DATA_PATH}embedded_BioClinicalBERT.csv", index = False) #save
