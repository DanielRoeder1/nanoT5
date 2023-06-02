import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
import os

######################################################
# Functions for tokenizing MSMACRO and GPTDolly data #
######################################################

def get_dataset(args, 
                tokenizer,
                column_lookup = {"query": "instruction", "passage": "context", "response": "response"}, 
                save_data = True):
    if args.model.mode == "q_p_a" and args.model.know_enc_name != "T5":
        assert args.model.know_enc_name is not None, "know_enc_name must be specified for mode q_p_a"
        know_tokenizer = AutoTokenizer.from_pretrained(args.model.know_enc_name) 
    else:
        know_tokenizer = tokenizer
    data = pd.read_csv(args.data.data_dir).dropna()
    process_dataset = []
    for query, passage, answer in tqdm(zip(data[column_lookup["query"]].values, data[column_lookup["passage"]].values, data[column_lookup["response"]].values)):
            inputs = data_switching(args.model.mode, query, passage, answer, tokenizer)
            if inputs is not None:
                process_dataset.append(inputs)
    df = Dataset.from_list(process_dataset)
    df = df.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "know_tokenizer": know_tokenizer}, remove_columns= df.column_names)
    df = df.train_test_split(test_size = args.data.test_size, seed = args.seed)
    if save_data:
        save_path = os.path.join(os.path.dirname(args.data.data_dir), "hf_processed_data")
        df.save_to_disk(save_path)
    return df

def data_switching(mode, query, p, a, tokenizer):
    """
    Changes data format based on mode input by user
    """
    assert mode in ["q_a", "qp_a", "q_pa","q_p_a"], "select one of the following modes: q_a, qp_a, q_pa, q_p_a"
    if mode in ["q_pa", "qp_a"]: 
        assert tokenizer.sep_token_id is not None, "tokenizer must have a sep_token_id for mode q_pa or qp_a"
        sep_token = tokenizer.sep_token
    query,p,a = query.strip(), p.strip(), a.strip()
    if mode == "q_a":
        output =  {'input_ids': query, 'labels': a}
    elif mode == "qp_a":
        query_p = query + " " + sep_token + " " + p
        output =  {'input_ids': query_p, 'labels': a}
    elif mode == "q_pa":
        a_p = p + " " + sep_token + " " + a
        output =  {'input_ids': query, 'labels': a_p}
    elif mode == "q_p_a":
        output =  {'input_ids': query, 'p_input_ids':p, 'labels': a}
    # Remove samples that are longer than the model max length
    if any([True if len(tokenizer.tokenize(v)) > tokenizer.model_max_length  else False for v in output.values()]): 
        return None
    return output
    
def tokenize_function(inputs, tokenizer, know_tokenizer):
    """
    For q_pa set the labels to -100 for all tokens that belong to the passage (context)
    """
    t_lookup = {"input_ids": tokenizer, "labels": tokenizer, "decoder_input_ids": tokenizer, "p_input_ids": know_tokenizer}
    tokenized_input = {k: t_lookup[k](v, truncation = True).input_ids for k,v in inputs.items()}
    
    # This would mask the context in the labels resulting on no loss being calculated for the context
    #if mask_context:
    #    tokenized_input['labels'] = [[-100 if i < l_i else  l for i,l in enumerate(label)] for label in tokenized_input['labels'] if (l_i:=label.index(tokenizer.sep_token_id)) >0]
    return tokenized_input