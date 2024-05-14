import json
import pandas as pd
import random
import sys
import random


from utils.edit_utils import make_edit_batches, make_rewrite 

sys.path.append("../../")


def get_edits(dataset, *args, **kwargs):
        if dataset == 'zsre':
            return get_edits_zsre(*args, **kwargs)
        elif dataset == 'mquake':
            return get_edits_mquake(*args, **kwargs)
        elif dataset == 'counterfact':
            return get_edits_counterfact(*args, **kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        

def get_edits_counterfact(number_of_edits=3, edit_set=1, file_path='data/counterfact/counterfact-edit.json'):
    # Assuming your JSON data is stored in a file named 'data.json'
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # Calculate start and end indices for the edits
    start_index = (edit_set - 1) * number_of_edits
    end_index = start_index + number_of_edits

    prompts = []
    ground_truth = []
    target_new = []
    subject = []
    rephrase_prompt = []
    # locality_prompt = []
    # locality_ground_truth = []
    locality_inputs = {
    'data': {
        'prompt': [],
        'ground_truth': []
    }
}

    # Extracting the first N edits
    for entry in json_data[start_index:end_index]:
        prompts.append(entry['prompt'])
        ground_truth.append(entry['ground_truth'])
        target_new.append(entry['target_new'])
        subject.append(entry['subject'])
        rephrase_prompt.append(entry['rephrase_prompt'])

        locality_inputs['data']['prompt'].append(entry['locality_prompt'])
        locality_inputs['data']['ground_truth'].append(entry['locality_ground_truth'])

    return prompts, ground_truth, target_new, subject, rephrase_prompt, locality_inputs

def get_edits_mquake(number_of_edits=3, edit_set=1, file_path='data/MQuAKE/MQuAKE-CF-3k.json'):
    """
    Create data folder with strcutre:
    
    data/MQuAKE/MQuAKE-CF-3k.json
    data/MQuAKE/MQuAKE-CF.json
    data/MQuAKE/MQuAKE-CF-T.json
    """
    
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # Calculate start and end indices for the edits
    start_index = (edit_set - 1) * number_of_edits
    end_index = start_index + number_of_edits
        
    prompts = []
    ground_truth = []
    target_new = []
    subject = []
    rephrase_prompt = []
    multiHop_prompts = []
    multiHop_answers = []
    singleHop_prompts = []
    singleHop_answers = []
    
    
    for json_element in json_data[start_index:end_index]:
        subject_str = json_element['requested_rewrite'][0]['subject']
        prompts.append(json_element['requested_rewrite'][0]['prompt'].format(subject_str))
        # prompts.append(json_element['requested_rewrite'][0]['prompt'])
        ground_truth.append(json_element['requested_rewrite'][0]['target_true']['str'])
        target_new.append(json_element['requested_rewrite'][0]['target_new']['str'])
        subject.append(subject_str)
        rephrase_prompt.append(json_element['requested_rewrite'][0]['question'])
        multiHop_prompts.append(json_element['questions'])
        multiHop_answers.append(json_element['new_answer'])
        singleHop_prompts.append(json_element['single_hops'][0]['question'])
        singleHop_answers.append(json_element['single_hops'][0]['answer'])

    
    return prompts, ground_truth, target_new, subject, rephrase_prompt, [singleHop_prompts, singleHop_answers]

def get_edits_zsre(number_of_edits=3, edit_set=1, train=True):
    """
    Download data folder from: https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view
    From the downloaded data folder move data/zsre to composable-interventions/data/zsre
    
    This returns zsre train or eval data based on the train boolean param
    """
    
    if train:
        file_path='data/zsre/zsre_mend_train.json'
    else:
        file_path='data/zsre/zsre_mend_eval.json'
    
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # Shuffle the data so that similar edits are not done together
    random.seed(100)
    random.shuffle(json_data)
    
    # Calculate start and end indices for the edits
    start_index = (edit_set - 1) * number_of_edits
    end_index = start_index + number_of_edits

    prompts = []
    ground_truth = []
    target_new = []
    subject = []
    rephrase_prompt = []
    locality_inputs = {
    'data': {
        'prompt': [],
        'ground_truth': []
    }
}
    
    for entry in json_data[start_index:end_index]:
        prompts.append(entry['src'])
        ground_truth.append(entry['answers'][0])
        target_new.append(entry['alt'])
        subject.append(entry['subject'])
        rephrase_prompt.append(entry['rephrase'])
        
        locality_inputs['data']['prompt'].append(entry['loc'].split("nq question:")[1].strip())
        locality_inputs['data']['ground_truth'].append(entry['loc_ans'])
    
    return prompts, ground_truth, target_new, subject, rephrase_prompt, locality_inputs

def get_edits_taxi(edit_method, number_of_edits=None):
    """
    Create data folder with strcutre:
    
    edit_method (str) : choose from [MEMIT, FT, ROME]. Can add more
        - if edit_method in [MEMIT] --> return batches of edits
    number_of_edits (int/None) : how many edits do you want to sample. None means return all 
    """
    
    # load datasets
    edits_df = pd.read_json("data/edits.json")
    eval_df = pd.read_json("data/edits-evaluation.json")
        
    all_train_rewrites = []
    all_eval_query = []
    
    if edit_method in ["MEMIT"]: # can add PMET here
        
        batches = make_edit_batches(edits_df)
        
        for b in batches:
            
            train_rewrites = b.apply(make_rewrite, 1).to_list()
            train_rewrites = {"prompts": [x["prompt"] for x in train_rewrites], "target_new": [x["target_new"] for x in train_rewrites], "subject": [x["subject"] for x in train_rewrites] }
            eval_query = b.filter(["edit"]).merge(eval_df, how = "left", on = "edit")
            
            all_train_rewrites.append(train_rewrites)
            all_eval_query.append(eval_query)
    else:
        
        for e in edits_df.itertuples():
            
            if e.edit_type == "category membership":
                
                if edit_method in ["ROME", "FT"]: # can add GRACE HERE
                    
                    train_rewrite = {
                            'prompts': [f'A {e.subj} is a kind of'],
                            'target_new': [e.entity],
                            'subject': [e.subj]
                            }
                    
                else:
                    
                    raise NotImplementedError(f"edit method {edit_method} not implemented")

                eval_query = eval_df.loc[lambda x: (x.edit_type == "category membership") & (x.entity == e.entity) & (x.subj == e.subj)]
            
            elif e.edit_type == "category property":
                
                if edit_method in ["ROME", "FT"]: # can add PMET, GRACE
                    
                    rewrite_prompt = e.query_fwd.replace("<subj>", e.entity).replace(" <answer>", "")
                    train_rewrite = {
                        'prompts': [rewrite_prompt],
                        'target_new': [e.answer_fwd], #{'str': e.entity},
                        'subject': [e.entity]
                    }
                    
                else:
                    
                    raise NotImplementedError(f"edit method {edit_method} not implemented")
                
                eval_query = eval_df.loc[lambda x: (x.edit == e.edit)]
                
            
            all_train_rewrites.append(train_rewrite)
            all_eval_query.append(eval_query)
            
    
    # randomly sample as many rewrites from train as requested 
    if number_of_edits is not None:
        sampled_edits = random.sample(all_train_rewrites, number_of_edits)
    else:
        sampled_edits = all_train_rewrites
    
    return sampled_edits, all_eval_query
    
        
    
    
    