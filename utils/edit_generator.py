import json


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

def get_edits_mquake(number_of_edits=3, file_path='data/MQuAKE/MQuAKE-CF-3k.json'):
    """
    Create data folder with strcutre:
    
    data/MQuAKE/MQuAKE-CF-3k.json
    data/MQuAKE/MQuAKE-CF.json
    data/MQuAKE/MQuAKE-CF-T.json
    """
    
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        
    prompts = []
    ground_truth = []
    target_new = []
    subject = []
    accuracy_prompt = []
    multiHop_prompts = []
    multiHop_answers = []
    
    
    for json_element in json_data[:number_of_edits]:
        prompts.append(json_element['requested_rewrite'][0]['prompt'])
        ground_truth.append(json_element['requested_rewrite'][0]['target_true'])
        target_new.append(json_element['requested_rewrite'][0]['target_new'])
        subject.append(json_element['requested_rewrite'][0]['subject'])
        accuracy_prompt.append(json_element['requested_rewrite'][0]['question'])
        multiHop_prompts.append(json_element['questions'])
        multiHop_answers.append(json_element['new_answer'])
    
    return prompts, ground_truth, target_new, subject, accuracy_prompt, multiHop_prompts, multiHop_answers

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
        
    
    
    