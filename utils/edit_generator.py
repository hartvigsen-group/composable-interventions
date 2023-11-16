import json

def get_edits(number_of_edits=3, file_path='data/counterfact/counterfact-edit.json'):
    # Assuming your JSON data is stored in a file named 'data.json'
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    prompts = []
    ground_truth = []
    target_new = []
    subject = []

    # Extracting the first N edits
    for entry in json_data[:number_of_edits]:
        prompts.append(entry['prompt'])
        ground_truth.append(entry['ground_truth'])
        target_new.append(entry['target_new'])
        subject.append(entry['subject'])

    return prompts, ground_truth, target_new, subject
