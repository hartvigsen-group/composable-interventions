import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer


def calculate_avg(data):
    # Initialize sums
    total_rewrite_acc = 0
    total_locality = 0

    # Loop through the data
    for entry in data:
        # Add rewrite accuracy
        total_rewrite_acc += entry['post']['rewrite_acc']

        # Add count of locality outputs
        total_locality += entry['post']['locality']['counterfact_output'][0]

    # Calculate averages
    average_rewrite_acc = total_rewrite_acc / len(data)
    average_locality = total_locality / len(data)

    return average_rewrite_acc, average_locality
    print(f'Average Rewrite Accuracy: {average_rewrite_acc}')
    print(f'Average Locality: {average_locality}')


def f1_locality_logits(model, locality_inputs, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = config.max_length
    model.eval()  # Ensure model is in evaluation mode

    f1_scores = []  # List to store F1 scores for each batch

    for prompt, ground_truth in zip(locality_inputs['counterfact']['prompt'], locality_inputs['counterfact']['ground_truth']):
        input_ids = tokenizer(prompt, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1).view(-1)
        predictions_no_pad = predictions[predictions != tokenizer.pad_token_id]
        decoded_prediction = tokenizer.decode(predictions_no_pad, skip_special_tokens=True)

        ground_truth_ids = tokenizer(ground_truth, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.view(-1)
        ground_truth_ids_no_pad = ground_truth_ids[ground_truth_ids != tokenizer.pad_token_id]
        decoded_ground_truth = tokenizer.decode(ground_truth_ids_no_pad, skip_special_tokens=True)

        print(f"Prompt: {prompt}")
        print(f"Model Output: {decoded_prediction}")
        print(f"Ground Truth: {decoded_ground_truth}")
        print("-" * 50)  # Separator for readability

        num_same = len(set(predictions_no_pad.cpu().numpy()).intersection(set(ground_truth_ids_no_pad.cpu().numpy())))
        if num_same == 0 or len(predictions_no_pad) == 0:
            f1_scores.append(0)
            continue
        precision = num_same / len(predictions_no_pad)
        recall = num_same / len(ground_truth_ids_no_pad)
        print(precision, recall)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        print(f1)
        f1_scores.append(f1)

    print(f1_scores)
    quit()
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0


def f1_locality_generate(model, locality_inputs, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = config.max_length

    model.eval()  # Ensure model is in evaluation mode

    f1_scores = []  # List to store F1 scores for each batch

    for prompt, ground_truth in zip(locality_inputs['counterfact']['prompt'], locality_inputs['counterfact']['ground_truth']):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_length = input_ids.size(1)  # Number of tokens in the prompt

        with torch.no_grad():
            generated_ids = model.generate(input_ids, temperature=1e-6, max_length=max_length)

        ground_truth_ids = tokenizer(ground_truth, return_tensors="pt").input_ids.view(-1)
        generated_ids = generated_ids[:, prompt_length:prompt_length+len(ground_truth_ids)-1]
        
        decoded_prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        decoded_ground_truth = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)

        # print(f"Prompt: {prompt}")
        # print(f"Model Output: {decoded_prediction}")
        # print(generated_ids)
        # print(f"Ground Truth: {decoded_ground_truth}")
        # print(ground_truth_ids[1:])
        # print("-" * 50)  # Separator for readability

        # Convert generated IDs and ground truth IDs to sets for comparison
        generated_set = set(generated_ids[0].cpu().numpy())
        ground_truth_set = set(ground_truth_ids[1:].cpu().numpy())
        num_same = len(generated_set.intersection(ground_truth_set))
        
        if num_same == 0 or len(generated_set) == 0:
            f1_scores.append(0)
            continue
        precision = num_same / len(generated_set)
        recall = num_same / len(ground_truth_set)
        # print(precision, recall)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        # print(f1)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0

def f1_accuracy_generate(model, prompts, target_new, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = config.max_length

    model.eval()

    f1_scores = []

    for prompt, ground_truth in zip(prompts, target_new):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_length = input_ids.size(1)  # Number of tokens in the prompt

        ground_truth_ids = tokenizer(ground_truth, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.view(-1)
        ground_truth_ids_no_pad = ground_truth_ids[(ground_truth_ids != tokenizer.pad_token_id) & (ground_truth_ids != tokenizer.bos_token_id)]
        decoded_ground_truth = tokenizer.decode(ground_truth_ids_no_pad, skip_special_tokens=True)

        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=max_length)
        generated_ids = outputs
        # predictions = generated_ids.view(-1)
        # generated_ids = generated_ids[:, prompt_length:prompt_length+len(ground_truth_ids_no_pad)]
        # predictions_no_pad = predictions[(predictions != tokenizer.pad_token_id) & (predictions != tokenizer.bos_token_id)]
        # decoded_prediction = tokenizer.decode(predictions_no_pad, skip_special_tokens=True)

        # Exclude prompt and filter out special tokens
        generated_response_ids = generated_ids[:, prompt_length:].view(-1)
        generated_response_ids_no_special = generated_response_ids[(generated_response_ids != tokenizer.pad_token_id) & (generated_response_ids != tokenizer.bos_token_id)]

        # Clip to the length of the ground truth, if necessary
        if len(generated_response_ids_no_special) > len(ground_truth_ids_no_pad):
            generated_response_ids_no_special = generated_response_ids_no_special[:len(ground_truth_ids_no_pad)]

        decoded_prediction = tokenizer.decode(generated_response_ids_no_special, skip_special_tokens=True)


        # Print information
        print(f"Prompt: {prompt}")
        print(f"Model Output: {decoded_prediction}")
        print(f"Generated IDs: {generated_response_ids_no_special}")
        print(f"Ground Truth: {decoded_ground_truth}")
        print(f"Ground Truth IDs: {ground_truth_ids_no_pad}")
        print("-" * 50)  # Separator for readability
        print(tokenizer.bos_token_id)
        print(tokenizer.eos_token_id)

        # Calculate F1 score
        common_tokens = set(generated_response_ids_no_special.cpu().numpy()).intersection(set(ground_truth_ids_no_pad.cpu().numpy()))

        if len(common_tokens) == 0 or len(generated_response_ids_no_special) == 0:
            f1_scores.append(0)
            continue

        precision = len(common_tokens) / len(generated_response_ids_no_special)
        recall = len(common_tokens) / len(ground_truth_ids_no_pad)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1)
        print(precision, recall)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0


def calculate_edit_accuracy_logits(model, prompts, target_new, config):
    """
    Calculate the edit accuracy for given lists of prompts and ground truths using a HuggingFace PyTorch model.

    :param model: The HuggingFace PyTorch model to evaluate.
    :param prompts: A list of prompts.
    :param ground_truths: A list of corresponding ground truths.
    :param tokenizer: The tokenizer associated with the model.
    :param max_length: Maximum sequence length for tokenization.
    :return: The edit accuracy rate.
    """

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = config.max_length

    # Ensure model is in evaluation mode
    model.eval()

    # Accumulate total and correct predictions
    total = 0
    correct = 0

    for prompt, target_new in zip(prompts, target_new):
        # Tokenize the prompt
        input_ids = tokenizer(prompt, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.to(model.device)

        # Tokenize the ground truth for comparison
        labels = tokenizer(target_new, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.to(model.device)

        # Forward pass, disable gradient calculation
        with torch.no_grad():
            logits = model(input_ids).logits

        # Compute probabilities and get the index of the maximum probability
        probs = torch.softmax(logits, -1).squeeze()
        argmaxs = torch.argmax(probs, dim=-1).squeeze()

        # Flatten labels and predictions
        labels_flat = labels.view(-1)
        argmaxs_flat = argmaxs.view(-1)

        # Decode model output and target
        decoded_prediction = tokenizer.decode(argmaxs_flat, skip_special_tokens=True)
        decoded_target = tokenizer.decode(labels_flat, skip_special_tokens=True)

        # Print for comparison
        # print(f"Prompt: {prompt}")
        # print(f"Model Output: {decoded_prediction}")
        # print(f"Target: {decoded_target}")
        # print("-" * 50)  # Just for better readability

        # Update correct and total count
        correct += (labels_flat == argmaxs_flat).float().sum()
        total += labels_flat.size(0)

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    return accuracy.item()

def calculate_edit_accuracy(model, prompts, target_new, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = config.max_length

    model.eval()

    total = 0
    correct = 0

    for prompt, target in zip(prompts, target_new):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_length = input_ids.size(1)

        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_length=max_length)
        generated_part = generated_ids[0, prompt_length:]
        decoded_prediction = tokenizer.decode(generated_part, skip_special_tokens=True)

        labels = tokenizer(target, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.to(model.device)
        labels_flat = labels.view(-1)

        decoded_target = tokenizer.decode(labels_flat, skip_special_tokens=True)

        # Compute edit accuracy
        num_same = len(set(generated_part.cpu().numpy()).intersection(set(labels_flat.cpu().numpy())))
        total += labels_flat.size(0)
        correct += num_same

    accuracy = correct / total if total > 0 else 0
    return accuracy


def calculate_success_metrics(model, prompts, ground_truth, target_new, locality_inputs, rephrase_prompt, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=True)
    model.eval()

    def get_probability(text, continuation):
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
        continuation_ids = tokenizer.encode(continuation, add_special_tokens=False, return_tensors="pt").to(model.device)

        # Check if continuation_ids are empty
        if continuation_ids.nelement() == 0:
            print(f"Error: Empty continuation_ids for continuation '{continuation}'")
            return 0

        # Ensure batch sizes match
        if input_ids.size(0) != continuation_ids.size(0):
            print(f"Error: Batch size mismatch. Input IDs batch size: {input_ids.size(0)}, Continuation IDs batch size: {continuation_ids.size(0)}")
            return 0

        # Debug prints
        print("Input IDs:", input_ids)
        print("Continuation IDs:", continuation_ids)

        with torch.no_grad():
            outputs = model(input_ids, labels=continuation_ids)
        logits = outputs.logits[:, -len(continuation_ids[0]):]
        return torch.softmax(logits, dim=-1)[0, -1, continuation_ids[0][-1]].item()


    es_count, ps_count, ns_count = 0, 0, 0
    total = len(prompts)

    for i in range(total):
        print(f"Processing prompt {i}: {prompts[i]}")  # Debug
        es_prob_gt = get_probability(prompts[i], ground_truth[i])
        es_prob_tn = get_probability(prompts[i], target_new[i])
        if es_prob_tn > es_prob_gt:
            es_count += 1

        for rephrased in rephrase_prompt[i]:
            print(f"Rephrased prompt: {rephrased}")  # Debug
            ps_prob_gt = get_probability(rephrased, ground_truth[i])
            if es_prob_tn > ps_prob_gt:
                ps_count += 1

        for related in locality_inputs['counterfact']['prompt']:
            print(f"Related prompt: {related}")  # Debug
            ns_prob_gt = get_probability(related, locality_inputs['counterfact']['ground_truth'][i])
            if ns_prob_gt > es_prob_gt:
                ns_count += 1

    es = es_count / total
    ps = ps_count / total
    ns = ns_count / total

    return es, ps, ns
