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


def F1_locality(model, locality_inputs, config):
    """
    Compute the F1 score for locality inputs using a HuggingFace PyTorch model.

    :param model: The HuggingFace PyTorch model to evaluate.
    :param locality_inputs: A dictionary with 'prompt' and 'ground_truth' keys.
    :param tokenizer: The tokenizer associated with the model.
    :param max_length: Maximum sequence length for tokenization.
    :return: The average F1 score for the inputs.
    """

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length=config.max_length

    # Ensure model is in evaluation mode
    model.eval()

    # List to store F1 scores for each batch
    f1_scores = []

    for prompt, ground_truth in zip(locality_inputs['counterfact']['prompt'], locality_inputs['counterfact']['ground_truth']):
        # Tokenize the prompt
        input_ids = tokenizer(prompt, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.to(model.device)

        # Forward pass, disable gradient calculation
        with torch.no_grad():
            outputs = model(input_ids)

        # Get logits from the model outputs (assuming outputs are logits)
        logits = outputs.logits

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1).view(-1)

        # Tokenize the ground truth for comparison
        ground_truth_ids = tokenizer(ground_truth, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.view(-1)

        # Decode model output
        decoded_prediction = tokenizer.decode(predictions, skip_special_tokens=True)

        # Print for comparison
        print(f"Prompt: {prompt}")
        print(f"Model Output: {decoded_prediction}")
        print(f"Ground Truth: {ground_truth}")
        print("-" * 50)  # Separator for readability

        # Compute F1 score
        f1 = f1_score(ground_truth_ids.cpu().numpy(), predictions.cpu().numpy(), average='weighted')
        f1_scores.append(f1)

    # Return the average F1 score across all batches
    return sum(f1_scores) / len(f1_scores)


def calculate_edit_accuracy(model, prompts, target_new, config, max_length=512):
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
