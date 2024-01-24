import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


def calculate_recall(generated_ids, ground_truth_ids, exclude_tokens_tensor, prompt, ground_truth):
    # # Calculate the cutoff length
    # cutoff_length = max(15, len(prompt) + len(ground_truth))

    # # Apply cutoff to generated_ids
    # generated_ids_cut = generated_ids[:, :cutoff_length].view(-1)

    # Exclude special tokens for generated_ids
    generated_response_ids_no_special = generated_ids[~(generated_ids[..., None] == exclude_tokens_tensor).any(-1)]

    # Calculate Recall
    common_tokens = set(generated_response_ids_no_special.cpu().numpy()).intersection(set(ground_truth_ids.cpu().numpy()))

    recall = len(common_tokens) / len(ground_truth_ids) if len(ground_truth_ids) > 0 else 0

    return recall

def f1_locality_generate(model, locality_inputs, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = config.max_length

    model.eval()  # Ensure model is in evaluation mode
    exclude_tokens = [tokenizer.pad_token_id, tokenizer.bos_token_id, 13]  # 13 is the new line token for llama
    exclude_tokens_tensor = torch.tensor(exclude_tokens, device=model.device)
        
    f1_scores = []  # List to store F1 scores for each batch
    recall_scores = []

    for prompt, ground_truth in zip(locality_inputs['data']['prompt'], locality_inputs['data']['ground_truth']):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_length = input_ids.size(1)  # Number of tokens in the prompt

        ground_truth_ids = tokenizer(ground_truth, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.view(-1)
        ground_truth_ids_no_pad = ground_truth_ids[(ground_truth_ids != tokenizer.pad_token_id) & (ground_truth_ids != tokenizer.bos_token_id)]
        decoded_ground_truth = tokenizer.decode(ground_truth_ids_no_pad, skip_special_tokens=True)

        with torch.no_grad():
            outputs = model.generate(input_ids, top_k=1, max_length=max_length)
        generated_ids = outputs
        # predictions = generated_ids.view(-1)
        # generated_ids = generated_ids[:, prompt_length:prompt_length+len(ground_truth_ids_no_pad)]
        # predictions_no_pad = predictions[(predictions != tokenizer.pad_token_id) & (predictions != tokenizer.bos_token_id)]
        # decoded_prediction = tokenizer.decode(predictions_no_pad, skip_special_tokens=True)

        # Exclude prompt and filter out special tokens
        generated_response_ids = generated_ids[:, prompt_length:].view(-1)
        generated_response_ids_no_special = generated_response_ids[~(generated_response_ids[..., None] == exclude_tokens_tensor).any(-1)]
        
        # Clip to the length of the ground truth, if necessary
        if len(generated_response_ids_no_special) > len(ground_truth_ids_no_pad):
            generated_response_ids_no_special = generated_response_ids_no_special[:len(ground_truth_ids_no_pad)]

        decoded_prediction = tokenizer.decode(generated_response_ids_no_special, skip_special_tokens=True)

        # Calculate recall
        recall = calculate_recall(outputs, ground_truth_ids_no_pad, exclude_tokens_tensor, prompt, ground_truth)
        recall_scores.append(recall)


        # Print information
        print(f"Prompt: {prompt}")
        print(f"Model Output: {decoded_prediction}")
        print(f"Generated IDs: {generated_response_ids_no_special}")
        print(f"Ground Truth: {decoded_ground_truth}")
        print(f"Ground Truth IDs: {ground_truth_ids_no_pad}")
        print("-" * 50)  # Separator for readability

        # Calculate F1 score
        common_tokens = set(generated_response_ids_no_special.cpu().numpy()).intersection(set(ground_truth_ids_no_pad.cpu().numpy()))

        if len(common_tokens) == 0 or len(generated_response_ids_no_special) == 0:
            f1_scores.append(0)
            continue

        precision = len(common_tokens) / len(generated_response_ids_no_special)
        recall = len(common_tokens) / len(ground_truth_ids_no_pad)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores), sum(recall_scores) / len(recall_scores) if f1_scores else 0

def f1_accuracy_generate(model, prompts, target_new, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = config.max_length

    model.eval()

    f1_scores = []
    recall_scores = []  # List to store recall scores

    exclude_tokens = [tokenizer.pad_token_id, tokenizer.bos_token_id, 13]  # 13 is the new line token for llama
    exclude_tokens_tensor = torch.tensor(exclude_tokens, device=model.device)

    for prompt, ground_truth in zip(prompts, target_new):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_length = input_ids.size(1)  # Number of tokens in the prompt

        ground_truth_ids = tokenizer(ground_truth, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").input_ids.view(-1)
        ground_truth_ids_no_pad = ground_truth_ids[(ground_truth_ids != tokenizer.pad_token_id) & (ground_truth_ids != tokenizer.bos_token_id)]
        decoded_ground_truth = tokenizer.decode(ground_truth_ids_no_pad, skip_special_tokens=True)

        with torch.no_grad():
            outputs = model.generate(input_ids, top_k=1, max_length=max_length)
        generated_ids = outputs
        # predictions = generated_ids.view(-1)
        # generated_ids = generated_ids[:, prompt_length:prompt_length+len(ground_truth_ids_no_pad)]
        # predictions_no_pad = predictions[(predictions != tokenizer.pad_token_id) & (predictions != tokenizer.bos_token_id)]
        # decoded_prediction = tokenizer.decode(predictions_no_pad, skip_special_tokens=True)

        # Exclude prompt and filter out special tokens
        generated_response_ids = generated_ids[:, prompt_length:].view(-1)
        generated_response_ids_no_special = generated_response_ids[~(generated_response_ids[..., None] == exclude_tokens_tensor).any(-1)]
        
        # Clip to the length of the ground truth, if necessary
        if len(generated_response_ids_no_special) > len(ground_truth_ids_no_pad):
            generated_response_ids_no_special = generated_response_ids_no_special[:len(ground_truth_ids_no_pad)]

        decoded_prediction = tokenizer.decode(generated_response_ids_no_special, skip_special_tokens=True)

        # Calculate recall
        recall = calculate_recall(outputs, ground_truth_ids_no_pad, exclude_tokens_tensor, prompt, ground_truth)
        recall_scores.append(recall)

        # Print information
        print(f"Prompt: {prompt}")
        print(f"Model Output: {decoded_prediction}")
        print(f"Generated IDs: {generated_response_ids_no_special}")
        print(f"Ground Truth: {decoded_ground_truth}")
        print(f"Ground Truth IDs: {ground_truth_ids_no_pad}")
        print(f"Recall: {recall}")
        print("-" * 50)  # Separator for readability

        # Calculate F1 score
        common_tokens = set(generated_response_ids_no_special.cpu().numpy()).intersection(set(ground_truth_ids_no_pad.cpu().numpy()))

        if len(common_tokens) == 0 or len(generated_response_ids_no_special) == 0:
            f1_scores.append(0)
            continue

        precision = len(common_tokens) / len(generated_response_ids_no_special)
        recall = len(common_tokens) / len(ground_truth_ids_no_pad)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores), sum(recall_scores) / len(recall_scores) if f1_scores else 0


def ppl_responses(model, prompts, responses, config, mask_prompt=True):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total_loss = 0
    total_response_tokens = 0

    for prompt, response in zip(prompts, responses):
        # Tokenize prompt and response
        prompt_encodings = tokenizer(prompt, add_special_tokens=True, return_tensors='pt')
        response_encodings = tokenizer(response, add_special_tokens=True, return_tensors='pt')

        # Concatenate prompt and response tokens
        input_ids = torch.cat([prompt_encodings.input_ids, response_encodings.input_ids[:, :]], dim=-1).to(device)

        # Prepare target_ids with prompt tokens masked
        target_ids = input_ids.clone()
        prompt_length = prompt_encodings.input_ids.size(1)
        if mask_prompt:
            target_ids[:, :prompt_length] = -100  # Masking prompt tokens

        # Calculate loss for response tokens
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * (input_ids.size(1) - prompt_length)
        
        total_loss += neg_log_likelihood.item()
        total_response_tokens += (input_ids.size(1) - prompt_length)
        # print(torch.exp(torch.tensor(neg_log_likelihood / (input_ids.size(1) - prompt_length), device=device)))
    # Calculate average perplexity
    avg_perplexity = torch.exp(torch.tensor(total_loss / total_response_tokens, device=device))


    return avg_perplexity.item()