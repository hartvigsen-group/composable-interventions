import torch
from lm_compose.utils import edit_generator
from transformers import AutoTokenizer


def get_exclude_tokens(tokenizer, device):
    space_tok = tokenizer(" Hello", return_tensors="np", add_special_tokens=False)["input_ids"][0][0]
    space_tok_check = tokenizer(" Word", return_tensors="np", add_special_tokens=False)["input_ids"][0][0]
    real_space_tok = tokenizer(" ", return_tensors="np", add_special_tokens=False)["input_ids"][0][0]

    if space_tok == space_tok_check:
        exclude_tokens = [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            space_tok,
            real_space_tok,
        ]
    else:
        exclude_tokens = [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            real_space_tok,
        ]

    return torch.tensor(exclude_tokens, device=device)


def get_f1(common_tokens, generated_response_ids_no_special, ground_truth_ids_no_special):
    generated_response_ids_no_special = set(generated_response_ids_no_special.cpu().numpy())
    ground_truth_ids_no_special = set(ground_truth_ids_no_special.cpu().numpy())

    precision = len(common_tokens) / len(generated_response_ids_no_special)
    recall = len(common_tokens) / len(ground_truth_ids_no_special)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1


def calculate_recall(generated_ids, ground_truth_ids, exclude_tokens_tensor, prompt, ground_truth):
    # Exclude special tokens for generated_ids
    # generated_response_ids_no_special = generated_ids[~(generated_ids[..., None] == exclude_tokens_tensor).any(-1)]

    # Calculate Recall
    common_tokens = set(generated_ids.cpu().numpy()).intersection(set(ground_truth_ids.cpu().numpy()))
    recall = len(common_tokens) / len(set(ground_truth_ids.cpu().numpy())) if len(ground_truth_ids) > 0 else 0

    return recall


def f1_locality_generate(model, locality_inputs, config, verbose=False):
    return f1_accuracy_generate(
        model,
        locality_inputs["data"]["prompt"],
        locality_inputs["data"]["ground_truth"],
        config,
        verbose=verbose,
    )


def f1_accuracy_generate(model, prompts, target_new, config, verbose=False):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = config.max_length

    model.eval()

    f1_scores = []
    recall_scores = []  # List to store recall scores

    exclude_tokens_tensor = get_exclude_tokens(tokenizer, model.device)

    for prompt, ground_truth in zip(prompts, target_new):
        if len(ground_truth) > 0:
            if ground_truth[0] != " ":
                # Space required for correct tokenization
                ground_truth = " " + ground_truth
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_length = input_ids.size(1)  # Number of tokens in the prompt

        ground_truth_ids = (
            tokenizer(
                ground_truth,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            .input_ids.view(-1)
            .to(model.device)
        )

        ground_truth_ids_no_special = ground_truth_ids[~(ground_truth_ids[..., None] == exclude_tokens_tensor).any(-1)]

        decoded_ground_truth = tokenizer.decode(ground_truth_ids_no_special, skip_special_tokens=False)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                top_k=1,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs

        # Exclude prompt and filter out special tokens
        generated_response_ids = generated_ids[:, prompt_length:].view(-1)
        generated_response_ids_no_special = generated_response_ids[~(generated_response_ids[..., None] == exclude_tokens_tensor).any(-1)]

        if verbose:
            print(f"Generated IDs:{generated_response_ids_no_special}")
            print(f"Ground Truth IDs:{ground_truth_ids_no_special}")
            print("--")
        # Clip to the length of the ground truth, if necessary
        if len(generated_response_ids_no_special) > len(ground_truth_ids_no_special):
            generated_response_ids_no_special = generated_response_ids_no_special[: len(ground_truth_ids_no_special)]

        decoded_prediction = tokenizer.decode(generated_response_ids_no_special, skip_special_tokens=True)

        # Calculate recall
        recall = calculate_recall(
            generated_response_ids_no_special,
            ground_truth_ids_no_special,
            exclude_tokens_tensor,
            prompt,
            ground_truth,
        )
        recall_scores.append(recall)
        # Calculate F1 score
        common_tokens = set(generated_response_ids_no_special.cpu().numpy()).intersection(set(ground_truth_ids_no_special.cpu().numpy()))

        if len(common_tokens) == 0 or len(generated_response_ids_no_special) == 0:
            f1 = 0
        else:
            f1 = get_f1(
                common_tokens,
                generated_response_ids_no_special,
                ground_truth_ids_no_special,
            )
        f1_scores.append(f1)

        # Print info
        if verbose:
            print(f"Prompt:{prompt}")
            print(f"Model Output:{decoded_prediction}")
            print(f"Generated IDs:{generated_response_ids}")
            print(f"Generated IDs:{generated_response_ids_no_special}")
            print(f"Ground Truth:{decoded_ground_truth}")
            print(f"Ground Truth IDs:{ground_truth_ids}")
            print(f"Ground Truth IDs:{ground_truth_ids_no_special}")
            print(f"F1: {f1}")
            print(f"Recall: {recall}")
            print("-" * 50)  # Separator for readability

    return (
        sum(f1_scores) / len(f1_scores),
        sum(recall_scores) / len(recall_scores) if f1_scores else 0,
    )


def ppl_responses(model, prompts, responses, config, mask_prompt=True):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    total_loss = 0
    total_response_tokens = 0

    for prompt, response in zip(prompts, responses):
        # Tokenize prompt and response
        prompt_encodings = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        response_encodings = tokenizer(response, add_special_tokens=True, return_tensors="pt")

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
        total_response_tokens += input_ids.size(1) - prompt_length
        # print(torch.exp(torch.tensor(neg_log_likelihood / (input_ids.size(1) - prompt_length), device=device)))
    # Calculate average perplexity
    avg_perplexity = torch.exp(torch.tensor(total_loss / total_response_tokens, device=device))

    return avg_perplexity.item()


def ppl_QA(model, config, mask_prompt=False):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model.to(f"cuda:{config.device}")
    model.eval()
    (
        prompts,
        ground_truth,
        target_new,
        subject,
        rephrase_prompt,
        locality_inputs,
    ) = edit_generator.get_edits(dataset=config.edit_dataset, number_of_edits=500, edit_set=2)

    total_loss = 0.0
    total_response_tokens = 0

    for prompt, response in zip(prompts, ground_truth):
        # Tokenize prompt and response
        prompt_encodings = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(f"cuda:{config.device}")
        response_encodings = tokenizer(response, add_special_tokens=True, return_tensors="pt").to(f"cuda:{config.device}")

        # Concatenate prompt and response tokens
        input_ids = torch.cat([prompt_encodings.input_ids, response_encodings.input_ids[:, :]], dim=-1).to(f"cuda:{config.device}")

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
        total_response_tokens += input_ids.size(1) - prompt_length
        # print(torch.exp(torch.tensor(neg_log_likelihood / (input_ids.size(1) - prompt_length), device=device)))
    # Calculate average perplexity
    avg_perplexity = torch.exp(torch.tensor(total_loss / total_response_tokens, device=f"cuda:{config.device}"))

    return avg_perplexity.item()
