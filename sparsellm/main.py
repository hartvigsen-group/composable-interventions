import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import copy
from sparsellm.lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from sparsellm.lib.eval import eval_ppl, eval_zero_shot


class LLMPruningAndValidation:
    def __init__(self, args, model):
        self.args = args
        self.device = torch.device("cuda:0")
        if model is None:
            self.model = self.get_llm(args.model, args.cache_dir)
        else:
            self.model = model
        self.model.seqlen = self.model.config.max_position_embeddings
        #self.original_model=copy.deepcopy(self.model)           ####Here i do copy for the model in cause the editing operation need the whole weights. Note: Prune process do not need this.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        self.Masks=None

    def get_llm(self, model_name, cache_dir="llm_weights"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            # cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
        model.seqlen = model.config.max_position_embeddings
        return model

    def get_Mask(self):
        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        if "30b" in args.model or "65b" in args.model:
            device = model.hf_device_map["lm_head"]
        print("use device ", device)

        if args.sparsity_ratio != 0:
            print("pruning starts")
            if args.prune_method == "wanda":
                self.Masks=prune_wanda(args, model, tokenizer, device)
            elif args.prune_method == "magnitude":
                self.Masks=prune_magnitude(args, model, tokenizer, device)
            elif args.prune_method == "sparsegpt":
                self.Masks=prune_sparsegpt(args, model, tokenizer, device)
            elif "ablate" in args.prune_method:
                self.Masks=prune_ablate(args, model, tokenizer, device)
    def prune(self):
        layers = self.model.model.layers 
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                subset[name].weight.data[self.Masks["Layer"+str(i)+"_"+name]]=0
    def sparsity_check(self):
        print("*" * 30)
        sparsity_ratio = check_sparsity(self.model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*" * 30)  

    def foward(self,input):
        return self.model(input)

    def validate(self):
        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"wikitext perplexity {ppl_test}")

        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

        if args.eval_zero_shot:
            accelerate=False
            if "30b" in args.model or "65b" in args.model or "70b" in args.model:
                accelerate=True

            task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
            num_shot = 0
            results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
            print("zero_shot evaluation results")
            print(results)

        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
    def Edit(self):
        pass
        #editing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level between 0 and 1')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    # Add all your argparse arguments here...
    args = parser.parse_args()

    pruning_and_validation = LLMPruningAndValidation(args)
    pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
    pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 
    pruning_and_validation.validate()           #It is a validation for general performance on common language benchmark such as wikitext.
