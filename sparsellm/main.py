import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import copy
from sparsellm.lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers,AverageBits
from sparsellm.lib.eval import eval_ppl, eval_zero_shot
from awq import AutoAWQForCausalLM
###GPTQ########
from lib.gptq import *
from lib.modelutils import *
from lib.quant import *
from lib.quant_llama import llama_pack3, llamaQuanti,llama_eval
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
DEV = torch.device('cuda:0')

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
    def quantization(self):
        args=self.args
        if args.quant_method=='gptq':
            model,quantizers=llamaQuanti(self.model,self.device,self.args)
            #self.model=llama_pack3(model, quantizers)
            self.model=model
        elif args.quant_method=='awq':
            quant_config={ "zero_point": args.zero_point, "q_group_size": args.groupsize, "w_bit": args.wbits, "version": "GEMM" }
            model = AutoAWQForCausalLM.from_pretrained(self.args.model, **{"low_cpu_mem_usage": True})
            model.quantize(self.tokenizer,quant_config=quant_config)
            model.save_quantized(args.save_model)
            self.model=model           
        else:
            print("Not implemented Yet!")
            assert False
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
    def average_bits(self):
        print("*" * 30)
        averageBits=AverageBits(self.model)
        print(f"average Bits check {averageBits:.4f}")
        print("*" * 30) 
    def foward(self,input):
        return self.model(input)

    def validate(self):
        args = self.args
        model = self.model.to(self.device)
        tokenizer = self.tokenizer
        device = self.device

        if args.method=='quant':
            if args.quant_method=='gptq':
                ppl_test = llama_eval(model, device,args)
            elif args.quant_method=='awq':
                print("dataset",args.dataset)
                lm_eval_model = LMEvalAdaptor(self.args.model, self.model, self.tokenizer, self.device, batch_size=1)
                results = evaluator.simple_evaluate(
                        model=lm_eval_model,
                        tasks=[self.args.dataset],
                        batch_size=1,
                        no_cache=True,
                        num_fewshot=0,
                    )
                ppl_test=results
        elif args.method=='sparse':
            ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"wikitext perplexity {ppl_test['results'][args.dataset]['word_perplexity']}")

        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{ppl_test['results'][args.dataset]['word_perplexity']:.4f}", file=f, flush=True)

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


if __name__ == '__main__':
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
     ############For Quantization##########################################
    parser.add_argument(
        '--quant_method', type=str, default='gptq', choices=['gptq','awq'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--method', type=str, default='sparse', choices=['sparse','quant'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--dataset', type=str,default='wikitext', choices=['wikitext2','wikitext','ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[3, 4],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )

    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    ############################################################################################################
    parser.add_argument(
        '--zero_point', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    # Add all your argparse arguments here...
    args = parser.parse_args()
    
    pruning_and_validation = LLMPruningAndValidation(args)
    ##Test Sparse##########
    #pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
    #pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 
    #pruning_and_validation.validate()           #It is a validation for general performance on common language benchmark such as wikitext.
    ####Test Quantization###########
    pruning_and_validation.quantization()
    pruning_and_validation.validate()    
