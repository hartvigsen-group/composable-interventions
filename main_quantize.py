import argparse
import os 
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import copy
import os
import gc
from torch.nn.functional import pad
from sparsellm.lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers,AverageBits
from sparsellm.lib.eval import eval_ppl, eval_zero_shot
from sparsellm.lib.data import get_c4
###GPTQ########
from sparsellm.lib.gptq import *
from sparsellm.lib.modelutils import *
from sparsellm.lib.quant import *
#from sparsellm.lib.quant_llama import llama_pack3, llamaQuanti,llama_eval
#from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq import AutoAWQForCausalLM

DEV = torch.device('cuda:0')
from calflops.calflops import calculate_flops
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.utils import Perplexity
class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy


def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))

class LLMPruningAndValidation:
    def __init__(self, args, model=None):
        self.args = args
        # if args.save_model is None:
        #     args.save_model="/scratch-shared/HTJ/"+args.model_name+"_"+args.method+"_"+args.quant_method+"_"+args.prune_method
        self.device = torch.device("cuda:0")
        if model:
            self.model = model
        
        self.get_llm(args.model_name)
        if model is not None:
            model=model.to(self.device)
            if self.args.method=='quant':
                del self.model4Quant.model 
                torch.cuda.empty_cache()
                self.model4Quant.model=model
            #else:    
            self.model=model
        self.model.seqlen = self.model.config.max_position_embeddings
        if self.args.method=='quant':
            self.model4Quant.model.seqlen=self.model4Quant.model.config.max_position_embeddings
        #self.original_model=copy.deepcopy(self.model)           ####Here i do copy for the model in cause the editing operation need the whole weights. Note: Prune process do not need this.
        if self.model.config.model_type=='gpt_neox' or self.model.config.model_type=='gptj':
            use_fast=True
        else:
            use_fast=False
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=use_fast)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        self.Masks=None

    def get_llm(self, model_name, cache_dir="llm_weights"):
        args=self.args
        print(self.args.method)
        if self.args.method=='quant':
            print(self.args.quant_method)
            if self.args.quant_method=='autogptq':
                quantize_config = BaseQuantizeConfig(
                    bits=args.wbits,  # quantize model to 4-bit
                    group_size=args.groupsize,  # it is recommended to set the value to 128
                    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
                )
                # load un-quantized model, by default, the model will always be loaded into CPU memory
                model = AutoGPTQForCausalLM.from_pretrained(self.args.model_name, quantize_config, trust_remote_code=True)
                self.model=model.model.to(self.device)
                self.model4Quant=model
            elif self.args.quant_method=='autoawq':
                #quant_config={ "zero_point": args.zero_point, "q_group_size": args.groupsize, "w_bit": args.wbits, "version": "GEMM" }
                model = AutoAWQForCausalLM.from_pretrained(self.args.model_name, **{"low_cpu_mem_usage": True}, safetensors=True, trust_remote_code=True)
                #print(model)
                self.model=model.model.to(self.device)
                self.model4Quant=model
                #print(self.model4Quant)
            else:
                print('Incorrect method and quant_method combination.')
                sys.exit()
        else:
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     model_name,
            #     torch_dtype=torch.float16, 
            #     low_cpu_mem_usage=True, 
            #     device_map="auto"
            # ).to(self.device)
            self.model.seqlen = self.model.config.max_position_embeddings
    def get_average_number_of_bits4Quantization(self,
        wbits: int = 4,
        qq_scale_bits: int = 16,
        qq_zero_bits: int = 16,
        groupsize: int = 128,
    ):
        # if not quantized stats are in full precision
        qq_scale_bits = qq_scale_bits or 16
        qq_zero_bits = qq_zero_bits or 16
        groupsize = groupsize or float("inf")

        if groupsize is None:
            wbits_avg = wbits
        else:
            wbits_avg = (
                wbits
                +  (qq_scale_bits + qq_zero_bits) / (groupsize )
            )
        
        return round(wbits_avg, 2)   
    def quantization(self):
        args=self.args
        if args.quant_method=='autogptq':
            quantize_config = BaseQuantizeConfig(
                bits=args.wbits,  # quantize model to 4-bit
                group_size=args.groupsize,  # it is recommended to set the value to 128
                desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
            )
            # load un-quantized model, by default, the model will always be loaded into CPU memory
            #model = AutoGPTQForCausalLM.from_pretrained(self.args.model_name, quantize_config)
            # examples = [
            #     self.tokenizer(
            #         "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            #     )
            # ]
            examples,_=get_c4(args.nsamples,0,self.model.seqlen,self.tokenizer)
            examples=[{"input_ids":each[0],"attention_mask":torch.ones_like(each[0])} for each in examples]

            # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
            self.model4Quant.quantize(examples)
            # self.model4Quant.save_quantized(args.save_model)
            print("Post init now.")
            self.model4Quant.post_init()
            self.model4Quant.model=self.model4Quant.model.to(self.device)
            self.model=self.model4Quant.model


        elif args.quant_method=='autoawq':
            quant_config={ "zero_point": args.zero_point, "q_group_size": args.groupsize, "w_bit": args.wbits, "version": "GEMM" }
            self.model4Quant.quantize(self.tokenizer,quant_config=quant_config,calib_data="pileval")
            # self.model4Quant.save_quantized(args.save_model)
            self.model4Quant.model=self.model4Quant.model.to(self.device)
            self.model=self.model4Quant.model       
        else:
            print("Not implemented Yet!")
            assert False
    def average_bits(self):
        print("*" * 30)
        if self.args.method=='quant':
            averageBits=self.get_average_number_of_bits4Quantization(self.args.wbits,groupsize=self.args.groupsize)
        else:
            averageBits=AverageBits(self.model)
        print(f"average Bits check {averageBits:.4f}")
        print("*" * 30)
        return averageBits 
    def pseudoQuantization(self,flag=False):
        args=self.args
        # if args.quant_method=='gptq':
        #     model,quantizers=llamaQuanti(self.model,self.device,self.args)
        #     self.model=model
        if args.quant_method=='autogptq':
            quantize_config = BaseQuantizeConfig(
                bits=args.wbits,  # quantize model to 4-bit
                group_size=args.groupsize,  # it is recommended to set the value to 128
                desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
            )
            # load un-quantized model, by default, the model will always be loaded into CPU memory
            #model = AutoGPTQForCausalLM.from_pretrained(self.args.model_name, quantize_config)
            # examples = [
            #     self.tokenizer(
            #         "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            #     )
            # ]
            print("C4 Data for autoGPTQ")
            examples,_=get_c4(args.nsamples,0,self.model.seqlen,self.tokenizer)
            examples=[{"input_ids":each[0],"attention_mask":torch.ones_like(each[0])} for each in examples]

            # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
            self.model4Quant.pseudoQuantize(examples)

            self.model4Quant.model=self.model4Quant.model.to(self.device)
            self.model4Quant.post_init()
            self.model=self.model4Quant.model           
        elif args.quant_method=='autoawq':
            quant_config={ "zero_point": args.zero_point, "q_group_size": args.groupsize, "w_bit": args.wbits, "version": "GEMM" }
            #model = AutoAWQForCausalLM.from_pretrained(self.args.model_name, **{"low_cpu_mem_usage": True})
            self.model4Quant.pseudoQuantize(self.tokenizer,quant_config=quant_config)
            self.model4Quant.model=self.model4Quant.model.to(self.device)
            self.model=self.model4Quant.model           
        else:
            print("Not implemented Yet!")
            assert False
    def get_Mask(self):
        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        if "30b" in args.model_name or "65b" in args.model_name:
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
        if self.model.config.model_type=='gptj':
            layers=self.model.transformer.h
        elif self.model.config.model_type=='gpt_neox':
            layers=self.model.gpt_neox.layers
        else:
            layers = self.model.model.layers
        if self.args.sparsity_ratio!=0: 
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
        return sparsity_ratio
    def FLOPs(self):
        assert self.args.method!='quant'
        batch_size, max_seq_length = 1, 128
        flops,macs,params=calculate_flops(model=self.model,input_shape=(batch_size,max_seq_length),transformer_tokenizer=self.tokenizer,is_sparse=True)
        print("FLOPs:%s, MACs:%s, Params:%s \n"%(flops,macs,params))
        return flops
        #$pass  

    def foward(self,input):
        return self.model(input)
    def CalculateLatency(self, model):
        from datasets import load_dataset
        dataset = load_dataset('lambada', split='validation[:1000]')
        evaluator = Evaluator(dataset, self.tokenizer)
        # quant_dir=self.args.save_model
        
        # if self.args.quant_method=='autogptq':
        #     model = AutoGPTQForCausalLM.from_quantized(quant_dir, device="cuda:0")
        # elif self.args.quant_method=='autoawq':
        #     model = AutoAWQForCausalLM.from_quantized(quant_dir,"", fuse_layers=False)
        acc_smoothquant, lantecy = evaluator.evaluate(model)
        print(f'per-sample lantecy: {lantecy:.3f} ms')
        return lantecy
    def validate(self,normal_test=True):
        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        device = self.device
        if normal_test:
            ppl_test = eval_ppl(args, model, tokenizer, device)
            print(f"wikitext perplexity {ppl_test}")
        else:
            if args.method=='quant':
                if args.quant_method=='autogptq':
                    print("start post init")
                    
                    self.model4Quant.post_init()
                    print("post init end")
                    model=self.model4Quant.model.to(self.device)
                    print("dataset",args.dataset)
                    ppl_test = eval_ppl(args, model, tokenizer, device)
                    print(f"wikitext perplexity {ppl_test}")
                elif args.quant_method=='autoawq':
                    print("dataset",args.dataset)
                    ppl_test = eval_ppl(args, model, tokenizer, device)
                    print(f"wikitext perplexity {ppl_test}")
            elif args.method=='sparse':
                ppl_test = eval_ppl(args, model, tokenizer, device)
                print(f"wikitext perplexity {ppl_test}")

        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            if type(ppl_test) is not float:
                ppl_test=ppl_test['results'][args.dataset]['word_perplexity']
            print(f"{ppl_test:.4f}", file=f, flush=True)

        if args.eval_zero_shot:
            accelerate=False
            if "30b" in args.model_name or "65b" in args.model_name or "70b" in args.model_name:
                accelerate=True
            task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
            num_shot = 0
            results = eval_zero_shot(args.model_name, model, tokenizer, task_list, num_shot, accelerate)
            print("zero_shot evaluation results")
            print(results)

        #if args.save_model:
        #    model.save_pretrained(args.save_model)
        #    tokenizer.save_pretrained(args.save_model)
        return ppl_test
    def Edit(self):
        pass
def get_args(parser):
    parser.add_argument('--model', type=str,default='meta-llama/Llama-2-7b-chat-hf', help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level between 0 and 1')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str,default='wanda', choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default="", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--eval_zero_shot", action="store_true")
     ############For Quantization##########################################
    parser.add_argument(
        '--quant_method', type=str, default='autogptq', choices=['autoawq','autogptq'],
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
        '--groupsize', type=int, default=128,
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
    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument(
        '--act-order', action='store_true',default=True,
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument("--split", type=str, default='test', help="Dataset split to use.")
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
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser=get_args(parser)
    args = parser.parse_args()
    
    pruning_and_validation = LLMPruningAndValidation(args)
    ##Test Sparse##########
    # pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
    # pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 
    # pruning_and_validation.validate()           #It is a validation for general performance on common language benchmark such as wikitext.
    ###Test Quantization###########
    print("Starting Quantization")
    pruning_and_validation.quantization()
    print("Starting Validating")
    pruning_and_validation.validate()
    pruning_and_validation.CalculateLatency()    