import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, now, unpad
from verl.protocol import DataProto
import json
logger = logging.getLogger(__file__)
logger.setLevel('INFO')
import math
import re
import numpy as np
from collections import Counter
from rank_bm25 import BM25Okapi
from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import QwenFnCallPrompt, FN_STOP_WORDS




#######################################################################
############################## TEMPLATE ###############################
#######################################################################

TEMPLATE_CALL_OR_ANSWER = """
You are a Retrieval Planner.  
Your ONLY task is to decide whether to perform another retrieval using `retrievesearch`, or STOP retrieving.

You MUST NOT answer the QUESTION.  
Another model will use MEMORY to answer later.

Guidelines:
- Retrieval is cheap. Unless MEMORY clearly contains all essential information, you are encouraged to retrieve.
- You may retrieve multiple times. At each step, refine your search direction.
- Avoid repeating any previous queries in RETRIEVAL_HISTORY (unless meaningfully refined).

When deciding whether to retrieve again:
1. Break the QUESTION into specific sub-questions or information needs.
2. Compare these needs with what MEMORY already contains.
3. Identify which facts are still missing, uncertain, or incomplete.
4. If something important is missing, design a NEW search query focused only on that missing information.
   - You may explore related clues hinted in MEMORY.
   - Queries should be concise, specific, and actionable.
5. If MEMORY already contains all necessary information, choose to STOP.

If you choose retrieval, you MUST output a function call to `retrievesearch` with:
- a new `query` (different from RETRIEVAL_HISTORY unless refined),
- and a `top_k` suited to your confidence (small: focused; large: broad exploration).
- In early retrieval steps, you may exlore more documents.
- In later steps, focus on refining MEMORY.


<question>
{prompt}
</question>

<retrieval_history>
{hist}
</retrieval_history>




<memory>
{memory}
</memory>

"""



TEMPLATE_RETRIEVE = """You are presented with a problem, the history of previous retrieve call, a section of retrieved article that may contain the answer to the problem, and a previous memory. 
Please read the provided section carefully and update the memory with the new information that helps to answer the problem, or help the retirever. the given section is retrieved chunk, which are retrieved by given question.

your memory will also be given to the model who retrieve the information. You are also encouraged to conclude some note to help the retriever recall what information might be relevant in later steps. 


<problem> 
{prompt} 
</problem> 

<retrieval_history> 
{hist} 
</retrieval_history> 

<retrieved_chunk> 
{retrieve} 
</retrieved_chunk> 

<memory> 
{memory} 
</memory>

Updated memory:
"""





TEMPLATE_RETRIEVE_RECURRENT = """You are presented with a problem, the history of previous retrieve call, a section of retrieved article that may contain the answer to the problem, and a previous memory. 
Please read the provided section carefully and update the memory with the new information that helps to answer the problem, or help the retirever. the given section is retrieved chunk, which are retrieved by given question.

your memory will also be given to the model who retrieve the information. You are also encouraged to conclude some note to help the retriever recall what information might be relevant in later steps. 


<problem> 
{prompt} 
</problem> 

<retrieval_history> 
{hist} 
</retrieval_history> 

<retrieved_chunk> 
{retrieve} 
</retrieved_chunk> 

<recurrent_chunk>
{chunk}
</recurrent_chunk>

<memory>
{memory}
</memory>


Updated memory:
"""

def safe_int(value, default=5):
    """Safely convert to int; fallback to default if invalid."""
    try:
        v = int(value)
        if v < 1:
            return default
        return v
    except Exception:
        return default
#######################################################################
############################ START TOOLS ##############################
#######################################################################
def tokenize(text: str):
    return text.lower().split()

def bm25_retrieve_for_recurrent_chunk(
    query: str,
    bm25_model: BM25Okapi,
    retrieve_docs,
    retrieve_meta,
    top_k: int,
    exclude_recurrent_idx: int | None = None,
):
    q_tokens = tokenize(query)
    scores = bm25_model.get_scores(q_tokens)
    ranked = np.argsort(-scores)

    top_k = safe_int(top_k, default=5)
    results = []
    for idx in ranked:
        if exclude_recurrent_idx is not None:
            if retrieve_meta[idx]["recurrent_idx"] == exclude_recurrent_idx:
                continue
        results.append(retrieve_docs[idx])

        if len(results) >= top_k:
            break
    return results


def after_think(solution_str):
    if "</think>" not in solution_str:
        return None, solution_str 
    final_answer = solution_str.split("</think>")[-1].strip()
    return final_answer, solution_str


# Function call special tokens (from Qwen-Agent)
FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'




def get_bm25search_tool_schema():
    """
    返回给 Qwen 的 retrievesearch 工具 schema（OpenAI / Qwen-Agent 风格）
    """
    return {
        "type": "function",
        "function": {
            "name": "retrievesearch",
            "description": "Search over pre-indexed context chunks using BM25.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query focusing on key entities / concepts."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (3-5).",
                        "minimum": 3,
                        "maximum": 5,
                        "default": 5
                    },
                },
                "required": ["query"]
            },
        },
    }




def _process_messages_for_function_call(messages, functions=None):
    """
    Process messages to text format suitable for Qwen model input.
    Extracted from _complete_qwen for reuse in generation.py
    """
    # Convert messages to proper format for Qwen processing
    from qwen_agent.llm.schema import Message, ContentItem
    qwen_messages = []
    for message in messages: # [{"role": ..., "content": ...}, ...]
        content = message["content"]
        if isinstance(content, str):
            content = [ContentItem(text=content)]
        elif isinstance(content, list):
            # Handle mixed content types
            content_items = []
            for item in content:
                if isinstance(item, str):
                    content_items.append(ContentItem(text=item))
                elif isinstance(item, dict) and "text" in item: # If the list item is a dictionary and has a "text" key, then extract value
                    content_items.append(ContentItem(text=item["text"]))
                else:
                    content_items.append(ContentItem(text=str(item)))
            content = content_items
        else:
            content = [ContentItem(text=str(content))]

        qwen_message = Message(role=message["role"], content=content)
        qwen_messages.append(qwen_message)

    # Preprocess messages with function calling format
    if functions: # If tool functions are available
        processed_messages = QwenFnCallPrompt.preprocess_fncall_messages(
            messages=qwen_messages,
            functions=functions,
            lang='en',  # Using English
            parallel_function_calls=True,
            function_choice='auto'
        )
    else:
        processed_messages = qwen_messages

    # Convert back to dict format
    dict_messages = []
    for message in processed_messages:
        content_text = ""
        for content_item in message.content:
            content_text += content_item.text
        dict_messages.append({
            "role": message.role,
            "content": content_text
        })
    return dict_messages



def _parse_response(response):
    """
    Parse the response from Qwen model to extract function calls and text content.
    Returns a list of assistant messages with either content or function_call.
    """
    function_calls, remaining_text,stop = _parse_function_calls_from_text(response)

    messages = []

    if function_calls:
        # Add main content message if there's remaining text
        if remaining_text.strip():
            messages.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": remaining_text.strip().strip("<think>").strip("</think>").strip()
            })

        # Add function call messages
        for func_call in function_calls:
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": {
                    "name": func_call["name"],
                    "arguments": func_call["arguments"]
                }
            })

    else:
        # No function calls - parse thinking and final response
        # Format: <think>...</think>\n\nfinal_response
        thinking_pattern = r'<think>(.*?)</think>\s*(.*)'
        match = re.search(thinking_pattern, response, re.DOTALL)

        if match:
            thinking_content = match.group(1).strip()
            final_content = match.group(2).strip()

            messages.append({
                "role": "assistant",
                "content": final_content,
                "reasoning_content": thinking_content
            })

        else:
            # No thinking tags found, treat entire response as content
            # Remove any incomplete thinking tags
            clean_response = response.strip()
            if clean_response.startswith('<think>') and '</think>' not in clean_response:
                # Incomplete thinking tag at start
                clean_response = clean_response[7:].strip()  # Remove '<think>'

            messages.append({
                "role": "assistant",
                "content": clean_response,
                "reasoning_content": "",
            })

    # If no content and no function calls, add an empty content message
    if not messages:
        messages.append({
            "role": "assistant",
            "content": ""
        })

    return messages,stop

MAX_RETRIEVED_LENGTH=12000
def _parse_function_calls_from_text(text: str):
    """
    Parse function calls from raw model output text.
    Returns (function_calls, remaining_text).
    Adapted from raw_token_function_calling.py
    """
    function_calls = []
    if "STOP" in text:
        return function_calls, "", True
    # Find all function call patterns
    pattern = f'{re.escape(FN_NAME)}:\\s*([^\\n]+)\\s*{re.escape(FN_ARGS)}:\\s*([^✿]+?)(?={re.escape(FN_RESULT)}|{re.escape(FN_EXIT)}|{re.escape(FN_NAME)}|$)'

    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        func_name = match.group(1).strip()
        func_args = match.group(2).strip()

        # Clean up arguments
        func_args = _remove_trailing_comment_of_fn_args(func_args)

        function_calls.append({
            'name': func_name,
            'arguments': func_args
        })

    # Extract text before first function call
    first_fn_pos = text.find(FN_NAME)
    if first_fn_pos >= 0:
        remaining_text = text[:first_fn_pos].strip()
    else:
        remaining_text = text.strip()

    # Remove incomplete special tokens
    remaining_text = _remove_incomplete_special_tokens(remaining_text)

    return function_calls, remaining_text,False

def _remove_incomplete_special_tokens(text: str) -> str:
    """Remove incomplete special tokens from the end of text."""
    special_tokens = (FN_NAME, FN_ARGS, FN_RESULT, FN_EXIT)
    text = text.rstrip()

    if text.endswith(special_tokens):
        for s in special_tokens:
            if text.endswith(s):
                text = text[:-len(s)]
                break
    else:
        trail_start = text.rfind('✿')
        if trail_start >= 0:
            trail_token = text[trail_start:]
            for s in special_tokens:
                if s.startswith(trail_token):
                    text = text[:trail_start]
                    break

    text = text.lstrip('\n').rstrip()
    return text


def _remove_trailing_comment_of_fn_args(fn_args: str) -> str:
    """Remove trailing comments from function arguments."""
    fn_args = fn_args.strip()

    if fn_args.startswith('{'):
        k = fn_args.rfind('}')
        if k > 0:
            fn_args = fn_args[:k + 1]

    if fn_args.startswith('```'):
        k = fn_args.rfind('\n```')
        if k > 0:
            fn_args = fn_args[:k + 4]

    return fn_args




def _ids_to_text_batch(tokenizer, messages_ids):
    """
    messages_ids:  (1) Tensor[B, T]  或 (2) list[tensor[T]] 或 (3) list[list[int]]
    return: list[str] length=B
    """
    if torch.is_tensor(messages_ids):
        # Tensor[B, T]
        return tokenizer.batch_decode(messages_ids, skip_special_tokens=True)

    # list[...] -> normalize to list[list[int]]
    norm = []
    for x in messages_ids:
        if torch.is_tensor(x):
            norm.append(x.tolist())
        else:
            norm.append(x)
    return tokenizer.batch_decode(norm, skip_special_tokens=True)



def _conversations_to_batch_ids(tokenizer, conversations, device=None, add_generation_prompt=True):
    """
    conversations: list[list[{"role":..., "content":...}, ...]]
    return: input_ids (B, T), attention_mask (B, T)
    """
    seqs = []
    for conv in conversations:
        if hasattr(tokenizer, "apply_chat_template"):
            ids = tokenizer.apply_chat_template(
                conv,
                tokenize=True,
                add_generation_prompt=False,  
                return_tensors="pt",
            )[0]

        else:
            # fallback（最原始、可控）
            text = "\n".join([m["content"] for m in conv])
            ids = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,   
            ).input_ids[0]

        seqs.append(ids[:-1])

    return seqs
    # batch = tokenizer.pad({"input_ids": seqs}, padding=True, return_tensors="pt")
    # input_ids = batch["input_ids"]
    # attention_mask = batch["attention_mask"]
    # if device is not None:
    #     input_ids = input_ids.to(device)
    #     attention_mask = attention_mask.to(device)
    return seqs

def strip_last_token_if_match(seqs, tok_id: int):
    """
    seqs: List[torch.Tensor] or List[List[int]]
    tok_id: token id to remove if it is the last token
    """
    out = []
    for s in seqs:
        if torch.is_tensor(s):
            if s.numel() > 0 and int(s[-1].item()) == tok_id:
                s = s[:-1]
        else:
            if len(s) > 0 and s[-1] == tok_id:
                s = s[:-1]
        out.append(s)
    return out
#######################################################################
############################# END TOOLS ###############################
#######################################################################


@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens3
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + 2048 + self.chunk_size + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length=recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)
        rm = row_dict.get("reward_model", {})
        gts = rm.get("ground_truth", [])

        # normalize to list[str]
        if isinstance(gts, np.ndarray):
            gts = gts.tolist()
        elif isinstance(gts, str):
            gts = [gts]
        elif not isinstance(gts, list):
            gts = list(gts)  # fallback

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        row_dict["answers"] = [
            self.tokenizer.encode(gt, add_special_tokens=False)
            for gt in gts
        ]
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt_ids","answers"]


#### template used for rewrite the retrieve query
TEMPLATE_RETRIEVER = """You are presented with the final question and a the previous memory. Please according to the given memory, output the most important element you need to retrieve. 
Your output will be used as the query for BM25, make sure you output the straightforward element directly. You are allowed to list multiple key elements.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>
Elements need to be retrieved:
"""


TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

class InfMem(RAgent):
    STAGE_REWRITE = "rewrite"        # 重写 query
    STAGE_MEMORY  = "memory_update"  # 正常 memory update
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # A trick to get a simple chat_template for any tokenizer
        # the output text looks like:
        # '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n'
        # This is a format string itself, '{message}' will be replaced by the actual message.
        self.chat_template = chat_template(tokenizer)
        self.token_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_RETRIEVE_RECURRENT), tokenizer)
        self.token_only_retrieve_chunk_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_RETRIEVE), tokenizer)
        self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
        
        # ADD: new template to rewrite the query
        self.token_retriever_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_RETRIEVER), tokenizer)
        self.token_call_or_answer_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_CALL_OR_ANSWER), tokenizer)
        
        
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template.length 
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {self.token_message_template.length}(message_template) = {self.max_input_length}\n')
        self.NO_MEMORY_TOKENS = tokenizer.encode("No previous memory", add_special_tokens=False)
        self.end_think  = tokenizer.encode("</think>", add_special_tokens=False)# [522, 26865]   # [522, 1708, 29]
        self.retrieve_separator = self.tokenizer.encode("\n\n", add_special_tokens=False)
        
        self.stage = self.STAGE_REWRITE
        self.function_call = None   # 每个 sample 的 rewrited prompt（token ids）



   ####################################################################################################
   ###############################                           ##########################################
   ###############################           tool            ##########################################
   ###############################                           ##########################################
   ####################################################################################################


   
    def extract_memory_from_generation(self, seq):
        """
        Extract first <summary>...</summary> block per sequence.
        seq: tensor of token ids
        Returns: tensor slice (summary token ids) or None
        """
        if seq is None or len(seq) == 0:
            return None
        
        # Locate the summary tokens using tensor operations
        # start_idx = self._find_first_subsequence_tensor(seq, self.start_tokens)
        end_think_idx = self._find_first_subsequence_tensor(seq, self.end_think)
        if end_think_idx is not None:
            start_content_idx = end_think_idx + len(self.end_think)+1
            return seq[start_content_idx:]
        else:
            # Start token not found → nothing to extract
            return None

    def _normalize_answers(self, ans):
        if isinstance(ans[0], list):  # multi-answer
            return ans
        return [ans]      # wrap single case

    def _find_first_subsequence_tensor(self, seq, target): 
        """ 
        Returns the start index of the first occurrence of target in seq (both tensors), 
        or None if not found. 
        """ 
        if seq is None or len(seq) == 0: 
            return None 
            
        # Convert target to tensor if it's a list
        if isinstance(target, list):
            target = torch.tensor(target, device=seq.device, dtype=seq.dtype)
        
        target_len = len(target) 
        seq_len = len(seq) 
        
        if target_len > seq_len:
            return None
        
        # Efficient tensor-based search
        for i in range(seq_len - target_len + 1): 
            if torch.equal(seq[i:i+target_len], target): 
                return i 
        return None


   ####################################################################################################
   ###############################                           ##########################################
   ###############################           tool            ##########################################
   ###############################                           ##########################################
   ####################################################################################################



    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        
        self.ctx_length = gen_batch.batch['context_length'] # if all context is used, then the sample will no more be active
        self.bsz = len(self.ctx_length)
        self.memory = np.empty(self.bsz, dtype=object)
        self.is_final = False
        self.answers_ids = gen_batch.non_tensor_batch['answers']


        self.gamma = 0.6
        self.ans_in_mem_rewards = torch.zeros(self.bsz, dtype=torch.float16)
        
        self.all_function_call_success = torch.ones(self.bsz, dtype=torch.float16)

        self.all_memory_finished = torch.ones(self.bsz, dtype=torch.float16)

        self.early_stop_reward = torch.zeros(self.bsz, dtype=torch.float16)

        self.first_time_answer_in_mem = torch.zeros(self.bsz, dtype=torch.int8)



        self.active_mask = torch.ones(self.bsz, dtype=bool)
        self.history = [[] for _ in range(self.bsz)]


        self.function_call = np.empty(self.bsz, dtype=object)

        self.stage = self.STAGE_REWRITE


        bm25_tool = get_bm25search_tool_schema()
        self.functions = [bm25_tool["function"]] 


        self.retrieve_chunk_size = getattr(self.config, "retrieve_chunk_size", 500)
        self.retrieve_top_k = getattr(self.config, "retrieve_top_k", 5)
        self.pad_token_id = self.tokenizer.pad_token_id

        self.bm25_list = [None for _ in range(self.bsz)]
        self.retrieve_docs_list = [None for _ in range(self.bsz)]
        self.retrieve_meta_list = [None for _ in range(self.bsz)]   

        context_ids_all = self.gen_batch.batch['context_ids']  # shape: [bsz, max_ctx_len]

        for b in range(self.bsz):
            ctx_len = int(self.ctx_length[b])
            if ctx_len <= 0:
                self.bm25_list[b] = None
                self.retrieve_doc_ids_list[b] = []
                continue

            ctx_ids = context_ids_all[b, :ctx_len].tolist()

            retrieve_docs = []   
            retrieve_meta = []   
            pos = 0
            while pos < len(ctx_ids):
                rs = pos
                re = min(len(ctx_ids), pos + self.retrieve_chunk_size)
                r_idx = re // self.config.chunk_size
                sub_ids = ctx_ids[rs:re]
                text = self.tokenizer.decode(sub_ids)
                retrieve_docs.append(text)
                retrieve_meta.append({
                    "recurrent_idx": r_idx,
                    "start": rs,
                    "end": re,
                })
                pos = re

            self.retrieve_docs_list[b] = retrieve_docs
            self.retrieve_meta_list[b] = retrieve_meta
            if len(retrieve_docs) == 0:
                self.bm25_list[b] = None
            else:
                self.bm25_list[b] = BM25Okapi(retrieve_docs)




    def bm25search_impl(self, index, query: str, top_k: int = 8, exclude_recurrent_idx=None):
        retrieve_docs = self.retrieve_docs_list[index]
        retrieve_meta = self.retrieve_meta_list[index]
        bm25_model = self.bm25_list[index]
        chunks = bm25_retrieve_for_recurrent_chunk(
            query=query,
            bm25_model=bm25_model,
            retrieve_docs=retrieve_docs,
            retrieve_meta=retrieve_meta,
            top_k=top_k,
            exclude_recurrent_idx=exclude_recurrent_idx,
        )
        return {
            "top_k": top_k,
            "query": query,
            "results": [
                {
                    "rank": i + 1,
                    "text": txt,
                }
                for i, txt in enumerate(chunks)
            ]
        }




   ####################################################################################################
   ###########################                                  #######################################
   ###########################  action for retrieve+recurrent   #######################################
   ###########################                                  #######################################
   ####################################################################################################




    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        # suppose 0 is pad_token_id
        # max_chunks = 3, chunk_sieze = 2
        # pi is token in prompt, ti is token in chat template, 
        # [1,2] [3,4] [5,0] | p0 string
        # [1,2] [3,0] [0,0] | p1,p1 string
        # [1,0] [0,0] [0,0] | p2,p2,p2 string
        # -------- round 1 ---------
        # [1,2]            [t0,p0,t1, m,t2, 1, 2,t3]                           [ 0, 0, 0,t0,p0,t1, m,t2, 1, 2,t3]
        # [1,2]  -format-> [t0,p1,p1,t1, m,t2, 1, 2,t3] -pad2Dlist2Tendors->   [ 0, 0,t0,p1,p1,t1, m,t2, 1, 2,t3]
        # [1,0]            [t0,p2,p2,p3,t1, m,t2, 1,t3]                        [ 0, 0,t0,p2,p2,p3,t1, m,t2, 1,t3]
        # get mask & positionids
        pad_id = self.tokenizer.pad_token_id
        if getattr(self, "_lock_step", False):
            # restore active_mask from previous turn
            self._lock_step = False
        else:
            active_mask = self.ctx_length > self.step * self.config.chunk_size
            self.active_mask = self.active_mask & active_mask
        # active_mask = self.ctx_length > self.step * self.config.chunk_size
        
        active_mask = self.active_mask
        gen_batch = self.gen_batch
        # if all context is used, and its not done, then it will be the final turn for this batch
        if active_mask.sum().item() == 0:
            self.is_final = True
            self.messages = [
                self.token_final_message_template.format(
                    prompt=prompt,
                    memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                )
                for prompt, memory in zip(gen_batch.non_tensor_batch['prompt_ids'], self.memory)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'FINAL TURN: InfMem.next() done')
        else:
            
            # print("active_mask prompt_i", active_mask)
            # ========== not the final round, need to rewrite then update memory ===========
            # 1. no need to pad prompt
            # 2. context padded for 2D indexing, elegant engineering
            # 3. no need to pad memory
            prompt_i = gen_batch.non_tensor_batch['prompt_ids'][active_mask]
            chunk_i = gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)] # bs * chunk_size
            memory_i = self.memory[active_mask]
            active_idx = active_mask.nonzero(as_tuple=True)[0].tolist()
            retrieval_history_i = [self.history[i] for i in active_idx]
            pad_id = self.tokenizer.pad_token_id  # or whatever your pad id is

            
            def build_history_text(retrieve_step: int, max_retrieve_steps: int, history: list[dict]) -> str:
                lines = [
                    f"You have taken {retrieve_step} retrieval steps,  Maximum allowed retrieval steps is {max_retrieve_steps}",
                ]
                for i, h in enumerate(history):
                    lines.append(f"Step {i+1}: query={h['query']!r}, top_k={h['top_k']}")
                    # lines.append(f"Step {i+1}: method={h['method']!r}, query={h['query']!r}, top_k={h['top_k']}")
                return "\n".join(lines)

            

            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask]
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool)

            # different messages for different stages
            if self.stage == self.STAGE_REWRITE:
                
                max_steps_i = (self.ctx_length[active_mask] // self.config.chunk_size)
                if torch.is_tensor(max_steps_i):
                    max_steps_i = max_steps_i.tolist()  # -> List[int]
                else:
                    max_steps_i = list(max_steps_i)

                history_texts = [
                    build_history_text(self.step, ms, rh)
                    for ms, rh in zip(max_steps_i, retrieval_history_i)
                ]

                # ---- 2) batch encode---
                hist_ids_batch = self.tokenizer(
                    history_texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,             
                    return_attention_mask=False,
                )["input_ids"]                  

                self.messages = [
                    self.token_call_or_answer_template.format(
                        prompt=prompt,                
                        hist=hist_ids,                 
                        memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                    )
                    for prompt, memory, hist_ids in zip(prompt_i, memory_i, hist_ids_batch)
                ]

                msg_texts = _ids_to_text_batch(self.tokenizer, self.messages)

                conversations = []
                for txt in msg_texts:
                    conv = [{"role": "user", "content": txt}]
                    conv_fc = _process_messages_for_function_call(conv, self.functions)
                    conversations.append(conv_fc)

                input_ids = _conversations_to_batch_ids(
                    self.tokenizer,
                    conversations,
                    device=self.device if hasattr(self, "device") else None,
                    add_generation_prompt=False,
                )
                im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                input_ids = strip_last_token_if_match(input_ids, im_end_id)
                self.messages = input_ids
                mode = self.STAGE_REWRITE

            else:
                #### -------- STAGE B: BM25 + update the memory --------
                
                lengths = (chunk_i != pad_id).sum(dim=1)  # [bs]
                avg_len = lengths.float().mean()
                lengths_mem = [
                    sum(1 for x in row if x != pad_id) if row is not None else 0
                    for row in memory_i
                ]

                TOP_K = 5

                avg_len_mem = sum(lengths_mem) / len(lengths_mem)

                print("Average memory length:", avg_len_mem)
                
                print("Average chunk length:", avg_len.item())
                
                # print("active_mask STAGE B: BM25 + update the memory", active_mask)
                retrieved_chunks = []   # per-sample: list[Tensor] for retrieve chunk
                recurrent_chunks = []   # per-sample: Tensor for the current 5k recurrent chunk

                for row, sample_idx in enumerate(sample_index):
                    tool_calls = []
                    retrieved_block = "No retrieve info"
                    # ----------- 1) query ids（优先 rewrite） -----------
                    if self.function_call[sample_idx] is not None:
                        parsed_messages = self.function_call[sample_idx] 
                        # 2) 遍历本轮 assistant 消息，看有没有 function_call
                        for message in parsed_messages:
                            if "function_call" in message:
                                tool_calls.append(message)
                        for message in tool_calls:
                            fn_name = message["function_call"]["name"]
                            raw_args = message["function_call"]["arguments"]
                            if fn_name == "retrievesearch":
                                try:
                                    args = json.loads(raw_args)
                                    query = args["query"]
                                    top_k = args["top_k"]
                                    
                                except Exception:
                                    print("function call invalid, use default value")
                                    self.all_function_call_success[sample_idx] = 0
                                    args = {}
                                    query = self.tokenizer.decode(prompt_i[row], skip_special_tokens=True)
                                    top_k = TOP_K
                                    
                                self.history[sample_idx].append({ "query": query, "top_k": top_k})
                                
                                ret = self.bm25search_impl(index=sample_idx, query=query, top_k=top_k)

                                retrieved_block = "\n\n".join(
                                    f"[Retrieved #{r['rank']}] {r['text']}"
                                    for r in ret["results"]
                                )
                                if len(retrieved_block)>MAX_RETRIEVED_LENGTH:
                                    retrieved_block = retrieved_block[:MAX_RETRIEVED_LENGTH]
                            else:
                                self.history[sample_idx].append({ "query": "", "top_k": ""})
                                
                                
                    retrieved_ids = self.tokenizer.encode(retrieved_block)

                    retrieved_chunks.append(retrieved_ids)
                    
                    cur_chunk = chunk_i[row]
                    cur_chunk = cur_chunk[cur_chunk != pad_id]
                    recurrent_chunks.append(cur_chunk)

                lengths_retireve_mem = [
                    sum(1 for x in row if x != pad_id) if row is not None else 0
                    for row in retrieved_chunks
                ]
                avg_len_retireve_mem = sum(lengths_retireve_mem) / len(lengths_mem)

                print("Average avg_len_retireve_mem length:", avg_len_retireve_mem)
                

                def build_history_text_2(retrieve_step: int, max_retrieve_steps: int, history: list[dict]) -> str:
                    try:
                        h = history[-1]
                    except:
                        h = {"query": "", "top_k": ""}
                    history = (
                            f"Current retrieval steps: {retrieve_step+1}, Maximum allowed retrieval steps: {max_retrieve_steps}\n"
                            + f"Step {len(history)}: query={h['query']!r}, top_k={h['top_k']}"
                            # + f"Step {len(history)}: method={h['method']!r}, query={h['query']!r}, top_k={h['top_k']}"
                        )
                    return history
                
                max_steps_i = (self.ctx_length[active_mask] // self.config.chunk_size)
                if torch.is_tensor(max_steps_i):
                    max_steps_i = max_steps_i.tolist()  # -> List[int]
                else:
                    max_steps_i = list(max_steps_i)

                history_texts = [
                    build_history_text_2(self.step, ms, rh)
                    for ms, rh in zip(max_steps_i, retrieval_history_i)
                ]

                
                
                hist_ids_batch = self.tokenizer(
                    history_texts,
                    add_special_tokens=False,
                    padding=False,  
                    truncation=False, 
                    return_attention_mask=False,
                )["input_ids"] 

                
                self.messages = []
                for prompt_ids, memory_ids, retrieve_ids_list, chunk_ids, hist_ids in zip(
                        prompt_i, memory_i, retrieved_chunks, recurrent_chunks, hist_ids_batch):




                    self.messages.append(
                        self.token_message_template.format(
                            prompt=prompt_ids,
                            hist=hist_ids,
                            memory=memory_ids if memory_ids is not None else self.NO_MEMORY_TOKENS,
                            retrieve=retrieve_ids_list,     # pass as-is
                            chunk=chunk_ids,                 # recurrent chunk ids
                        )
                    )

                mode = self.STAGE_MEMORY
            self.meta_info = {'input_pad_to': self.max_input_length,
                    'pad_to': self.config.gen_pad_to,
                    'generation_kwargs': {
                    'max_tokens': self.config.gen_max_tokens_memorization,
                    'n': 1 # note that we have already repeat n times in ray_trainer
                },
                'mode': mode,  
            }
            logger.info(f'InfMem.action() done, stage={self.stage}')


        fixed_messages = []
        for msg in self.messages:
            if not torch.is_tensor(msg):
                msg = torch.tensor(msg, dtype=torch.long, device=chunk_i.device)


            if msg.size(0) > self.max_input_length:
                msg = msg[-self.max_input_length:]


            fixed_messages.append(msg)

        self.messages = fixed_messages
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info



    def update(self, gen_output: DataProto, not_that_correct = 0.5) -> DataProto:
        import re

        def normalize_text(s: str) -> str:
            s = s.strip()
            s = re.sub(r"\s+", " ", s)
            return s

        def contains_answer_text(mem_ids, global_i, tokenizer) -> bool:
            
            ans = self.answers_ids[global_i]
            if isinstance(ans[0], int):
                ans = [ans]
            if isinstance(mem_ids, torch.Tensor):
                mem_ids = mem_ids.tolist()
            mem_text = tokenizer.decode(mem_ids, skip_special_tokens=True)
            mem_norm = normalize_text(mem_text).lower()
            for a in ans:
                answer = tokenizer.decode(a, skip_special_tokens=True)
                gt_norm = normalize_text(answer).lower()
                if bool(gt_norm and gt_norm in mem_norm):
                    return True
            return False


        stage = self.stage

        
        if not self.is_final:
            if stage == self.STAGE_REWRITE:
                # ===== 阶段 A：把模型输出解释为 rewrited prompt =====
                unpadded = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
                # rewritten_ids_batch = [self.extract_memory_from_generation(x) for x in unpadded]
                
                rewritten_ids_batch = unpadded

                act_idx = np.where(self.active_mask)[0]
                device = gen_output.batch['responses'].device

                for local_i, global_i in enumerate(act_idx):
                    rew = rewritten_ids_batch[local_i]
                    function_call_text = self.tokenizer.decode(rew, skip_special_tokens=True)
                    parsed_messages,stop = _parse_response(function_call_text)
                    if global_i == 0 and stop:
                        logger.info(f"[I decide to stop] {function_call_text}")
                    self.function_call[global_i] = parsed_messages
                    tool_calls = []
                    answer_message = None  
                    
                    for message in parsed_messages:
                        if "function_call" in message:
                            tool_calls.append(message)
                        elif message.get("content"):  # 这就是 ANSWER 模式下的最终回答
                            answer_message = message



                    if stop:
                        self.active_mask[global_i] = 0

                        # if answer were 
                        if self.ans_in_mem_rewards[global_i] < 1:
                            self.early_stop_reward[global_i] = 0
                        else:
                            d = self.step - self.first_time_answer_in_mem[global_i]
                            self.early_stop_reward[global_i] = min (1,(self.gamma ** (d-1)))


                self.stage = self.STAGE_MEMORY
                self.log_step(gen_output)
                
                self.step = self.step   # lock step
                self._lock_step = True
                return gen_output
            else:
                
                unpadded = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
                mem_ids_batch = [self.extract_memory_from_generation(x) for x in unpadded]
                # mem_ids_batch = unpadded
                act_idx = np.where(self.active_mask)[0]

                assert len(act_idx) == len(mem_ids_batch), \
                    f"mask mismatch: {len(act_idx)=}, {len(mem_ids_batch)=}"

                for local_i, global_i in enumerate(act_idx):
                    if mem_ids_batch[local_i] is None:
                        self.all_memory_finished[global_i]=0
                    else:
                        self.memory[global_i] = mem_ids_batch[local_i]
                device = gen_output.batch['responses'].device

                for local_i, global_i in enumerate(act_idx):
                    mem_ids = mem_ids_batch[local_i]
                    if not torch.is_tensor(mem_ids):
                        if mem_ids is None or len(mem_ids) == 0:
                            mem_ids = torch.empty(0, dtype=torch.long, device=device)
                        else:
                            mem_ids = torch.tensor(mem_ids, device=device, dtype=torch.long)


                    found = contains_answer_text(mem_ids, global_i, self.tokenizer)


                    if found:
                        if self.ans_in_mem_rewards[global_i]==0 and self.first_time_answer_in_mem[global_i]==0:
                            self.first_time_answer_in_mem[global_i]=self.step
                        self.ans_in_mem_rewards[global_i] = 1.0
                self.log_step(gen_output)
                self.step += 1
                
                self.stage = self.STAGE_REWRITE

        return gen_output

    
    
    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        
        for global_i in range(len(self.early_stop_reward)):
            # for those never stopped
            if (
                self.first_time_answer_in_mem[global_i] != 0
                and self.early_stop_reward[global_i] == 0
            ):
                d = self.step - self.first_time_answer_in_mem[global_i]
                
                self.early_stop_reward[global_i] = min(1.0, self.gamma ** (d - 1))
                
        final_rewards = 0.6 * self.early_stop_reward + 0.5 * self.all_function_call_success + 0.5 * self.all_memory_finished
        # final_rewards =  0.6 * self.ans_in_mem_rewards + 0.9 * self.early_stop_reward + 0.5 * self.all_function_call_success
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.ans_in_mem_rewards
        del self.early_stop_reward
        del self.all_memory_finished
        del self.all_function_call_success
        del self.first_time_answer_in_mem
        del self.final_mask_list
        del self.sample_index_list
        del self.step
        return final_mask, sample_index, final_rewards
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=2000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.step if not self.is_final else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT] STEP{step}{'='*30}")

        # Message and Response section
        if self.active_mask[0]:
            decoded_message = self.tokenizer.decode(self.messages[0])
            rsp0 = gen_output.batch['responses'][0]
            correct_answer=self.tokenizer.decode(self.answers_ids[0][0])
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
            logger.info(f"[CORRECT ANSWER] {len(self.answers_ids[0])} answers in total, first is: [{correct_answer}]")
            logger.info(f"{' '*10}{'-'*20}ground truth answer end{'-'*20}{' '*10}")
            
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")
            logger.info(f"[CURRENT REWARD] ans_in_mem_rewards:{self.ans_in_mem_rewards.mean()}, early_stop_reward: {self.early_stop_reward.mean()}, all_function_call_success: {self.all_function_call_success.mean()}, all_memory_finished: {self.all_memory_finished.mean()}")
            
            


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=InfMem)











   ###################################################################################################
   #############################                             #########################################
   #############################  action for only retrieve   #########################################
   #############################                             #########################################
   ###################################################################################################

    # @override
    # def action(self) -> Tuple[List[torch.Tensor], dict]:
    #     # suppose 0 is pad_token_id
    #     # max_chunks = 3, chunk_sieze = 2
    #     # pi is token in prompt, ti is token in chat template, 
    #     # [1,2] [3,4] [5,0] | p0 string
    #     # [1,2] [3,0] [0,0] | p1,p1 string
    #     # [1,0] [0,0] [0,0] | p2,p2,p2 string
    #     # -------- round 1 ---------
    #     # [1,2]            [t0,p0,t1, m,t2, 1, 2,t3]                           [ 0, 0, 0,t0,p0,t1, m,t2, 1, 2,t3]
    #     # [1,2]  -format-> [t0,p1,p1,t1, m,t2, 1, 2,t3] -pad2Dlist2Tendors->   [ 0, 0,t0,p1,p1,t1, m,t2, 1, 2,t3]
    #     # [1,0]            [t0,p2,p2,p3,t1, m,t2, 1,t3]                        [ 0, 0,t0,p2,p2,p3,t1, m,t2, 1,t3]
    #     # get mask & positionids
    #     pad_id = self.tokenizer.pad_token_id
    #     if getattr(self, "_lock_step", False):
    #         # restore active_mask from previous turn
    #         self._lock_step = False
    #     else:
    #         active_mask = 6 > self.step
    #         self.active_mask = self.active_mask & active_mask
    #     # active_mask = self.ctx_length > self.step * self.config.chunk_size
        
    #     active_mask = self.active_mask
    #     gen_batch = self.gen_batch
    #     # if all context is used, and its not done, then it will be the final turn for this batch
    #     if active_mask.sum().item() == 0:
    #         self.is_final = True
    #         self.messages = [
    #             self.token_final_message_template.format(
    #                 prompt=prompt,
    #                 memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
    #             )
    #             for prompt, memory in zip(gen_batch.non_tensor_batch['prompt_ids'], self.memory)
    #         ]
    #         sample_index = torch.arange(self.bsz, dtype=torch.int)
    #         final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all False
    #         self.meta_info = {'input_pad_to': self.max_input_length,
    #                      'pad_to': self.config.gen_pad_to,
    #                      'generation_kwargs': {
    #                       'max_tokens': self.config.gen_max_tokens_memorization,
    #                       'n': 1 # note that we have already repeat n times in ray_trainer
    #                     }}
    #         logger.info(f'FINAL TURN: InfMem.next() done')
    #     else:
            
    #         # print("active_mask prompt_i", active_mask)
    #         # ========== not the final round, need to rewrite then update memory ===========
    #         # 1. no need to pad prompt
    #         # 2. context padded for 2D indexing, elegant engineering
    #         # 3. no need to pad memory
    #         prompt_i = gen_batch.non_tensor_batch['prompt_ids'][active_mask]
    #         chunk_i = gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)] # bs * chunk_size
    #         memory_i = self.memory[active_mask]
    #         active_idx = active_mask.nonzero(as_tuple=True)[0].tolist()
    #         retrieval_history_i = [self.history[i] for i in active_idx]
    #         pad_id = self.tokenizer.pad_token_id  # or whatever your pad id is

            
    #         def build_history_text(retrieve_step: int, max_retrieve_steps: int, history: list[dict]) -> str:
    #             lines = [
    #                 f"You have taken {retrieve_step} retrieval steps,  Maximum allowed retrieval steps is {max_retrieve_steps}",
    #             ]
    #             for i, h in enumerate(history):
    #                 lines.append(f"Step {i+1}: query={h['query']!r}, top_k={h['top_k']}")
    #                 # lines.append(f"Step {i+1}: method={h['method']!r}, query={h['query']!r}, top_k={h['top_k']}")
    #             return "\n".join(lines)

            

    #         sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask]
    #         final_mask = torch.full(sample_index.shape, False, dtype=torch.bool)

    #         # different messages for different stages
    #         if self.stage == self.STAGE_REWRITE:
                
    #             # ---- 1) 先把每个 sample 的 history 变成 text（仍然是 python list 操作）----
    #             # 注意：max_retrieve_steps 最好按 sample 算，而不是用 [0].item()
    #             # 这里假设 ctx_length_active 是 active 后的 ctx_length（torch tensor 或 numpy 都行）
    #             max_steps_i = (self.ctx_length[active_mask] // self.config.chunk_size)
    #             if torch.is_tensor(max_steps_i):
    #                 max_steps_i = max_steps_i.tolist()  # -> List[int]
    #             else:
    #                 max_steps_i = list(max_steps_i)

    #             history_texts = [
    #                 build_history_text(self.step, ms, rh)
    #                 for ms, rh in zip(max_steps_i, retrieval_history_i)
    #             ]

    #             # ---- 2) batch encode，一次性得到每个 sample 的 hist_ids（List[List[int]]）----
    #             hist_ids_batch = self.tokenizer(
    #                 history_texts,
    #                 add_special_tokens=False,
    #                 padding=False,                 # 不要 padding
    #                 truncation=False,              # 需要截断再开
    #                 return_attention_mask=False,
    #             )["input_ids"]                     # -> List[List[int]]

    #             # ---- 3) 再 loop 插入到 template 里（你原来的 ids-only 流程不变）----
    #             self.messages = [
    #                 self.token_call_or_answer_template.format(
    #                     prompt=prompt,                 # ids
    #                     hist=hist_ids,                 # ids
    #                     memory=memory if memory is not None else self.NO_MEMORY_TOKENS,  # ids
    #                 )
    #                 for prompt, memory, hist_ids in zip(prompt_i, memory_i, hist_ids_batch)
    #             ]

    #             # 2) ids -> text
    #             msg_texts = _ids_to_text_batch(self.tokenizer, self.messages)

    #             # 3) text -> conversation，并做 function-call 预处理
    #             conversations = []
    #             for txt in msg_texts:
    #                 conv = [{"role": "user", "content": txt}]
    #                 conv_fc = _process_messages_for_function_call(conv, self.functions)
    #                 conversations.append(conv_fc)

    #             # 4) conversation -> batch ids（回到你要的 ids 维度）
    #             input_ids = _conversations_to_batch_ids(
    #                 self.tokenizer,
    #                 conversations,
    #                 device=self.device if hasattr(self, "device") else None,
    #                 add_generation_prompt=False,
    #             )
    #             im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
    #             input_ids = strip_last_token_if_match(input_ids, im_end_id)
    #             # 你后面继续走 ids 流程即可
    #             self.messages = input_ids
    #             mode = self.STAGE_REWRITE

    #         else:
    #             #### -------- STAGE B: BM25 + update the memory --------
                
    #             # sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask]
    #             # print("sample_index STAGE B:", sample_index)
    #             # chunk_i: [bs, chunk_size]
    #             lengths = (chunk_i != pad_id).sum(dim=1)  # [bs]
    #             avg_len = lengths.float().mean()
    #             lengths_mem = [
    #                 sum(1 for x in row if x != pad_id) if row is not None else 0
    #                 for row in memory_i
    #             ]

    #             TOP_K = 5

    #             avg_len_mem = sum(lengths_mem) / len(lengths_mem)

    #             print("Average memory length:", avg_len_mem)
                
    #             print("Average chunk length:", avg_len.item())
                
    #             # print("active_mask STAGE B: BM25 + update the memory", active_mask)
    #             retrieved_chunks = []   # per-sample: list[Tensor] for retrieve chunk
    #             recurrent_chunks = []   # per-sample: Tensor for the current 5k recurrent chunk

    #             for row, sample_idx in enumerate(sample_index):
    #                 tool_calls = []
    #                 retrieved_block = "No retrieve info"
    #                 # ----------- 1) query ids（优先 rewrite） -----------
    #                 if self.function_call[sample_idx] is not None:
    #                     parsed_messages = self.function_call[sample_idx] 
    #                     # 2) 遍历本轮 assistant 消息，看有没有 function_call
    #                     for message in parsed_messages:
    #                         if "function_call" in message:
    #                             tool_calls.append(message)
    #                     for message in tool_calls:
    #                         fn_name = message["function_call"]["name"]
    #                         raw_args = message["function_call"]["arguments"]
    #                         if fn_name == "retrievesearch":
    #                             try:
    #                                 args = json.loads(raw_args)
    #                                 query = args["query"]
    #                                 top_k = args["top_k"]
    #                                 # method = args["method"]   # 比如默认让它用 bge
    #                             except Exception:
    #                                 print("function call invalid, use default value")
    #                                 self.all_function_call_success[sample_idx] = 0
    #                                 args = {}
    #                                 query = self.tokenizer.decode(prompt_i[row], skip_special_tokens=True)
    #                                 top_k = TOP_K
    #                                 # method = "bm25"
    #                             # self.history[sample_idx].append({"method": method, "query": query, "top_k": top_k})
    #                             self.history[sample_idx].append({ "query": query, "top_k": top_k})
    #                             # if method == "bm25":
    #                             ret = self.bm25search_impl(index=sample_idx, query=query, top_k=top_k)
    #                             # else:
    #                             #     ret = bgesearch_impl(query=query, top_k=top_k)
    #                             # 把检索结果整成一个 block，给下一轮 TEMPLATE 用
    #                             retrieved_block = "\n\n".join(
    #                                 f"[Retrieved #{r['rank']}] {r['text']}"
    #                                 for r in ret["results"]
    #                             )
    #                         else:
    #                             self.history[sample_idx].append({ "query": "", "top_k": ""})
    #                             # self.history[sample_idx].append({"method": "", "query": "", "top_k": ""})
    #                         #### UPDATE MEMORY #####

                      
                        
    #                     # query_ids = prompt_i[row].tolist()
                    
    #                 # ----------- 2) BM25 检索（得到多个 1k 小块） -----------
    #                 retrieved_ids = self.tokenizer.encode(retrieved_block)


    #                 # print
    #                 # top_doc_ids_list 是 [List[int], List[int], ...]
    #                 # 我们保持为 list[list[int]]，等下格式化时交给 template
    #                 retrieved_chunks.append(retrieved_ids)

    #                 # ----------- 3) 本轮 recurrent chunk（直接去 pad） -----------
    #                 cur_chunk = chunk_i[row]
    #                 cur_chunk = cur_chunk[cur_chunk != pad_id]
    #                 recurrent_chunks.append(cur_chunk)

    #             lengths_retireve_mem = [
    #                 sum(1 for x in row if x != pad_id) if row is not None else 0
    #                 for row in retrieved_chunks
    #             ]
    #             avg_len_retireve_mem = sum(lengths_retireve_mem) / len(lengths_mem)

    #             print("Average avg_len_retireve_mem length:", avg_len_retireve_mem)
    #             # ----------- 4) 构建 messages，不 fuse，只分字段传入 -----------
                
    #             # ---- 1) 先把每个 sample 的 history 变成 text（仍然是 python list 操作）----

    #             def build_history_text_2(retrieve_step: int, max_retrieve_steps: int, history: list[dict]) -> str:
    #                 try:
    #                     h = history[-1]
    #                 except:
    #                     h = {"query": "", "top_k": ""}
    #                 history = (
    #                         f"Current retrieval steps: {retrieve_step+1}, Maximum allowed retrieval steps: {max_retrieve_steps}\n"
    #                         + f"Step {len(history)}: query={h['query']!r}, top_k={h['top_k']}"
    #                         # + f"Step {len(history)}: method={h['method']!r}, query={h['query']!r}, top_k={h['top_k']}"
    #                     )
    #                 return history
    #             # 注意：max_retrieve_steps 最好按 sample 算，而不是用 [0].item()
    #             # 这里假设 ctx_length_active 是 active 后的 ctx_length（torch tensor 或 numpy 都行）
    #             max_steps_i = (self.ctx_length[active_mask] // self.config.chunk_size)
    #             if torch.is_tensor(max_steps_i):
    #                 max_steps_i = max_steps_i.tolist()  # -> List[int]
    #             else:
    #                 max_steps_i = list(max_steps_i)

    #             history_texts = [
    #                 build_history_text_2(self.step, ms, rh)
    #                 for ms, rh in zip(max_steps_i, retrieval_history_i)
    #             ]

    #             # ---- 2) batch encode，一次性得到每个 sample 的 hist_ids（List[List[int]]）----
    #             hist_ids_batch = self.tokenizer(
    #                 history_texts,
    #                 add_special_tokens=False,
    #                 padding=False,                 # 不要 padding
    #                 truncation=False,              # 需要截断再开
    #                 return_attention_mask=False,
    #             )["input_ids"]                     # -> List[List[int]]

                
    #             self.messages = []
    #             for prompt_ids, memory_ids, retrieve_ids_list, hist_ids in zip(
    #                     prompt_i, memory_i, retrieved_chunks, hist_ids_batch):

    #                 # convert retrieve_ids_list = [list[int], list[int]...]
    #                 # into a single tensor or list for template
    #                 # 这里我们保持 list[list[int]]，交给 token_message_template 做格式化
    #                 # 若 template 需要 tensor，你可以 flatten 或者手动处理

    #                 self.messages.append(
    #                     self.token_only_retrieve_chunk_template.format(
    #                         prompt=prompt_ids,
    #                         hist=hist_ids,
    #                         memory=memory_ids if memory_ids is not None else self.NO_MEMORY_TOKENS,
    #                         retrieve=retrieve_ids_list,     # pass as-is
    #                     )
    #                 )

    #             mode = self.STAGE_MEMORY
    #         self.meta_info = {'input_pad_to': self.max_input_length,
    #                 'pad_to': self.config.gen_pad_to,
    #                 'generation_kwargs': {
    #                 'max_tokens': self.config.gen_max_tokens_memorization,
    #                 'n': 1 # note that we have already repeat n times in ray_trainer
    #             },
    #             'mode': mode,   # 传给后续，update 时可选用
    #         }
    #         logger.info(f'InfMem.action() done, stage={self.stage}')


    #     fixed_messages = []
    #     for msg in self.messages:
    #         if not torch.is_tensor(msg):
    #             msg = torch.tensor(msg, dtype=torch.long, device=chunk_i.device)

    #         # 截断：超长就只保留末尾 self.max_input_length
    #         if msg.size(0) > self.max_input_length:
    #             msg = msg[-self.max_input_length:]

    #     #     # 左侧 pad 到固定长度
    #     #     if msg.size(0) < self.max_input_length:
    #     #         pad_len = self.max_input_length - msg.size(0)
    #     #         pad = torch.full((pad_len,), pad_id, dtype=msg.dtype, device=msg.device)
    #     #         msg = torch.cat([pad, msg], dim=0)

    #         fixed_messages.append(msg)

    #     self.messages = fixed_messages
    #     self.final_mask_list.append(final_mask)
    #     self.sample_index_list.append(sample_index)
    #     return self.messages, self.meta_info





            
        #     # format: we use our token_template to avoid decoding & formatting with str function & encoding back.
        #     self.messages = [
        #         self.token_message_template.format(
        #                 prompt=prompt,
        #                 memory=memory if memory is not None else self.NO_MEMORY_TOKENS, # use pre-tokenized "No previous memory" for first round
        #                 chunk=chunk[chunk != self.tokenizer.pad_token_id], # unpadding needed here
        #         )
        #         for prompt, memory, chunk in zip(prompt_i, memory_i, chunk_i)
        #     ]
        #     sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
        #     final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False
        #     self.meta_info = {'input_pad_to': self.max_input_length,
        #                  'pad_to': self.config.gen_pad_to,
        #                  'generation_kwargs': {
        #                   'max_tokens': self.config.gen_max_tokens_memorization,
        #                   'n': 1 # note that we have already repeat n times in ray_trainer
        #                 }}
        #     logger.info(f'InfMem.action() done')
        # self.final_mask_list.append(final_mask)
        # self.sample_index_list.append(sample_index)
        # return self.messages, self.meta_info
