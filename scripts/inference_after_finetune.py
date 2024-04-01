import re
import copy
import json
import time
import random
import csv
import torch
from typing import Dict, List, Literal, Optional, Union, Tuple
from http.client import HTTPException
from dataclasses import dataclass

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

@dataclass
class ChatMessage:
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str]
    function_call: Optional[Dict] = None

@dataclass
class ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]

@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

FUNCS = [
    {
        "name_for_human": "中文字词重组",
        "name_for_model": "zh_relation_search",
        "description_for_model": "中文字词重组是一个查找并输出符合要求的中文字词的工具，用于从图数据库中查找目标字词能关联到的其他字词。"
                                 "比如字加部首组成新字，字加字组成新字，字换掉部首组成新字，字和字组成词语，"
                                 "字和字组成成语，查找某形式的成语，查找某种读音的字词"
                                 + " Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "input",
                "description": "要查询的中文输入",
                "required": True,
                "schema": {"type": "list"},
            },
            {
                "name": "node_type",
                "description": "中文输入的形式, 必须是：字，词，成语",
                "required": True,
                "schema": {"type": "string", "enum": ["character", "word", "idiom"]}
            },
            {
                "name": "edge_type",
                "description": "能使用的边类型，如‘character_to_character’，‘character_to_word’",
                "required": True,
                "schema": {"type": "string",
                           "enum": ["character_to_character", "character_to_word", "radical_to_character",
                                    "character_to_idiom"]}
            },
            {
                "name": "position",
                "description": "输入字词在返回字词中的位置，除非用户指定，否则默认是包含（middle）",
                "required": False,
                "schema": {"type": "string", "enum": ["start", "end", "middle"]}
            },
            {
                "name": "number",
                "description": "返回字词的数量，除非用户指定个数，否则默认是一个",
                "required": False,
                "schema": {"type": "int"}
            },
            {
                "name": "length",
                "description": "每一个返回字词的字数，默认不传入该参数，除非用户指定",
                "required": False,
                "schema": {"type": "int"}
            },
        ],
    },
    {
        "name_for_human": "中文字词属性",
        "name_for_model": "zh_property_acquire",
        "description_for_model": "中文字词属性是一个中文字词的属性查询工具。"
                                 "输入中文字词本身和所要查询的属性名称，可以按照指定的数量和顺序，返回字词内容对应的属性内容。"
                                 "支持的中文字词形式包括：字，词，成语。"
                                 "属性包含以下内容：写法，同音字，近义词，反义词，例句，意思，部首；相近的表述也合并到这几大类属性中。"
                                 "如果未指定属性名称，则默认返回输入字词的写法。"
                                 + " Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "input",
                "description": "被查询的中文字词，必须是独立的字，词语，成语的形式，不是一句话",
                "required": True,
                "schema": {"type": "string"},
            },
            {
                "name": "zh_type",
                "description": "中文字词的形式",
                "required": True,
                "schema": {"type": "string", "enum": ["character", "word", "idiom"]}
            },
            {
                "name": "prop",
                "description": "被查询属性在数据库里的标签："
                               "写法对应write, 同音对应tongyin, 近义词对应synonyms, 反义词对应antonyms, "
                               "造句对应sentences, 意思对应meaning, 部首对应radical，拼音对应pinyin，笔画数对应strokeCount，"
                               "结构对应structure，查某一部分对应partial，形近对应xingjin，包含的故事对应story",
                "required": True,
                "schema": {"type": "string"}
            },
            {
                "name": "number",
                "description": "如果查询到多个，要返回的结果数量，除非用户指定个数，否则默认是1",
                "required": False,
                "schema": {"type": "int"}
            }
        ],
    },
    {
        "name_for_human": "量词搭配",
        "name_for_model": "zh_liangci",
        "description_for_model": "量词搭配是一个用来获得量词-名词的匹配信息的工具，输入名词返回合适搭配的量词，输入量词返回合适搭配的名词"
                                 + " Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "input",
                "description": "要匹配的中文输入，量词或者需要匹配量词的名词",
                "required": True,
                "schema": {"type": "string"},
            },
        ],
    },
    {
        "name_for_human": "中文字词格式召回",
        "name_for_model": "zh_format_retrieval",
        "description_for_model": "中文字词召回是一个能返回某种格式字词的工具。"
                                 "可以直接限定格式，也可以根据用户给出的例子来识别目标格式"
                                 "也可以召回形容喜悦的词语，这里‘喜悦’含义，‘悲伤’含义是需要满足的条件。"
                                 + " Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "format",
                "description": "要满足的格式，用英文大写字母字符串表示，相同的字母表示相同的字，比如冷冷清清的格式是AABB",
                "required": True,
                "schema": {"type": "string"}
            },
            {
                "name": "zh_type",
                "description": "要返回的字词形式, 只能是：字，词，成语",
                "required": True,
                "schema": {"type": "string", "enum": ['word', 'idiom']}
            },
        ],
    },
    {
        "name_for_human": "中文字词含义召回",
        "name_for_model": "zh_meaning_retrieval",
        "description_for_model": "中文字词含义召回是一个返回特定含义字词的工具。"
                                 "比如可以召回形容喜悦的词语，这里‘喜悦’是含义，‘悲伤’是含义。"
                                 + " Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "meaning",
                "description": "要满足的特定含义，用自然语言描述",
                "required": True,
                "schema": {"type": "string"}
            },
            {
                "name": "zh_type",
                "description": "要返回的字词形式, 只能是：字，词，成语",
                "required": True,
                "schema": {"type": "string", "enum": ['word', 'idiom']}
            },
        ],
    }
]

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You must access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def parse_messages(messages, functions):
    if all(m.role != "user" for m in messages):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )

    messages = copy.deepcopy(messages)
    default_system = "You are a helpful assistant."
    system = ""
    if messages[0].role == "system":
        system = messages.pop(0).content.lstrip("\n").rstrip()
        if system == default_system:
            system = ""

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get("name", "")
            name_m = func_info.get("name_for_model", name)
            name_h = func_info.get("name_for_human", name)
            desc = func_info.get("description", "")
            desc_m = func_info.get("description_for_model", desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        system += "\n\n" + REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        system = system.lstrip("\n").rstrip()

    dummy_thought = {
        "en": "\nThought: I now know the final answer.\nFinal answer: ",
        "zh": "\nThought: 我会作答了。\nFinal answer: ",
    }

    _messages = messages
    messages = []
    for m_idx, m in enumerate(_messages):
        role, content, func_call = m.role, m.content, m.function_call
        if content:
            content = content.lstrip("\n").rstrip()
        if role == "function":
            if (len(messages) == 0) or (messages[-1].role != "assistant"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role assistant before role function.",
                )
            messages[-1].content += f"\nObservation: {content}"
            if m_idx == len(_messages) - 1:
                messages[-1].content += "\nThought:"
        elif role == "assistant":
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role user before role assistant.",
                )
            last_msg = messages[-1].content
            last_msg_has_zh = len(re.findall(r"[\u4e00-\u9fff]+", last_msg)) > 0
            if func_call is None:
                if functions:
                    content = dummy_thought["zh" if last_msg_has_zh else "en"] + content
            else:
                f_name, f_args = func_call["name"], func_call["arguments"]
                if not content:
                    if last_msg_has_zh:
                        content = f"Thought: 我可以使用 {f_name} API。"
                    else:
                        content = f"Thought: I can use {f_name}."
                content = f"\n{content}\nAction: {f_name}\nAction Input: {f_args}"
            if messages[-1].role == "user":
                messages.append(
                    ChatMessage(role="assistant", content=content.lstrip("\n").rstrip())
                )
            else:
                messages[-1].content += content
        elif role == "user":
            messages.append(
                ChatMessage(role="user", content=content.lstrip("\n").rstrip())
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid request: Incorrect role {role}."
            )

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == "user":
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail="Invalid request")

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            usr_msg = messages[i].content.lstrip("\n").rstrip()
            bot_msg = messages[i + 1].content.lstrip("\n").rstrip()
            if system and (i == len(messages) - 2):
                usr_msg = f"{system}\n\nQuestion: {usr_msg}"
                system = ""
            for t in dummy_thought.values():
                t = t.lstrip("\n")
                if bot_msg.startswith(t) and ("\nAction: " in bot_msg):
                    bot_msg = bot_msg[len(t) :]
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Expecting exactly one user (or function) role before every assistant role.",
            )
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{system}\n\nQuestion: {query}"
    return query, history


def parse_response(response):
    """Parse model's output with Openai's function-call format"""
    func_name, func_args = "", ""
    i = response.rfind("\nAction:")
    j = response.rfind("\nAction Input:")
    k = response.rfind("\nObservation:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + "\nObservation:"  # Add it back.
        k = response.rfind("\nObservation:")
        func_name = response[i + len("\nAction:") : j].strip()
        func_args = response[j + len("\nAction Input:") : k].strip()
    if func_name:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content=response[:i],
                function_call={"name": func_name, "arguments": func_args},
            ),
            finish_reason="function_call",
        )
        return choice_data
    z = response.rfind("\nFinal Answer: ")
    if z >= 0:
        response = response[z + len("\nFinal Answer: ") :]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )
    return choice_data


def trim_stop_words(response, stop_words):
    """Crop response before stop_words"""
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


# To work around that unpleasant leading-\n tokenization issue!
def add_extra_stop_words(stop_words):
    """Add new stop_words"""
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip("\n")
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def text_complete_last_message(history, stop_words_ids, gen_kwargs):
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    for i, (query, response) in enumerate(history):
        query = query.lstrip("\n").rstrip()
        response = response.lstrip("\n").rstrip()
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"
    prompt = prompt[: -len(im_end)]

    _stop_words_ids = [tokenizer.encode(im_end)]
    if stop_words_ids:
        for s in stop_words_ids:
            _stop_words_ids.append(s)
    stop_words_ids = _stop_words_ids

    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = model.generate(input_ids, stop_words_ids=stop_words_ids, **gen_kwargs).tolist()[0]
    output = tokenizer.decode(output, errors="ignore")
    assert output.startswith(prompt)
    output = output[len(prompt) :]
    output = trim_stop_words(output, ["<|endoftext|>", im_end])
    print(f"<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>")
    return output

def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    gen_kwargs = {}
    if request.temperature is not None:
        if request.temperature < 0.01:
            gen_kwargs['top_k'] = 1  # greedy decoding
        else:
            # Not recommended. Please tune top_p instead.
            gen_kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        gen_kwargs['top_p'] = request.top_p

    stop_words = add_extra_stop_words(request.stop)
    if request.functions:
        stop_words = stop_words or []
        if "Observation:" not in stop_words:
            stop_words.append("Observation:")

    query, history = parse_messages(request.messages, request.functions)


def merge_lora():
    """Merge LoRA finetuned adapter with Qwen-14"""
    # After LoRA
    model = AutoPeftModelForCausalLM.from_pretrained(
        "/mnt/cfs/NLP/wky/output_qwen/checkpoint-300",
        device_map="auto",
        trust_remote_code=True,
        bf16=True
    ).eval()

    merged_model = model.merge_and_unload()
    # max_shard_size and safe serialization are not necessary. 
    # They respectively work for sharding checkpoint and save the model to safetensors
    merged_model.save_pretrained("merge_qwen14_ftv1_300", max_shard_size="2048MB", safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
                "/mnt/cfs/NLP/wky/output_qwen/checkpoint-300", # path to the output directory
                trust_remote_code=True
    )
                
    tokenizer.save_pretrained("merge_qwen14_ftv1_300")


def parse_action(text: str) -> Tuple[str, str]:
    #t = text.rfind('\nThought:')
    i = text.rfind('Action: ')
    j = text.rfind('\nAction Input: ')
    k = text.rfind("\nObservation:")
    if 0 <= i < j:
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + "\nObservation:"  # Add it back.
        k = text.rfind("\nObservation:")
        action = text[i + len('Action: '):j].strip()
        action_input = text[j + len('\nAction Input: '):k].strip()
        return action, action_input
    return '', ''


if __name__ == "__main__":

    #merge_lora()

    #MODEL_DIR = "merge_qwen14_ftv1_300"
    MODEL_DIR = "/mnt/cfs/NLP/jiali/work/pretrained_model/Qwen-14B-Chat"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto", 
        trust_remote_code=True
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        trust_remote_code=True,
        max_new_tokens=50,
        top_p=0.5,
        temperature=0.1,
        )

    random.seed(42)

    with open("zhzici_test.txt") as f:
        queries = f.readlines()
    random.shuffle(queries)
    
    out_lst = []
    for m in queries:
        m2 = ChatMessage(role="user", content=m)
        q, h_in = parse_messages([m2], functions=FUNCS)
        response, _ = model.chat(tokenizer, q, history=None)
        action, action_input = parse_action(response)
        
        out_lst.append([m.replace("\n", ""), action, action_input])
        print(out_lst[-1])

    fields = ["Query", "Function", "Arguments"]
    with open("qwen14_response.csv", "w") as csvfr:
        csvwriter = csv.writer(csvfr)
        csvwriter.writerow(fields)
        csvwriter.writerows(out_lst)