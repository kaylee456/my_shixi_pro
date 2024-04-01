import random
import pandas as pd
import json

# character
df = pd.read_excel('./raw_data/char_word_idiom/char3500.xls')
char_list = [x.strip() for x in df['hz']]
random.shuffle(char_list)
# word
word_list = [x.split('\t')[0] for x in open('./raw_data/char_word_idiom/word.txt').readlines()]
random.shuffle(word_list)
# idiom
idiom_list = [x.split('\t')[0].strip() for x in open('./raw_data/char_word_idiom/idiom.txt').readlines()]
random.shuffle(idiom_list)

TOOLS = [
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


def add_zi(char):
    """天 -> 天字"""
    return random.choice([f'{char}字', char])

def get_char_from_word(word: str):
    """秋天的天"""
    slot = random.choice(word)
    return f'{word}的{slot}', slot

zh_property_acquire = [
    {
        'prop': {
            'write': ['写法'],
            'tongyin': ['同音字', '同音词'],
            'synonyms': ['近义词'],
            'antonyms': ['反义词'],
            'meaning': ['意思'],
            'sentences': ['例句', '造句', '好句'],
            'radical': ['部首'],
            'partial': ['上半部分', '下半部分', '一半', '左半边', '右半边'],
            'strokeCount': ['笔画数'],
            'structure': ['结构', '构造'],
            'xingjin': ['形近字'],
            'story': ['故事']
        },
        'suffix': ['', '是什么', '有啥', '是啥', '有什么', '是', '应该是'],
        'number': ['', '一个', '两个', '三个', '四个', '五个', '六个', '七个', '八个', '九个']  # 1-9
    },
    {
        'middle': ['怎么', '如何', '是啥', '该怎么', '要怎么', '可以怎么', '咋'],
        'prop': {
            'write': ['写'],
            'pinyin': ['读', '拼', '说'],
            'sentences': ['造句'],
        }
    },
]

zh_relation_search = [
    # 三个开头是一的四字成语
    {
        'target': {
            'word': '词语',
            'idiom': '成语',
        },
        'position': {
            'start': ['开头是'],
            'middle': ['包含', '含有'],
            'end': ['结尾是']
        },
        'number': ['', '一个', '两个', '三个', '四个', '五个', '六个', '七个', '八个', '九个'],
        'length': {
            0: [''],
            2: ['二字', '两个字'],
            3: ['三字', '三个字'],
            4: ['四字', '四个字']
        },
        'end': ['', '有哪些', '有啥', '举例'],
    },
    # 天组词
    {
        'target': {
            'word': '词',
            'idiom': '成语',
        },
        'middle': ['能组什么', '组'],
        'length': {
            0: [''],
            2: ['二字', '两个字'],
            3: ['三字', '三个字'],
            4: ['四字', '四个字']
        }
    }
]


def create_query_input(slot, func, zh_type, real_slot=None):
    if real_slot is None:
        real_slot = slot
    if func == 'zh_property_acquire':
        if zh_type == 'char':
            slot = add_zi(slot)
        format_prop_0, format_prop_1 = zh_property_acquire[0], zh_property_acquire[1]

        # format_prop_0
        prop = random.choice(list(format_prop_0['prop'].keys()))
        str_prop = random.choice(format_prop_0['prop'][prop])

        str_suffix = random.choice(format_prop_0['suffix'])

        num = random.choices([i for i in range(10)], weights=[8, 1, 2, 2, 2, 2, 2, 1, 1, 1], k=1)[0]
        str_num = format_prop_0['number'][num]

        query_0 = random.choice([f'{str_num}{slot}的{str_prop}{str_suffix}', f'{slot}的{str_prop}{str_suffix}{str_num}'])
        gt_action_input_0 = {'input': real_slot,
                             'zh_type': zh_type,
                             'prop': prop,
                             'number': max(1, num)}

        # format_prop_1
        prop = random.choice(list(format_prop_1['prop'].keys()))
        str_prop = random.choice(format_prop_1['prop'][prop])

        str_middle = random.choice(format_prop_1['middle'])
        query_1 = f'{slot}{str_middle}{str_prop}'
        gt_action_input_1 = {'input': real_slot,
                             'zh_type': zh_type,
                             'prop': prop,
                             }

        query, gt_action_input = random.choices([(query_0, gt_action_input_0), (query_1, gt_action_input_1)], weights=[3, 2], k=1)[0]

    elif func == 'zh_relation_search':
        format_relation_0, format_relation_1 = zh_relation_search[0], zh_relation_search[1]
        
        # format_relation_0
        num = random.choices([i for i in range(10)], weights=[8, 1, 2, 2, 2, 2, 2, 1, 1, 1], k=1)[0]
        str_num = format_relation_0['number'][num]
        
        position = random.choice(list(format_relation_0['position'].keys()))
        str_pos = random.choice(format_relation_0['position'][position])
        
        target = random.choice(list(format_relation_0['target'].keys()))
        str_tar = format_relation_0['target'][target]
        
        length = random.choices(list(format_relation_0['length'].keys()), weights=[5, 1, 1, 1], k=1)[0]
        str_len = random.choice(format_relation_0['length'][length])

        end = random.choices(format_relation_0['end'], weights=[5, 1, 1, 1], k=1)[0]
        
        query_0 = f'{str_num}{str_pos}{slot}的{str_len}{str_tar}{end}'
        gt_action_input_0 = {'input': [real_slot],
                             'node_type': zh_type,
                             'edge_type': f'character_to_{target}',
                             'position': position,
                             'number': max(1, num),
                             'length': length
                             }


        # format_relation_1
        target = random.choice(list(format_relation_0['target'].keys()))
        str_tar = format_relation_0['target'][target]

        length = random.choices(list(format_relation_0['length'].keys()), weights=[5, 1, 1, 1], k=1)[0]
        str_len = random.choice(format_relation_0['length'][length])
        if length == 0:  # 不指定length的话，输入None
            length = None

        middle = random.choice(format_relation_1['middle'])

        query_1 = f'{slot}{middle}{str_tar}{str_len}'
        gt_action_input_1 = {'input': [real_slot],
                             'node_type': zh_type,
                             'edge_type': f'character_to_{target}',
                             'length': length
                             }

        query, gt_action_input = random.choice([(query_0, gt_action_input_0), (query_1, gt_action_input_1)])

    else:
        raise NotImplementedError

    gt_action = func

    return query, gt_action, gt_action_input


def build_react_instruction(functions):

    TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

    REACT_INSTRUCTION = """Answer the following questions as best you can. You must access to the following APIs:

    {tools_text}

    Use the following format:

    Question: the input question you must answer
    Action: the action to take, should be one of [{tools_name_text}]
    Action Input: the input to the action
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!"""

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
            description_for_model=desc_m,
            parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
        )
        tools_text.append(tool)
        tools_name_text.append(name_m)
    tools_text = "\n\n".join(tools_text)
    tools_name_text = ", ".join(tools_name_text)
    instruction = REACT_INSTRUCTION.format(
        tools_text=tools_text,
        tools_name_text=tools_name_text,
    )
    return instruction


def format_train_sample(messages):
    #
    # You do not need the `function` role, as Qwen's function calling is actually implemented via ReAct,
    # not by adding a `function` role or `function_call` message. See openai_api.py for details.
    #
    # If you need the `system` role, you might need to modify `finetune.py` accordingly.
    #
    assert set(m["role"] for m in messages) == {"user", "assistant"}

    sample = {
        "conversations": [
            {
                "from": m["role"],
                "value": m["content"],
            }
            for m in messages
        ]
    }
    return sample


def char_samples_generate(num, function):

    data = random.sample(char_list, num)
    trainset = data[:int(num * 0.8)]
    testset = data[int(num * 0.8):]

    train_label, train_sample = [], []
    for s in trainset:
        query, gt_action, gt_action_input = create_query_input(s, function, 'character')

        instruction = build_react_instruction(TOOLS)
        train_sample.append(
            format_train_sample(
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": f"""
        Action: {gt_action}\nAction Input: {gt_action_input}
                        """.strip(),
                },
            ]
            )
        )
        train_label.append((query, s, gt_action, gt_action_input))

    df_train = pd.DataFrame(train_label, columns=['query', 'slot', 'action', 'action_input'])

    test_sample, test_label = [], []
    for s in testset:
        query, gt_action, gt_action_input = create_query_input(s, function, 'character')
        instruction = build_react_instruction(TOOLS)
        test_sample.append(
            format_train_sample(
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": f"""
        Action: {gt_action}\nAction Input: {gt_action_input}
                        """.strip(),
                },
            ]
            )
        )
        test_label.append((query, s, gt_action, gt_action_input))

    df_test = pd.DataFrame(test_label, columns=['query', 'slot', 'action', 'action_input'])

    return df_train, df_test, train_sample, test_sample


def char_from_word_generate(num, function):

    data = random.sample(word_list, num)
    trainset = data[:int(num * 0.8)]
    testset = data[int(num * 0.8):]

    train_sample, train_label = [], []
    for s in trainset:
        slot, real_slot = get_char_from_word(s)
        query, gt_action, gt_action_input = create_query_input(slot, function, 'character', real_slot)

        instruction = build_react_instruction(TOOLS)
        train_sample.append(
            format_train_sample(
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": f"""
        Action: {gt_action}\nAction Input: {gt_action_input}
                        """.strip(),
                },
            ]
            )
        )
        train_label.append((query, s, gt_action, gt_action_input))
    df_train = pd.DataFrame(train_label, columns=['query', 'slot', 'action', 'action_input'])

    test_sample, test_label = [], []
    for s in testset:
        slot, real_slot = get_char_from_word(s)
        query, gt_action, gt_action_input = create_query_input(slot, function, 'character', real_slot)
        instruction = build_react_instruction(TOOLS)
        test_sample.append(
            format_train_sample(
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": f"""
        Action: {gt_action}\nAction Input: {gt_action_input}
                        """.strip(),
                },
            ]
            )
        )
        test_label.append((query, real_slot, gt_action, gt_action_input))
    df_test = pd.DataFrame(test_label, columns=['query', 'slot', 'action', 'action_input'])

    return df_train, df_test, train_sample, test_sample


def word_samples_generate(num, function):

    data = random.sample(word_list, num)
    trainset = data[:int(num * 0.8)]
    testset = data[int(num * 0.8):]

    train_sample, train_label = [], []
    for s in trainset:
        query, gt_action, gt_action_input = create_query_input(s, function, 'word')

        instruction = build_react_instruction(TOOLS)
        train_sample.append(
            format_train_sample(
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": f"""
                Action: {gt_action}\nAction Input: {gt_action_input}
                                """.strip(),
                },
            ]
            )
        )
        train_label.append((query, s, gt_action, gt_action_input))
    df_train = pd.DataFrame(train_label, columns=['query', 'slot', 'action', 'action_input'])

    test_sample, test_label = [], []
    for s in testset:
        query, gt_action, gt_action_input = create_query_input(s, function, 'word')
        instruction = build_react_instruction(TOOLS)
        test_sample.append(
            format_train_sample(
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": f"""
                Action: {gt_action}\nAction Input: {gt_action_input}
                                """.strip(),
                },
            ]
            )
        )
        test_label.append((query, s, gt_action, gt_action_input))
    df_test = pd.DataFrame(test_label, columns=['query', 'slot', 'action', 'action_input'])

    return df_train, df_test, train_sample, test_sample


def idiom_samples_generate(num, function):

    data = random.sample(idiom_list, num)
    trainset = data[:int(num * 0.8)]
    testset = data[int(num * 0.8):]

    train_sample, train_label = [], []
    for s in trainset:
        query, gt_action, gt_action_input = create_query_input(s, function, 'idiom')
        instruction = build_react_instruction(TOOLS)
        train_sample.append(
            format_train_sample(
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": f"""
                Action: {gt_action}\nAction Input: {gt_action_input}
                                """.strip(),
                },
            ]
            )
        )
        train_label.append((query, s, gt_action, gt_action_input))
    df_train = pd.DataFrame(train_label, columns=['query', 'slot', 'action', 'action_input'])

    test_sample, test_label = [], []
    for s in testset:
        query, gt_action, gt_action_input = create_query_input(s, function, 'idiom')

        instruction = build_react_instruction(TOOLS)
        test_sample.append(
            format_train_sample(
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": f"""
                    Action: {gt_action}\nAction Input: {gt_action_input}
                    """.strip(),
                },
            ]
            )
        )
        test_label.append((query, s, gt_action, gt_action_input))
    df_test = pd.DataFrame(test_label, columns=['query', 'slot', 'action', 'action_input'])

    return df_train, df_test, train_sample, test_sample


def main():
    """960 trainset, 240 testset"""
    train_samples = []
    valid_samples = []

    _, _, t, v = char_samples_generate(200, 'zh_property_acquire')
    train_samples.extend(t)
    valid_samples.extend(v)

    _, _, t, v = char_samples_generate(200, 'zh_relation_search')
    train_samples.extend(t)
    valid_samples.extend(v)

    _, _, t, v = char_from_word_generate(200, 'zh_property_acquire')
    train_samples.extend(t)
    valid_samples.extend(v)

    _, _, t, v = char_from_word_generate(200, 'zh_relation_search')
    train_samples.extend(t)
    valid_samples.extend(v)

    _, _, t, v = word_samples_generate(200, 'zh_property_acquire')
    train_samples.extend(t)
    valid_samples.extend(v)

    _, _, t, v = idiom_samples_generate(200, 'zh_property_acquire')
    train_samples.extend(t)
    valid_samples.extend(v)

    #print(train_samples[0]['conversations'][1])

    random.shuffle(train_samples)  # Necessary ? HF's Trainer will shuffle automaticlly
    random.shuffle(valid_samples)

    print(f'{len(train_samples)} train samples.')
    print(f'{len(valid_samples)} valid samples.')

    with open(
        "func_call_train_samples.json", "w"
    ) as fout:  # data for fine-tuning
        fout.write(json.dumps(train_samples, indent=2, ensure_ascii=False))

    with open(
        "func_call_test_samples.json", "w"
    ) as fout:  # data for fine-tuning
        fout.write(json.dumps(train_samples, indent=2, ensure_ascii=False))


main()
