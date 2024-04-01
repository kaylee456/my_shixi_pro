import pandas as pd
import openai
import random
import time
import signal
import functools
import csv
import requests
import json
from typing import List

openai.api_base = "http://120.48.110.5:8315/v1"  # 8316 是qwen-14B
openai.api_key = "none"

random.seed(42)

functions = [
    {
        "name_for_human": "中文字词重组",
        "name_for_model": "zh_relation_search",
        "description_for_model": "中文字词重组是一个查找并输出符合要求的中文字词的工具，用于从图数据库中查找目标字词能关联到的其他字词。"
                                 "比如字加部首组成新字，字加字组成新字，字换掉部首组成新字，字和字组成词语，"
                                 "字和字组成成语，查找某形式的成语，查找某种读音的字词",
        "parameters": [
            {
                "name": "input",
                "description": "要查询的中文输入",
                "required": True,
                "schema": {"type": "list"},
            },
            {
                "name": "node_type",
                "description": "中文输入的形式, 必须是：字，词，成语，格式，标签",
                "required": True,
                "schema": {"type": "string", "enum": ["zi", "word", "idiom", "format", "tag"]}
            },
            {
                "name": "edge_type",
                "description": "能使用的边类型，如‘zi_to_zi’，‘zi_to_word’",
                "required": True,
                "schema": {"type": "string",
                           "enum": ["zi_to_zi", "zi_to_word", "radical_to_zi", "zi_to_idiom", "format_to_idiom",
                                    "tag_to_word", "tag_to_idiom"]}
            },
            {
                "name": "position",
                "description": "输入字词在返回字词中的位置",
                "required": False,
                "schema": {"type": "string", "enum": ["start", "end", "middle", "not_specify"]}
            },
            {
                "name": "number",
                "description": "如果查询到多个，要返回的结果数量",
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
                                 "如果未指定属性名称，则默认返回输入字词的写法。",
        "parameters": [
            {
                "name": "input",
                "description": "被查询的中文字词，必须是独立的字，词语，成语的形式，不是一句话",
                "required": True,
                "schema": {"type": "list"},
            },
            {
                "name": "zh_type",
                "description": "中文字词的形式, 可以是：字，词，成语",
                "required": True,
                "schema": {"type": "string", "enum": ["zi", "word", "idiom"]}
            },
            {
                "name": "prop",
                "description": "被查询属性在数据库里的标签："
                               "写法对应strokeOrderURL, 同音对应tongyin, 近义词对应synonyms, 反义词对应antonyms, 例句对应sentences, 意思对应meaning, 部首radical",
                "required": True,
                "schema": {"type": "string"}
            },
            {
                "name": "number",
                "description": "如果查询到多个，要返回的结果数量",
                "required": False,
                "schema": {"type": "int"}
            },
            {
                "name": "order",
                "description": "如果查询到多个，按什么顺序返回；默认是按照出现频率顺序",
                "required": False,
                "schema": {"type": "string", "enum": ["frequence", "random"]}
            }
        ],
    },
    {
        "name_for_human": "量词搭配",
        "name_for_model": "zh_liangci",
        "description_for_model": "量词搭配是一个用来获得量词-名词的匹配信息的工具，输入名词返回量词，输入量词返回名词",
        "parameters": [
            {
                "name": "input",
                "description": "要匹配的中文输入",
                "required": True,
                "schema": {"type": "list"},
            },
        ],
    },
    {
        "name_for_human": "中文字词采样",
        "name_for_model": "zh_sample",
        "description_for_model": "中文字词采样是一个返回能满足输入特征的字词的工具，支持的属性只有形式和意思",
        "parameters": [
            {
                "name": "prop",
                "description": "输入的特征，形式或者意思",
                "required": True,
                "schema": {"type": "list", "enum": ['format', 'meaning']},
            },
            {
                "name": "zh_type",
                "description": "要返回的形式, 可以是：字，词，成语",
                "required": True,
                "schema": {"type": "string", "enum": ['zi', 'word', 'idiom']}
            },
        ],
    }
]

functions_v1 = [
    {
        "name_for_human": "中文字词重组",
        "name_for_model": "zh_relation_search",
        "description_for_model": "中文字词重组是一个查找并输出符合要求的中文字词的工具，用于从图数据库中查找目标字词能关联到的其他字词。"
                                 "比如字加部首组成新字，字加字组成新字，字换掉部首组成新字，字和字组成词语，"
                                 "字和字组成成语，查找某形式的成语，查找某种读音的字词",
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
                "description": "输入字词在返回字词中的位置",
                "required": False,
                "schema": {"type": "string", "enum": ["start", "end", "middle", "not_specify"]}
            },
            {
                "name": "number",
                "description": "如果查询到多个，要返回的结果数量",
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
                                 "如果未指定属性名称，则默认返回输入字词的写法。",
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
                               "写法对应strokeOrderURL, 同音对应tongyin, 近义词对应synonyms, 反义词对应antonyms, "
                               "例句对应sentences, 意思对应meaning, 部首对应radical，拼音对应pinyin，笔画数对应strokeCount，"
                               "结构对应structure，查某一部分对应partial，形近对应xingjin，故事对应story",
                "required": True,
                "schema": {"type": "string"}
            },
            {
                "name": "number",
                "description": "如果查询到多个，要返回的结果数量，默认是1",
                "required": False,
                "schema": {"type": "int"}
            },
            {
                "name": "order",
                "description": "如果查询到多个，按什么顺序返回；默认是按照频率",
                "required": False,
                "schema": {"type": "string", "enum": ["frequency", "random"]}
            }
        ],
    },
    {
        "name_for_human": "量词搭配",
        "name_for_model": "zh_liangci",
        "description_for_model": "量词搭配是一个用来获得量词-名词的匹配信息的工具，输入名词返回量词，输入量词返回名词",
        "parameters": [
            {
                "name": "input",
                "description": "要匹配的中文输入",
                "required": True,
                "schema": {"type": "list"},
            },
        ],
    },
    {
        "name_for_human": "中文字词采样",
        "name_for_model": "zh_sample",
        "description_for_model": "中文字词采样是一个返回能满足输入特征的字词的工具，支持的属性只有格式和意思"
                                 "比如可以搜索AABB格式的成语，AAB格式的词语，也可以搜索表达喜悦的成语，表达悲伤的词语。这里面AABB，AAB是格式，‘喜悦’，‘悲伤’是意思。",
        "parameters": [
            {
                "name": "prop",
                "description": "输入的特征，格式或者意思",
                "required": True,
                "schema": {"type": "list", "enum": ['format', 'meaning']},
            },
            {
                "name": "zh_type",
                "description": "要返回的字词形式, 必须是：字，词，成语",
                "required": True,
                "schema": {"type": "string", "enum": ['word', 'idiom']}
            },
        ],
    }
]

functions_v2 = [
    {
        "name_for_human": "中文字词重组",
        "name_for_model": "zh_relation_search",
        "description_for_model": "中文字词重组是一个查找并输出符合要求的中文字词的工具，用于从图数据库中查找目标字词能关联到的其他字词。"
                                 "比如字加部首组成新字，字加字组成新字，字换掉部首组成新字，字和字组成词语，"
                                 "字和字组成成语，查找某形式的成语，查找某种读音的字词",
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
                "description": "如果查询到多个，要返回的结果数量，除非用户指定个数，否则默认是1",
                "required": False,
                "schema": {"type": "int"}
            },
            {
                "name": "length",
                "description": "返回字词的总长度，默认不传入该参数，除非用户指定",
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
                                 "如果未指定属性名称，则默认返回输入字词的写法。",
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
                               "写法对应strokeOrderURL, 同音对应tongyin, 近义词对应synonyms, 反义词对应antonyms, "
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
            },
            {
                "name": "order",
                "description": "如果查询到多个，按什么顺序返回；默认是按照频率",
                "required": False,
                "schema": {"type": "string", "enum": ["frequency", "random"]}
            }
        ],
    },
    {
        "name_for_human": "量词搭配",
        "name_for_model": "zh_liangci",
        "description_for_model": "量词搭配是一个用来获得量词-名词的匹配信息的工具，输入名词返回合适搭配的量词，输入量词返回合适搭配的名词",
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
        "name_for_human": "中文字词召回",
        "name_for_model": "zh_retrieval",
        "description_for_model": "中文字词召回是一个返回满足某些特征的字词的工具，支持的特征只有格式和含义"
                                 "比如可以召回AABB格式的成语，这里AABB格式是需要满足的条件，代表着前两个字一样且后两个字一样的成语，比如寻寻觅觅；"
                                 "也可以召回形容喜悦的词语，这里‘喜悦’含义，‘悲伤’含义是需要满足的条件。",
        "parameters": [
            {
                "name": "condition",
                "description": "要满足的条件，用自然语言描述",
                "required": True,
                "schema": {"type": "string"}
            },
            {
                "name": "prop",
                "description": "输入的特征，格式或者意思",
                "required": True,
                "schema": {"type": "list", "enum": ['format', 'meaning']},
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

functions_v3 = [
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

functions_v3_openai = [
    {
        "name": "zh_relation_search",
        "description": "中文字词重组是一个查找并输出符合要求的中文字词的工具，用于从图数据库中查找目标字词能关联到的其他字词。"
                       "比如字加部首组成新字，字加字组成新字，字换掉部首组成新字，字和字组成词语，"
                       "字和字组成成语，查找某形式的成语，查找某种读音的字词",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "list",
                    "description": "要查询的中文输入"
                },
                "node_type": {
                    "type": "string",
                    "description": "中文输入的形式, 必须是字或者部首",
                    "enum": ["character", "radical"]
                },
                "edge_type": {
                    "type": "string",
                    "description": "能使用的边类型，如‘character_to_character’，‘character_to_word’",
                    "enum": ["character_to_character", "character_to_word", "radical_to_character",
                             "character_to_idiom"]
                },
                "position": {
                    "type": "string",
                    "description": "输入字词在返回字词中的位置，除非用户指定，否则默认是包含（middle）",
                    "enum": ["start", "end", "middle"]
                },
                "number": {
                    "type": "int",
                    "description": "返回字词的数量，除非用户指定个数，否则默认是一个",
                },
                "length": {
                    "type": "int",
                    "description": "每一个返回字词的字数，默认不传入该参数，除非用户指定",
                }
            },
            "required": ["input", "node_type", "edge_type"],
        },
    },
    {
        "name_for_model": "zh_property_acquire",
        "description": "中文字词属性是一个中文字词的属性查询工具。"
                       "输入中文字词本身和所要查询的属性名称，可以按照指定的数量和顺序，返回字词内容对应的属性内容。"
                       "支持的中文字词形式包括：字，词，成语。"
                       "属性包含以下内容：写法，同音字，近义词，反义词，例句，意思，部首；相近的表述也合并到这几大类属性中。"
                       "如果未指定属性名称，则默认返回输入字词的写法。",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "被查询的中文字词，必须是独立的字，词语，成语的形式，不是一句话",
                },
                "zh_type": {
                    "type": "string",
                    "description": "中文字词的形式",
                    "enum": ["character", "word", "idiom"]
                },
                "prop": {
                    "type": "string",
                    "description": "被查询属性在数据库里的标签："
                                   "写法对应write, 同音对应tongyin, 近义词对应synonyms, 反义词对应antonyms, "
                                   "造句对应sentences, 意思对应meaning, 部首对应radical，拼音对应pinyin，笔画数对应strokeCount，"
                                   "结构对应structure，查某一部分对应partial，形近对应xingjin，包含的故事对应story",
                },
                "number": {
                    "type": "int",
                    "description": "如果查询到多个，要返回的结果数量，除非用户指定个数，否则默认是1"
                }
            },
            "required": ["input", "zh_type", "prop"],
        },
    },
    {
        "name_for_model": "zh_liangci",
        "description": "量词搭配是一个用来获得量词-名词的匹配信息的工具，输入名词返回合适搭配的量词，输入量词返回合适搭配的名词",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "要匹配的中文输入，量词或者需要匹配量词的名词"
                }
            },
            "required": ["input"]
        },
    },
    {
        "name": "zh_format_retrieval",
        "description": "中文字词格式召回是一个能返回某种格式字词的工具。"
                       "可以直接限定格式，也可以根据用户给出的例子来识别目标格式"
                       "也可以召回形容喜悦的词语，这里‘喜悦’含义，‘悲伤’含义是需要满足的条件。",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "要满足的格式，用英文大写字母字符串表示，相同的字母表示相同的字，比如冷冷清清的格式是AABB",
                },
                "zh_type": {
                    "type": "string",
                    "description": "要返回的字词形式, 只能是：字，词，成语",
                    "enum": ["word", "idiom"]
                }
            },
            "required": ["format", "zh_type"]
        },
    },
    {
        "name": "zh_meaning_retrieval",
        "description": "中文字词含义召回是一个返回特定含义字词的工具。"
                       "比如可以召回形容喜悦的词语，这里‘喜悦’是含义，‘悲伤’是含义。",
        "parameters": {
            "type": "object",
            "properties": {
                "meaning": {
                    "type": "sting",
                    "description": "要满足的特定含义，用自然语言描述",
                },
                "zh_type": {
                    "type": "sting",
                    "description": "要返回的字词形式, 只能是：字，词，成语",
                    "enum": ['word', 'idiom']
                }
            },
            "required": ["meaning", "zh_type"],
        },
    }
]

functions_v3_openai_select_first = [
    {
        "name": "select_func",
        "description": "根据各个函数功能，选择一个最可能回答问题的函数",
        "parameters": {
            "type": "object",
            "properties": {
                "func": {
                    "type": "string",
                    "description": "Function names to complete the task.",
                    "enum": ["zh_relation_search", "zh_property_acquire", "zh_liangci", "zh_format_retrieval",
                             "zh_meaning_retrieval"]
                }
            },
            "required": ["func"]
        }
    },
    {
        "name": "zh_relation_search",
        "description": "中文字词重组是一个查找并输出符合要求的中文字词的工具，用于从图数据库中查找目标字词能关联到的其他字词。"
                       "比如字加部首组成新字，字加字组成新字，字换掉部首组成新字，字和字组成词语，"
                       "字和字组成成语，查找某形式的成语，查找某种读音的字词",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "list",
                    "description": "要查询的中文输入"
                },
                "node_type": {
                    "type": "string",
                    "description": "中文输入的形式, 必须是：字，词，成语",
                    "enum": ["character", "word", "idiom"]
                },
                "edge_type": {
                    "type": "string",
                    "description": "能使用的边类型，如‘character_to_character’，‘character_to_word’",
                    "enum": ["character_to_character", "character_to_word", "radical_to_character",
                             "character_to_idiom"]
                },
                "position": {
                    "type": "string",
                    "description": "输入字词在返回字词中的位置，除非用户指定，否则默认是包含（middle）",
                    "enum": ["start", "end", "middle"]
                },
                "number": {
                    "type": "int",
                    "description": "返回字词的数量，除非用户指定个数，否则默认是一个",
                },
                "length": {
                    "type": "int",
                    "description": "每一个返回字词的字数，默认不传入该参数，除非用户指定",
                }
            },
            "required": ["input", "node_type", "edge_type"],
        },
    },
    {
        "name_for_model": "zh_property_acquire",
        "description": "中文字词属性是一个中文字词的属性查询工具。"
                       "输入中文字词本身和所要查询的属性名称，可以按照指定的数量和顺序，返回字词内容对应的属性内容。"
                       "支持的中文字词形式包括：字，词，成语。"
                       "属性包含以下内容：写法，同音字，近义词，反义词，例句，意思，部首；相近的表述也合并到这几大类属性中。"
                       "如果未指定属性名称，则默认返回输入字词的写法。",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "被查询的中文字词，必须是独立的字，词语，成语的形式，不是一句话",
                },
                "zh_type": {
                    "type": "string",
                    "description": "中文字词的形式",
                    "enum": ["character", "word", "idiom"]
                },
                "prop": {
                    "type": "string",
                    "description": "被查询属性在数据库里的标签："
                                   "写法对应write, 同音对应tongyin, 近义词对应synonyms, 反义词对应antonyms, "
                                   "造句对应sentences, 意思对应meaning, 部首对应radical，拼音对应pinyin，笔画数对应strokeCount，"
                                   "结构对应structure，查某一部分对应partial，形近对应xingjin，包含的故事对应story",
                },
                "number": {
                    "type": "int",
                    "description": "如果查询到多个，要返回的结果数量，除非用户指定个数，否则默认是1"
                }
            },
            "required": ["input", "zh_type", "prop"],
        },
    },
    {
        "name_for_model": "zh_liangci",
        "description": "量词搭配是一个用来获得量词-名词的匹配信息的工具，输入名词返回合适搭配的量词，输入量词返回合适搭配的名词",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "要匹配的中文输入，量词或者需要匹配量词的名词"
                }
            },
            "required": ["input"]
        },
    },
    {
        "name": "zh_format_retrieval",
        "description": "中文字词召回是一个能返回某种格式字词的工具。"
                       "可以直接限定格式，也可以根据用户给出的例子来识别目标格式"
                       "也可以召回形容喜悦的词语，这里‘喜悦’含义，‘悲伤’含义是需要满足的条件。",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "要满足的格式，用英文大写字母字符串表示，相同的字母表示相同的字，比如冷冷清清的格式是AABB",
                },
                "zh_type": {
                    "type": "string",
                    "description": "要返回的字词形式, 只能是：字，词，成语",
                    "enum": ["word", "idiom"]
                }
            },
            "required": ["format", "zh_type"]
        },
    },
    {
        "name": "zh_meaning_retrieval",
        "description": "中文字词含义召回是一个返回特定含义字词的工具。"
                       "比如可以召回形容喜悦的词语，这里‘喜悦’是含义，‘悲伤’是含义。",
        "parameters": {
            "type": "object",
            "properties": {
                "meaning": {
                    "type": "sting",
                    "description": "要满足的特定含义，用自然语言描述",
                },
                "zh_type": {
                    "type": "sting",
                    "description": "要返回的字词形式, 只能是：字，词，成语",
                    "enum": ['word', 'idiom']
                }
            },
            "required": ["meaning", "zh_type"],
        },
    }
]


def ori1_extract():
    """Extract query from ori1.json"""
    out = []
    with open("ori1.json") as oj:
        qas = oj.readlines()
        for qa in qas:
            query = qa.split('"')[3]
            aim = qa.split('"')[-2]
            out.append([query, aim])

    fields = ["Query", "Aim"]
    with open("zhzici_real.csv", 'a') as csvfi:
        csvwriter = csv.writer(csvfi)
        csvwriter.writerow(fields)
        csvwriter.writerows(out)

    return

# ori1_extract()


def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after sec seconds
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func

    return decorator


@timeout(30)
def reply(message, functions, force_func=None):
    """get one-step response from Qwen"""
    if not force_func:
        response = openai.ChatCompletion.create(
            model="Qwen-72B-Chat",
            messages=message,
            seed=42,  # If set, try to be determinstic; Not supported in Qwen
            temperature=0.1,
            functions=functions,
            stream=False,
            stop=[]
        )
    else:
        response = openai.ChatCompletion.create(
            model="Qwen-72B-Chat",
            messages=message,
            seed=42,  # If set, try to be determinstic; Not supported in Qwen
            temperature=0.1,
            functions=functions,
            function_call={'name': force_func},  # Force the model to use only this function; Not supported in Qwen
            stream=False,
            stop=[]
        )

    return response


def test(num: int, functions: List, input_queries: None):
    # 中文字词测试query
    with open("zhzici_test.txt") as f:
        queries_test = f.readlines()
    random.shuffle(queries_test)

    # 中文字词真实query
    df = pd.read_csv("zhzici_real.csv")
    df_zh = df[df['Aim'] == '查中文字词成语']
    queries_real = df_zh['Query'].tolist()
    random.shuffle(queries_real)

    queries = queries_test

    queries = input_queries if input_queries is not None else queries
    print(f"Query amount in total: {len(queries)}")

    out_lst = []
    for q in queries:
        print(q)
        start = time.time()
        # qwen blog 的function call是通过ReAct格式prompt针对训练的，而且表示system role没有用
        m = [{"role": "system", "content": "You have to use the provided functions."}]  # 果然没用
        m.append({"role": "user", "content": q})
        try:
            response = reply(m, functions)
            r = response.choices[0].message
            print(r.content)
            r_dict = dict(r.function_call)
            func, arguments = r_dict['name'], r_dict['arguments']
            arguments = " ".join(arguments.splitlines())

        except:
            func, arguments = None, None
            pass

        end = time.time()

        out = [q.strip(), f"{end - start}"[:4], func, arguments]
        out_lst.append(out)

    fields = ["Query", "Time", "Function", "Arguments"]
    with open("qwen_respond.csv", "a") as csvfr:
        csvwriter = csv.writer(csvfr)
        csvwriter.writerow(fields)
        csvwriter.writerows(out_lst)


def online_test(in_query: None):
    """v2 machine online test"""
    url = "https://genie-internal.vdyoo.net/lui-api-gray/v1/lui/nlu"

    payload = json.dumps({
        'app_id': '200011',
        'request_id': '1',
        'version': '3',
        'asr_pinyin': '',
        'slot_fill_list': [{
            'key': '',
            'id': '',
            'name': ''
        }],
        'grade_id': '5',
        'platform': '5',
        'location': '0',
        'tal_id': 'TaloFwinat0Amf4OL2ZxkEkZag',
        'device_id': '76DD082391300103',
        'os_version': '4',
        'grade': '五年级',
        'asr_info': in_query
    })
    headers = {
        'X-Genie-AppId': '200011',
        'Content-Type': 'application/json; charset=utf-8'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    result = dict(json.loads(response.text))
    print(result)

    # 细分意图
    small_intent = result['data']['data'][0]['data']['task']
    #intent = result['data'].get('model_output_intent')
    # 端上结果
    tts = result['data'].get('tts_show')

    widget = result['data']['data'][0]['data']['widget']
    contents = dict(json.loads(widget))['content']

    titles = []
    if not len(contents) == 0:
        for d in contents:
            if isinstance(d, dict):
                titles.append(d['title'])

    output = ''.join(titles)

    for s in titles:
        if s in in_query:
            output = s

    return small_intent, output


#si, tis = online_test("迎风飘扬中杨的笔顺")

#test(50, functions_v3, ["用走造句"])

def test_with_online(num: int, functions: List, input_queries: None):
    # 中文字词测试query
    with open("zhzici_test.txt") as f:
        queries_test = f.readlines()
    random.shuffle(queries_test)

    # 中文字词真实query
    df = pd.read_csv("zhzici_real.csv")
    df_zh = df[df['Aim'] == '查中文字词成语']
    queries_real = df_zh['Query'].tolist()
    random.shuffle(queries_real)

    queries = queries_test

    queries = input_queries if input_queries is not None else queries
    print(f"Query amount in total: {len(queries)}")

    out_lst = []
    for q in queries:
        print(q)
        start = time.time()
        # qwen blog 的function call是通过ReAct格式prompt针对训练的，而且表示system role没有用
        m = [{"role": "system", "content": "You have to use the provided functions."}]  # 果然没用
        m.append({"role": "user", "content": q})
        try:
            response = reply(m, functions)
            r = response.choices[0].message
            r_dict = dict(r.function_call)
            func, arguments = r_dict['name'], r_dict['arguments']
            arguments = " ".join(arguments.splitlines())
            # print(r.content)
        except:
            func, arguments = None, None
            pass

        end = time.time()

        intent, c_intent, tts = online_test(q)

        out = [q.strip(), f"{end - start}"[:4], func, arguments, intent, c_intent, tts]
        out_lst.append(out)

    fields = ["Query", "Time", "Function", "Arguments", "Online_intent", "Chinese_intent", "Report"]
    with open("qwen_respond.csv", "a") as csvfr:
        csvwriter = csv.writer(csvfr)
        csvwriter.writerow(fields)
        csvwriter.writerows(out_lst)

# test_with_online(50, functions_v3, input_queries=None)


def two_step_test(num: int, functions: List, input_queries: None):
    """失败，qwen不能限定函数调用，function_call 参数不起作用"""
    # 中文字词测试query
    with open("zhzici_test.txt") as f:
        queries_test = f.readlines()
    random.shuffle(queries_test)

    # 中文字词真实query
    df = pd.read_csv("zhzici_real.csv")
    df_zh = df[df['Aim'] == '查中文字词成语']
    queries_real = df_zh['Query'].tolist()
    random.shuffle(queries_real)

    queries = queries_test

    queries = input_queries if input_queries is not None else queries
    print(f"Query amount in total: {len(queries)}")

    out_lst = []
    for q in queries[:num]:
        print(q)
        start = time.time()
        # qwen blog 的function call是通过ReAct格式prompt针对训练的，而且表示system role没有用
        m = [{"role": "system", "content": "You have to use the provided functions."}]
        m.append({"role": "user", "content": q})
        try:
            response = reply(m, functions, force_func='select_func')
            r = response.choices[0].message
            r_dict = dict(r.function_call)
            func, arguments = r_dict['name'], r_dict['arguments']
            arguments = " ".join(arguments.splitlines())
            print(func)
            try:
                assert func == "select_func"
            except AssertionError:
                func, arguments = None, None
                pass

        except:
            func, arguments = None, None
            pass

        if arguments is not None:
            a_dict = dict(arguments)
            selected_func = a_dict.get('func', None)
            print(selected_func)
            response_2nd = reply(m, functions, force_func=selected_func)
            r_2nd = response_2nd.choices[0].message
            r_2nd_dict = dict(r_2nd.function_call)
            func, arguments = r_2nd_dict['name'], r_2nd_dict['arguments']
            arguments = " ".join(arguments.splitlines())

        end = time.time()

        out = [q.strip(), f"{end - start}"[:4], func, arguments]
        out_lst.append(out)

    fields = ["Query", "Time", "Function", "Arguments"]
    with open("qwen_respond.csv", "a") as csvfr:
        csvwriter = csv.writer(csvfr)
        csvwriter.writerow(fields)
        csvwriter.writerows(out_lst)

# two_step_test(50, functions_v3_openai_select_first, input_queries=None)


def slot_type_test():
    with open("zhzici_test.txt") as f:
        queries = f.readlines()
    random.shuffle(queries)

    out_lst = []
    for q in queries:
        intent, titles = online_test(q)
        out = [q.strip(), intent, titles]
        out_lst.append(out)

    fields = ["Query", "Online_intent", "Slots"]
    with open("online_response.csv", "a") as csvfr:
        csvwriter = csv.writer(csvfr)
        csvwriter.writerow(fields)
        csvwriter.writerows(out_lst)

    return


# slot_type_test()