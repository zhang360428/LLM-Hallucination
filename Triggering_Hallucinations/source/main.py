import os
from transformers import pipeline, AutoTokenizer
import random
import re
from nltk.corpus import wordnet as wn
import nltk
import string

# 设置 NLTK 数据目录为本地目录
nltk.data.path.append('./Model/nltk_data')

# 加载预训练的填空模型，指定设备（如果有GPU可用，则使用GPU）
model_name = "./Model/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
fill_mask = pipeline("fill-mask", model="./Model/distilbert-base-uncased", device=-1)  # 使用GPU，如果没有GPU则使用device=-1


def generate_synonyms_from_bert(word):
    """使用预训练模型生成同义词"""
    candidates = fill_mask(f"The [MASK] cat")[0]['sequence'].replace('[CLS]', '').replace('[SEP]', '')
    candidates = re.findall(r'$(.*?)$', candidates)
    return [c.replace(' ', '') for c in candidates if c != word and c.strip()]


def perturb_text_with_token_replacement(text, num_replacements=1):
    """进行弱语义攻击：通过编码、替换 token、解码"""
    # 对原始文本进行编码
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # 随机选择要替换的 token
    replacement_positions = random.sample(range(len(tokens)), min(num_replacements, len(tokens)))

    perturbed_tokens = tokens[:]

    for pos in replacement_positions:
        token = tokens[pos]
        word = tokenizer.decode([token]).strip()

        # 获取该词的同义词
        synonyms = generate_synonyms_from_wordnet(word)

        # 如果有同义词，则随机替换
        if synonyms:
            synonym = random.choice(synonyms)
            new_token = tokenizer.encode(synonym, add_special_tokens=False)
            perturbed_tokens[pos] = new_token[0]  # 替换为新token

    # 解码为修改后的文本
    perturbed_text = tokenizer.decode(perturbed_tokens, skip_special_tokens=True)
    return perturbed_text

def generate_synonyms_from_wordnet(word):
    """使用 WordNet 库生成同义词"""
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:  # 去除原始词
                synonyms.add(lemma.name())
    return list(synonyms)


def insert_irrelevant_info_bak(text, num_inserts=1):
    """在文本中随机插入无关信息，增强无关信息的多样性"""
    words = text.split()
    irrelevant_infos = [
        "out of nowhere", "on the other hand", "by the way", "it’s hard to believe",
        "for no reason", "during a break", "as a side note", "on the fly",
        "by accident", "unexpectedly", "without warning"
    ]

    # 可以生成更丰富的无关信息，比如通过随机插入时间、地点、情感等词汇
    random_infos = ["in the park", "yesterday", "on TV", "in the morning", "suddenly", "loudly", "over the weekend"]
    irrelevant_infos.extend(random_infos)

    for _ in range(num_inserts):
        insert_pos = random.randint(0, len(words) - 1)
        irrelevant_info = random.choice(irrelevant_infos)
        words = words[:insert_pos] + [irrelevant_info] + words[insert_pos:]
    return ' '.join(words)

def insert_irrelevant_info(text, num_inserts=3):
    """在文本中随机插入无意义的乱码，包含特殊符号"""
    words = text.split()

    # 生成随机的乱码（包括字母、数字和特殊符号）
    def generate_gibberish(length=3):
        # 包括字母、数字、标点符号和空格
        charset = string.ascii_letters + string.digits + string.punctuation + ' '
        return ''.join(random.choices(charset, k=length))

    for _ in range(num_inserts):
        insert_pos = random.randint(0, len(words) - 1)
        gibberish = generate_gibberish()  # 生成一段乱码
        words = words[:insert_pos] + [gibberish] + words[insert_pos:]
    return ' '.join(words)


def semantic_perturbation(text, num_words=3):
    """对文本进行语义扰动，如词序调整"""
    words = text.split()
    perturbed_text = words[:]  # 创建单词列表的副本
    for _ in range(num_words):
        idx1, idx2 = random.sample(range(len(perturbed_text)), 2)
        perturbed_text[idx1], perturbed_text[idx2] = perturbed_text[idx2], perturbed_text[idx1]
    return ' '.join(perturbed_text)


def perturb_text(text, num_synonyms=2, num_inserts=2, num_words=3):
    """对文本进行幻觉诱导"""
    words = text.split()
    perturbed_text = []
    for word in words:
        if random.random() < 0.3:  # 30%的概率替换为同义词
            # 首先尝试使用BERT模型生成同义词
            synonyms = generate_synonyms_from_bert(word)
            if not synonyms:
                # 如果BERT没有返回同义词，则使用WordNet作为备用
                synonyms = generate_synonyms_from_wordnet(word)
            if synonyms:
                perturbed_text.append(random.choice(synonyms))
            else:
                perturbed_text.append(word)
        else:
            perturbed_text.append(word)

    perturbed_text = perturb_text_with_token_replacement(' '.join(perturbed_text))

    # 插入无关信息
    perturbed_text = insert_irrelevant_info(perturbed_text, num_inserts)

    # 进行语义扰动
    # perturbed_text = semantic_perturbation(perturbed_text, num_words)

    return perturbed_text

#数据目录，约定读取挂载数据的目录
data_path = "/datacon2024/AI"

#具体的文件目录
file_path=f"{data_path}/CQ1/Q.txt"
# file_path = "./Q.txt"

#输出答案的文件
result_path = "/result.txt"
# result_path = "./result.txt"


# A(·)
def gen_Q_star(file_path, output_path='./result.txt'):
    """
    Generates a new file Q_star based on the content of the provided Q file.

    This function reads the content of the file specified by file_path, 
    processes it to generate a new file Q_star, and saves it to the location 
    specified by output_path.

    :param file_path: The path to the original Q file.
    :type file_path: str

    :param output_path: The path where the generated Q_star file will be saved.
    :type output_path: str

    :return: None
    """

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(file_path, 'r', encoding='utf-8') as input_file, \
             open(output_path, 'a', encoding='utf-8') as output_file:
                
            # 逐行读取输入文件
                for q in input_file:
                    # induce hallucination for every query in Q.txt
                    q_star = perturb_text(q)
                    output_file.write(q_star.strip() + '\n')
                    
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except IOError as e:
        print(f"An I/O error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    else:
        print(f"File Q_star has been successfully generated at {output_path}")



if __name__ == "__main__": 

    gen_Q_star(file_path=file_path,
               output_path=result_path)
