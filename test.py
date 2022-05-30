# load dict and test using it

import json
import sys
import pandas as pd

import torch
from tqdm import tqdm

# from transformers import EncoderDecoderConfig, BertConfig, EncoderDecoderModel
from transformers import BartForConditionalGeneration
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer


@torch.no_grad()
def inference():
    step = sys.argv[1]

    model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
    ckpt = "model.pt"
    device = "cuda"

    model.load_state_dict(
        torch.load(
            f"saved/{ckpt}.{step}", map_location="cuda"
        ),
        strict=True,
    )

    model = model.half().eval().to(device)
    tokenizer = get_kobart_tokenizer()

    # text = """
    #  업무상과실치상죄에 있어서의 ‘업무’란 사람의 사회생활면에서 하나의 지위로서 계속적으로 종사하는 사무를 말하고, 여기에는 수행하는 직무 자체가 위험성을 갖기 때문에 안전배려를 의무의 내용으로 하는 경우는 물론 사람의 생명·신체의 위험을 방지하는 것을 의무내용으로 하는 업무도 포함되는데, 안전배려 내지 안전관리 사무에 계속적으로 종사하여 위와 같은 지위로서의 계속성을 가지지 아니한 채 단지 건물의 소유자로서 건물을 비정기적으로 수리하거나 건물의 일부분을 임대하였다는 사정만으로는 업무상과실치상죄에 있어서의 ‘업무’로 보기 어렵다. 
    # """

    test_data = open("data/test.jsonl", "r").read().splitlines()
    submission = []
    # PATH = "./model.pt"

    test_set = []
    for data in test_data:
        data = json.loads(data)
        article_original = data["article_original"]
        article_original = " ".join(article_original)
        news_id = data["id"]
        test_set.append((news_id, article_original))

    # with open("./submission.tsv", 'a', encoding='utf-8-sig', newline='') as f:
    for i, (news_id, text) in tqdm(enumerate(test_set)):
        # if i < 2293:
        #     continue
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids.to(device), eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # wr = csv.writer(f, delimiter='\t')
        # wr.writerow([i, news_id, output])
        # print(news_id, output)
        case_dict = {}
        case_dict['id'] = news_id
        case_dict['output'] = output
        submission.append(case_dict)
        print(news_id, output)
    case_df = pd.DataFrame(submission)
    case_df.to_csv(f"submission.csv", encoding='utf-8-sig', index=False)

if __name__ == '__main__':
    inference()
