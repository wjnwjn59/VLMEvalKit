from .image_base import ImageBaseDataset
from ..smp import *
import re
import ast

ARTICLES = {'a', 'an', 'the'}
MANUAL_MAP = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
              'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hes": "he's", "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "lets": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "somebodyd": "somebody'd", "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "wholl": "who'll", "whos": "who's", "whove": "who've", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all", "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've"}

# anls
def normalize_infovqa_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = re.sub(r'\s+', ' ', ans)
    ans = re.sub(r'[\.,;:!?"\'\[\](){}]', '', ans)
    for k, v in MANUAL_MAP.items():
        ans = re.sub(r'\b' + re.escape(k) + r'\b', v, ans)
    return ans


def anls_score(pred: str, gts: list[str], threshold: float = 0.5) -> float:
    pred_norm = normalize_infovqa_answer(pred)
    gts_norm = [normalize_infovqa_answer(gt) for gt in gts]

    def levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if len(a) == 0:
            return len(b)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        curr = [0] * (len(b) + 1)
        for i, ca in enumerate(a):
            curr[0] = i + 1
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
            prev, curr = curr, prev
        return prev[len(b)]
    scores: list[float] = []
    for gt in gts_norm:
        dist = levenshtein(pred_norm, gt)
        length = max(len(pred_norm), len(gt))
        score = 1.0 - dist / length if length > 0 else 0.0
        scores.append(score)
    best = max(scores) if scores else 0.0
    return best if best >= threshold else 0.0

# relaxed
def to_float(text: object) -> float | None:
    try:
        return float(str(text).strip().strip('%'))
    except ValueError:
        return None


def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    def _to_float(text: object) -> float | None:
        try:
            text = str(text).strip()
            if text.endswith('%'):
                return float(text.rstrip('%')) / 100.0
            return float(text)
        except:
            return None
    pred_f, tgt_f = _to_float(prediction), _to_float(target)
    if pred_f is not None and tgt_f:
        return abs(pred_f - tgt_f) / abs(tgt_f) <= max_relative_change
    return str(prediction).lower().strip() == str(target).lower().strip()


# vqa score
def normalize_vqa_answer(ans: str) -> str:
    if len(ans) == 1:
        return ans.lower()
    punct = [';', r'/', '[', ']', '"',
             '}', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
    comma_strip = re.compile(r'(\d)(,)(\d)')

    out_text = ans
    for p in punct:
        if (p + ' ' in out_text or ' ' + p in out_text) or (re.search(comma_strip, out_text) is not None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = period_strip.sub('', out_text)

    words = [
        CONTRACTIONS.get(word, word)
        for word in out_text.lower().split()
        if word not in ARTICLES
    ]
    words = [MANUAL_MAP.get(word, word) for word in words]
    return ' '.join(words).strip()


def vqa_score(pred: str, gts: list[dict]) -> float:
    pred_norm = normalize_vqa_answer(pred)
    gts_norm = [normalize_vqa_answer(gt['answer']) for gt in gts]
    matches = [pred_norm == gt for gt in gts_norm]
    return min(1.0, sum(matches) / 3.0) if gts_norm else 0.0


class OurDataset(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'our_dataset': 'https://huggingface.co/datasets/KoiiVN/our_dataset/blob/main/our_dataset.tsv'
    }

    DATASET_MD5 = {
        'our_dataset': 'fc652743ec59bbcb8d00b9986a1d4a35'
    }

    def __init__(self, dataset='our_dataset', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['our_dataset']

    def build_prompt(self, line):
        msgs = []

        if isinstance(line['image'], list):
            msgs.extend([dict(type='image', value=img)
                        for img in line['image']])
        else:
            msgs.append(dict(type='image', value=line['image']))

        question = (
            "Answer the question according to the image using a single word or phrase. "
            "If the image does not contain enough evidence, answer exactly: unanswerable. "
            "Do not use outside knowledge.\n"
            f"{(line['question']).strip()}\n"
            'The last line of your response should be of the form "ANSWER: $ANSWER" '
            '(without quotes) where $ANSWER is the answer to the question.'
        )

        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        # Load predictions
        data = load(eval_file)

        correct = 0
        total = 0

        anls_scores = []
        relaxed_scores = []
        vqa_scores = []

        def clean_pred_text(p):
            s = str(p)
            s = s.replace('the answer is', '').replace('The answer is', '')
            s = s.rstrip('.').strip()
            return s

        for _, row in data.iterrows():
            pred_raw = row['prediction']
            ans_raw = row['answer']

            # ---------- Exact Match ----------
            pred_em = str(pred_raw).strip().upper()
            ans_em = str(ans_raw).strip().upper()
            if pred_em == ans_em:
                correct += 1
            total += 1

            pred_txt = clean_pred_text(pred_raw)
            ans_str = str(ans_raw).strip()

            # ---------- ANLS & relaxed_accuracy ----------
            gt_list = None
            try:
                if isinstance(ans_raw, list):
                    gt_list = [str(x) for x in ans_raw]
                else:
                    # cố gắng parse string thành Python object (list, dict, ...)
                    try:
                        parsed = ast.literal_eval(ans_str)
                    except Exception:
                        parsed = ans_str

                    if isinstance(parsed, list):
                        gt_list = [str(x).strip(" '") for x in parsed]
                    elif isinstance(parsed, dict) and 'answer' in parsed:
                        gt_list = [str(parsed['answer'])]
                    else:
                        gt_list = [str(parsed)]
            except Exception:
                gt_list = None

            # ---------- ANLS ----------
            if gt_list is not None:
                try:
                    score_anls = anls_score(pred_txt, gt_list, threshold=0.5)
                    anls_scores.append(score_anls)
                except Exception:
                    pass

            # ---------- Relaxed Accuracy ----------
            if gt_list is not None:
                try:
                    score_relaxed = 1.0 if any(
                        relaxed_correctness(gt_item, pred_txt) for gt_item in gt_list
                    ) else 0.0
                    relaxed_scores.append(score_relaxed)
                except Exception:
                    pass

            # ---------- VQA Score ----------
            try:
                gts_vqa = None
                if isinstance(ans_raw, list) and all(
                    isinstance(x, dict) and 'answer' in x for x in ans_raw
                ):
                    gts_vqa = ans_raw
                elif isinstance(ans_raw, str):
                    parsed = ast.literal_eval(ans_raw)
                    if isinstance(parsed, list) and all(
                        isinstance(x, dict) and 'answer' in x for x in parsed
                    ):
                        gts_vqa = parsed

                if gts_vqa is not None:
                    score_vqa = vqa_score(pred_txt, gts_vqa)
                    vqa_scores.append(score_vqa)
            except Exception:
                pass

        accuracy = correct / total * 100 if total > 0 else 0.0

        results = {
            'Acc': round(accuracy, 4),
        }

        if anls_scores:
            results['anls'] = round(
                sum(anls_scores) / len(anls_scores) * 100, 4)

        if relaxed_scores:
            results['relaxed_accuracy'] = round(
                sum(relaxed_scores) / len(relaxed_scores) * 100, 4)

        if vqa_scores:
            results['vqa_score'] = round(
                sum(vqa_scores) / len(vqa_scores) * 100, 4)

        return results
