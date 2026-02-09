import os
import re
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from .smiles_canonicalization import canonicalize_molecule_smiles, get_molecule_id


RDLogger.DisableLog('rdApp.*')


def convert_smiles_list_into_mol_list(smiles_list, raise_error_when_error=False):
    mol_list = []
    no_answer_labels = []
    invalid_labels = []
    for smiles in smiles_list:
        if smiles == '':
            mol = 'NA'
            no_answer_labels.append(True)
            if raise_error_when_error:
                raise ValueError('SMILES is empty.')
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol = 'INVALID'
                invalid_labels.append(True)
                if raise_error_when_error:
                    raise ValueError('SMILES is not valid: %s' % smiles)
        mol_list.append(mol)
    
    no_answer_labels = np.array(no_answer_labels)
    invalid_labels = np.arange(invalid_labels)

    return mol_list, no_answer_labels, invalid_labels


def judge_exact_match(pred_can_smiles_list, gold_can_smiles_list):
    assert len(pred_can_smiles_list) == len(gold_can_smiles_list)
    exact_match_labels = []
    for pred_smiles, gold_smiles_list in zip(pred_can_smiles_list, gold_can_smiles_list):
        if pred_smiles is None:
            exact_match_labels.append(False)
            continue
        pred_smiles_inchi = get_molecule_id(pred_smiles)
        sample_exact_match = False
        for gold_smiles in gold_smiles_list:
            assert gold_smiles is not None
            gold_smiles_inchi = get_molecule_id(gold_smiles)
            if pred_smiles_inchi == gold_smiles_inchi:
                sample_exact_match = True
                break
        exact_match_labels.append(sample_exact_match)
    return np.array(exact_match_labels)


def calculate_fingerprint_similarity(pred_mol_list, gold_mols_list, morgan_r=2):
    assert len(pred_mol_list) == len(gold_mols_list)
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    for pred_mol, gold_mol_list in zip(pred_mol_list, gold_mols_list):
        if pred_mol is None or type(pred_mol) == str:
            raise ValueError(type(pred_mol))
        tmp_MACCS, tmp_RDK, tmp_morgan = 0, 0, 0
        for gold_mol in gold_mol_list:
            tmp_MACCS = max(tmp_MACCS, DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gold_mol), MACCSkeys.GenMACCSKeys(pred_mol), metric=DataStructs.TanimotoSimilarity))
            tmp_RDK = max(tmp_RDK, DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gold_mol), Chem.RDKFingerprint(pred_mol), metric=DataStructs.TanimotoSimilarity))
            tmp_morgan = max(tmp_morgan, DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gold_mol,morgan_r), AllChem.GetMorganFingerprint(pred_mol, morgan_r)))
        MACCS_sims.append(tmp_MACCS)
        RDK_sims.append(tmp_RDK)
        morgan_sims.append(tmp_morgan)
    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    return maccs_sims_score, rdk_sims_score, morgan_sims_score


def judge_multiple_match(pred_can_smiles_list, golds_can_smiles_list):
    assert len(pred_can_smiles_list) == len(golds_can_smiles_list)
    subset_labels = []
    intersection_labels = []
    for pred_smiles, gold_smiles_list in zip(pred_can_smiles_list, golds_can_smiles_list):
        if pred_smiles is None:
            subset_labels.append(False)
            intersection_labels.append(False)
            continue

        pred_ele_set = set()
        for smiles in pred_smiles.split('.'):
            pred_ele_set.add(get_molecule_id(smiles, remove_duplicate=False))

        intersection_label = False
        subset_label = False
        for gold_smiles in gold_smiles_list:
            assert gold_smiles is not None
            gold_ele_set = set()
            for smiles in gold_smiles.split('.'):
                gold_ele_set.add(get_molecule_id(smiles, remove_duplicate=False))

            if len(pred_ele_set & gold_ele_set) > 0:
                intersection_label = True
                g_p = gold_ele_set - pred_ele_set
                if len(g_p) >= 0 and len(pred_ele_set - gold_ele_set) == 0:
                    subset_label = True
                    break
        intersection_labels.append(intersection_label)
        subset_labels.append(subset_label)
    
    return intersection_labels, subset_labels


def calculate_smiles_metrics(
        preds_smiles_list, 
        golds_smiles_list,
        metrics=('exact_match', 'fingerprint')
):
    num_all = len(preds_smiles_list)
    if num_all == 0:
        # 返回空结果，而不是抛出异常
        return {
            'num_all': 0,
            'num_t1_no_answer': 0,
            'num_t1_invalid': 0,
            'num_t1_exact_match': 0,
            't1_rdk_fps': 0.0,
            't1_maccs_fps': 0.0,
            't1_morgan_fps': 0.0,
        }
    assert num_all == len(golds_smiles_list)
    
    # 确定k值（每个预测的候选数量）
    k = 1  # 默认值
    for pred_smiles_list in preds_smiles_list:
        if pred_smiles_list is not None:
            try:
                k = len(pred_smiles_list)
                break  # 找到第一个非None的元素，确定k值
            except (TypeError, IndexError):
                continue

    dk_pred_smiles_list_dict = {}
    dk_pred_no_answer_labels_dict = {}
    dk_pred_invalid_labels_dict = {}
    for dk in range(k):
        dk_pred_smiles_list_dict[dk] = []
        dk_pred_no_answer_labels_dict[dk] = []
        dk_pred_invalid_labels_dict[dk] = []
    for pred_smiles_list in tqdm(preds_smiles_list):
        if pred_smiles_list is None:
            for dk in range(k):
                dk_pred_no_answer_labels_dict[dk].append(True)
                dk_pred_invalid_labels_dict[dk].append(False)
                dk_pred_smiles_list_dict[dk].append(None)
            continue
        assert len(pred_smiles_list) == k
        for dk, item in enumerate(pred_smiles_list):
            # item = item.strip()
            if item == '' or item is None:
                item = None
                dk_pred_no_answer_labels_dict[dk].append(True)
                dk_pred_invalid_labels_dict[dk].append(False)
            else:
                dk_pred_no_answer_labels_dict[dk].append(False)
                item = canonicalize_molecule_smiles(item)
                if item is None:
                    dk_pred_invalid_labels_dict[dk].append(True)
                else:
                    dk_pred_invalid_labels_dict[dk].append(False)
            dk_pred_smiles_list_dict[dk].append(item)
    
    new_list = []
    for gold_smiles_list in tqdm(golds_smiles_list):
        sample_gold_smiles_list = []
        for gold in gold_smiles_list:
            item = gold.strip()
            new_item = canonicalize_molecule_smiles(item, return_none_for_error=False)
            # if new_item is None:
            #     new_item = item  #TODO
            # assert new_item is not None, item
            sample_gold_smiles_list.append(new_item)
        new_list.append(sample_gold_smiles_list)
    golds_smiles_list = new_list

    metric_results = {'num_all': num_all}

    tk_pred_no_answer_labels = np.array([True] * num_all)
    tk_pred_invalid_labels = np.array([True] * num_all)
    for dk in range(k):
        dk_no_answer_labels = dk_pred_no_answer_labels_dict[dk]
        dk_invalid_labels = dk_pred_invalid_labels_dict[dk]
        tk_pred_no_answer_labels = tk_pred_no_answer_labels & dk_no_answer_labels
        tk_pred_invalid_labels = tk_pred_invalid_labels & dk_invalid_labels
        metric_results['num_t%d_no_answer' % (dk + 1)] = tk_pred_no_answer_labels.sum().item()
        metric_results['num_t%d_invalid' % (dk + 1)] = tk_pred_invalid_labels.sum().item()
    
    # d1_no_answer_labels = dk_pred_no_answer_labels_dict[0]
    # # print(np.array(d1_no_answer_labels).sum().item())
    # for label, item in zip(d1_no_answer_labels, preds_smiles_list):
    #     if label:
    #         print(item)

    for metric in metrics:
        if metric == 'exact_match':
            tk_exact_match_labels = np.array([False] * num_all)
            for dk in range(k):
                dk_pred_smiles_list = dk_pred_smiles_list_dict[dk]
                dk_exact_match_labels = judge_exact_match(dk_pred_smiles_list, golds_smiles_list)
                tk_exact_match_labels = tk_exact_match_labels | dk_exact_match_labels
                metric_results['num_t%d_exact_match' % (dk + 1)] = tk_exact_match_labels.sum().item()
        elif metric == 'fingerprint':
            d1_pred_mol_list = []
            gold_mols_list = []
            for pred_smiles, gold_smiles_list, no_answer, invalid in zip(dk_pred_smiles_list_dict[0], golds_smiles_list, dk_pred_no_answer_labels_dict[0], dk_pred_invalid_labels_dict[0]):
                if pred_smiles is None or pred_smiles.strip() == '' or no_answer is True or invalid is True:
                    continue
                pred_mol = Chem.MolFromSmiles(pred_smiles)
                # if pred_mol is None:  # TODO
                #     continue
                assert pred_mol is not None, pred_smiles
                gold_mol_list = []
                for gold_smiles in gold_smiles_list:
                    gold_mol = Chem.MolFromSmiles(gold_smiles)
                    # if gold_mol is None:
                    #     continue  # TODO
                    assert gold_mol is not None, gold_smiles
                    gold_mol_list.append(gold_mol)
                # if len(gold_mol_list) == 0:
                #     continue  # TODO
                d1_pred_mol_list.append(pred_mol)
                gold_mols_list.append(gold_mol_list)
            maccs_sims_score, rdk_sims_score, morgan_sims_score = calculate_fingerprint_similarity(d1_pred_mol_list, gold_mols_list)
            metric_results['t1_maccs_fps'] = maccs_sims_score
            metric_results['t1_rdk_fps'] = rdk_sims_score
            metric_results['t1_morgan_fps'] = morgan_sims_score
        elif metric == 'multiple_match':
            tk_intersection_labels = np.array([False] * num_all)
            tk_subset_labels = np.array([False] * num_all)
            for dk in range(k):
                dk_intersection_labels, dk_subset_labels = judge_multiple_match(dk_pred_smiles_list_dict[dk], golds_smiles_list)
                tk_intersection_labels = tk_intersection_labels | dk_intersection_labels
                tk_subset_labels = tk_subset_labels | dk_subset_labels
                metric_results['num_t%d_subset' % (dk + 1)] = tk_subset_labels.sum().item()
                metric_results['num_t%d_intersection' % (dk + 1)] = tk_intersection_labels.sum().item()
        else:
            raise ValueError(metric)
    
    return metric_results


def judge_string_exact_match(pred_string_list, golds_string_list):
    """
    判断字符串精确匹配
    改进：标准化后再比较（去除多余空格、统一大小写）
    """
    exact_match_labels = []
    for pred_string, gold_string_list in zip(pred_string_list, gold_string_list):
        if not pred_string or not isinstance(pred_string, str):
            exact_match_labels.append(False)
            continue
        
        # 标准化pred：去除首尾空格，统一大小写
        pred_normalized = pred_string.strip().lower()
        
        exact_match = False
        for gold_string in gold_string_list:
            if not gold_string or not isinstance(gold_string, str):
                continue
            # 标准化gold
            gold_normalized = gold_string.strip().lower()
            if pred_normalized == gold_normalized:
                exact_match = True
                break
        exact_match_labels.append(exact_match)
    return np.array(exact_match_labels)


def judge_string_split_match(pred_string_list, golds_string_list, separater=';'):
    """
    判断字符串split匹配（用于IUPAC任务等）
    改进：标准化字符串（去除多余空格、统一大小写等）后再比较
    """
    exact_match_labels = []
    for pred_string, gold_string_list in zip(pred_string_list, golds_string_list):
        if not pred_string or not isinstance(pred_string, str):
            exact_match_labels.append(False)
            continue
            
        # 标准化pred：去除多余空格、标点，统一大小写
        pred_normalized = [s.strip().lower() for s in pred_string.split(separater) if s.strip()]
        pred_item = tuple(sorted(pred_normalized))
        
        exact_match = False
        for gold_string in gold_string_list:
            if not gold_string or not isinstance(gold_string, str):
                continue
            # 标准化gold
            gold_normalized = [s.strip().lower() for s in gold_string.split(separater) if s.strip()]
            gold_item = tuple(sorted(gold_normalized))
            if pred_item == gold_item:
                exact_match = True
                break
        exact_match_labels.append(exact_match)
    return np.array(exact_match_labels)


def parse_molecule(molecular_formula):
    valid = re.match(r'([A-Za-z]\d*)+([\+\-]\d*)*$', molecular_formula)
    if valid is None:
        raise ValueError("Molecular formula \"%s\" is not valid." % molecular_formula)

    stack = [defaultdict(int)]

    def _parse_formula(formula, _stack):

        # Set remainder equal to 'None'
        r = None

        # Regular expression matching for each of the three cases:
        atom = re.match(r'([A-Z][a-z]?)(\d+)?', formula)
        opening = re.match(r'[\(\[\{]', formula)
        closing = re.match(r'[\)\]\}](\d+)?', formula)

        # If atom is identified:
        if atom:
            r = formula[len(atom.group()):]
            _stack[-1][atom.group(1)] += int(atom.group(2) or 1)

        # If opening brackets encountered:
        elif opening:
            r = formula[len(opening.group()):] #this sets the remainder equal to everything after the opening brackets
            _stack.append(defaultdict(int)) 

        # If closing brackets encountered:
        elif closing:
            r = formula[len(closing.group()):] #this sets the remainder equal to everything after the closing brackets
            for (k, v) in _stack.pop().items():
                _stack[-1][k] += v * int(closing.group(1) or 1) #v times amount of molecule k, depending on nesting

        # If anything remains, process remainders recursively as nested formulas:
        if r:
            _parse_formula(r, _stack)

        return dict(_stack[0])
    
    result = _parse_formula(molecular_formula, stack)

    charge = re.search(r'[\+\-]\d*', molecular_formula)
    if charge is not None:
        charge_str = charge.group()
        charge_type = charge_str[0]
        if len(charge_str) == 1:
            charge_num = 1
        else:
            charge_num = int(charge_str[1:])
        result[charge_type] = charge_num

    return result


def count_element_match(pred_formula_list, golds_formula_list):
    assert len(pred_formula_list) == len(golds_formula_list)
    ele_match_labels = []
    ele_invalid_labels = []
    for pred_formula, gold_formula_list in zip(pred_formula_list, golds_formula_list):
        if pred_formula == '' or pred_formula is None:
            ele_invalid_labels.append(False)
            ele_match_labels.append(False)
            continue
        try:
            pred_ele = parse_molecule(pred_formula)
        except KeyboardInterrupt:
            raise
        except:
            # print(pred_formula)
            # print('=====')
            ele_invalid_labels.append(True)
            ele_match_labels.append(False)
            continue
        ele_invalid_labels.append(False)
        ele_match = False
        for gold_formula in gold_formula_list:
            gold_ele = parse_molecule(gold_formula)
            if pred_ele == gold_ele:
                ele_match = True
                break
        ele_match_labels.append(ele_match)
    return ele_match_labels, ele_invalid_labels


def calculate_formula_metrics(
        preds_formula_list,
        golds_formula_list,
        metrics=('element_match',)
):
    """
    Calculate metrics for molecular formula. Here we use element_match (equals to exact_match used in our paper) by default, which compares the atom numbers and ignore the orders.
    For example, C5H8 == H8C5.
    """
    num_all = len(preds_formula_list)
    assert len(preds_formula_list) == len(golds_formula_list)
    
    # 处理空列表
    if num_all == 0:
        result = {'num_all': 0}
        for metric in metrics:
            if metric == 'element_match':
                result['num_t1_ele_match'] = 0
                result['num_t1_formula_invalid'] = 0
            elif metric == 'split_match':
                result['num_t1_split_match'] = 0
            elif metric == 'exact_match':
                result['num_t1_exact_match'] = 0
        return result
    
    # 确定k值（每个预测的候选数量）
    k = 1  # 默认值
    for sample_formula_list in preds_formula_list:
        if sample_formula_list is not None:
            try:
                k = len(sample_formula_list)
                break  # 找到第一个非None的元素，确定k值
            except (TypeError, IndexError):
                continue
    
    dk_pred_formula_list_dict = dict()
    for dk in range(k):
        dk_pred_formula_list_dict[dk] = []
    for sample_formula_list in preds_formula_list:
        if sample_formula_list is None:
            for dk in range(k):
                dk_pred_formula_list_dict[dk].append('')
            continue
        # 确保sample_formula_list是列表或可迭代对象
        try:
            formula_len = len(sample_formula_list)
            if formula_len != k:
                # 如果长度不匹配，转换为列表并补齐或截断
                sample_formula_list = list(sample_formula_list)[:k]
                while len(sample_formula_list) < k:
                    sample_formula_list.append('')
        except (TypeError, AttributeError):
            # 如果不是可迭代对象（如单个字符串），将其转换为单元素列表
            sample_formula_list = [sample_formula_list]
            while len(sample_formula_list) < k:
                sample_formula_list.append('')
        
        for dk in range(k):
            item = sample_formula_list[dk] if dk < len(sample_formula_list) else ''
            dk_pred_formula_list_dict[dk].append(item)
    golds_formula_list = [[small_item.strip() for small_item in item] for item in golds_formula_list]
    new_golds_formula_list = []
    for item in golds_formula_list:
        new_item = []
        for small_item in item:
            small_item = small_item.strip()
            assert small_item != ''
            new_item.append(small_item)
        new_golds_formula_list.append(new_item)
    golds_formula_list = new_golds_formula_list


    metric_results = {'num_all': num_all}

    tk_no_answer_labels = np.array([True] * num_all)
    for dk in range(k):
        dk_pred_formula_list = dk_pred_formula_list_dict[dk]
        dk_no_answer_labels = []
        for item in dk_pred_formula_list:
            if item == '' or item is None:
                dk_no_answer_labels.append(True)
            else:
                dk_no_answer_labels.append(False)
        dk_no_answer_labels = np.array(dk_no_answer_labels)
        tk_no_answer_labels = tk_no_answer_labels & dk_no_answer_labels
        metric_results['num_t%d_no_answer' % (dk + 1)] = tk_no_answer_labels.sum().item()

    for metric in metrics:
        if metric == 'exact_match':
            tk_exact_match_labels = np.array([False] * num_all)
            for dk in range(k):
                dk_pred_formula_list = dk_pred_formula_list_dict[dk]
                dk_exact_match_labels = judge_string_exact_match(dk_pred_formula_list, golds_formula_list)
                tk_exact_match_labels = tk_exact_match_labels | dk_exact_match_labels
                metric_results['num_t%d_exact_match' % (dk + 1)] = tk_exact_match_labels.sum().item()
        elif metric == 'element_match':
            tk_ele_match_labels = np.array([False] * num_all)
            tk_formula_invalid_labels = np.array([True] * num_all)
            for dk in range(k):
                dk_pred_formula_list = dk_pred_formula_list_dict[dk]
                dk_ele_match_labels, dk_formula_invalid_labels = count_element_match(dk_pred_formula_list, golds_formula_list)
                tk_ele_match_labels = tk_ele_match_labels | dk_ele_match_labels
                tk_formula_invalid_labels = tk_formula_invalid_labels & dk_formula_invalid_labels
                metric_results['num_t%d_ele_match' % (dk + 1)] = tk_ele_match_labels.sum().item()
                metric_results['num_t%d_formula_invalid' % (dk + 1)] = tk_formula_invalid_labels.sum().item()
        elif metric == 'split_match':
            tk_exact_match_labels = np.array([False] * num_all)
            for dk in range(k):
                dk_pred_formula_list = dk_pred_formula_list_dict[dk]
                dk_exact_match_labels = judge_string_split_match(dk_pred_formula_list, golds_formula_list)
                tk_exact_match_labels = tk_exact_match_labels | dk_exact_match_labels
                metric_results['num_t%d_split_match' % (dk + 1)] = tk_exact_match_labels.sum().item()
        else:
            raise ValueError(metric)
    
    return metric_results


def calculate_text_metrics(
    pred_text_list,
    gold_text_list,
    text_model=None,
    text_trunc_length=512,
):
    assert len(pred_text_list) == len(gold_text_list)
    
    # 处理空列表
    if len(pred_text_list) == 0:
        return {
            'num_all': 0,
            'num_no_answer': 0,
            'bleu2': 0.0,
            'bleu4': 0.0,
            'rouge_1': 0.0,
            'rouge_2': 0.0,
            'rouge_l': 0.0,
            'meteor_score': 0.0,
        }
    
    pred_text_list = [(item[0].strip() if item is not None else '') for item in pred_text_list]
    gold_text_list = [item[0].strip() for item in gold_text_list]

    num_no_answer = 0
    for pred_formula in pred_text_list:
        if pred_formula == '':
            num_no_answer += 1

    if text_model is None:
        text_model = os.getenv("SCIBERT_MODEL_PATH", "allenai/scibert_scivocab_uncased")

    try:
        text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
        use_simple_tokenizer = False
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"[WARN] Failed to load tokenizer '{text_model}': {exc}")
        print("[WARN] Falling back to simple whitespace tokenizer (metrics may differ).")

        class _SimpleTokenizer:
            def tokenize(self, text, truncation=True, max_length=None, padding=None):
                tokens = text.strip().split()
                if max_length is not None:
                    tokens = tokens[:max_length]
                return tokens

        text_tokenizer = _SimpleTokenizer()
        use_simple_tokenizer = True

    meteor_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(zip(gold_text_list, pred_text_list)):
        if out == '':
            continue

        gt_tokens = text_tokenizer.tokenize(
            gt,
            truncation=True,
            max_length=text_trunc_length,
            padding='max_length',
        )
        out_tokens = text_tokenizer.tokenize(
            out,
            truncation=True,
            max_length=text_trunc_length,
            padding='max_length',
        )

        if not use_simple_tokenizer:
            # Remove special tokens emitted by pretrained tokenizers.
            for special in ('[PAD]', '[CLS]', '[SEP]'):
                gt_tokens = list(filter((special).__ne__, gt_tokens))
                out_tokens = list(filter((special).__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

    _meteor_score = np.mean(meteor_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(zip(gold_text_list, pred_text_list)):
        if out == '':
            continue

        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

    result = {
        'num_all': len(pred_text_list),
        'num_no_answer': num_no_answer,
        'bleu2': bleu2,
        'bleu4': bleu4,
        'rouge_1': rouge_1,
        'rouge_2': rouge_2,
        'rouge_l': rouge_l,
        'meteor_score': _meteor_score,
    }

    return result


def calculate_number_metrics(pred_text_list, gold_text_list):
    assert len(pred_text_list) == len(gold_text_list)
    num_all = len(pred_text_list)
    
    # 处理空列表
    if num_all == 0:
        return {
            'num_all': 0,
            'num_no_answer': 0,
            'num_invalid': 0,
            'RMSE': 0.0,
        }
    
    metrics = {}
    metrics['num_all'] = num_all
    num_no_answer = 0
    num_invalid = 0
    new_pred_text_list, new_gold_text_list = [], []
    for (pred_item, gold_item) in zip(pred_text_list, gold_text_list):
        if pred_item is None:
            num_no_answer += 1
            continue
        assert len(pred_item) == 1
        assert len(gold_item) == 1
        pred_item = pred_item[0]
        gold_item = gold_item[0]
        if pred_item == '':
            num_no_answer += 1
            continue
        try:
            pred_item = float(pred_item)
        except (SyntaxError, ValueError):
            # print("\"%s\"" % pred_item)
            num_invalid += 1
            continue
        gold_item = float(gold_item)
        new_pred_text_list.append(pred_item)
        new_gold_text_list.append(gold_item)
    
    new_pred_text_list = np.array(new_pred_text_list)
    new_gold_text_list = np.array(new_gold_text_list)
    score = np.sqrt(((new_pred_text_list - new_gold_text_list) ** 2).mean())
    
    metrics['num_no_answer'] = num_no_answer
    metrics['num_invalid'] = num_invalid
    metrics['RMSE'] = score

    return metrics


def _normalize_bool_to_yesno(text: str):
    """
    将各种布尔表示规范化为 yes/no
    支持: yes/no, true/false, y/n, 1/0, positive/negative, toxic/non-toxic, safe/unsafe等
    """
    if not isinstance(text, str):
        return None
    text = text.strip().lower()
    
    # 移除末尾的标点
    text = text.rstrip('.,;:!?')
    
    # 映射表 - 正面值（Yes/True/Positive）
    yes_values = {"yes", "y", "true", "t", "1", "positive", "toxic", "unsafe", "harmful"}
    # 映射表 - 负面值（No/False/Negative）
    no_values = {"no", "n", "false", "f", "0", "negative", "non-toxic", "non toxic", "nontoxic", "safe", "non-harmful"}
    
    # 直接匹配
    if text in yes_values:
        return "yes"
    elif text in no_values:
        return "no"
    
    # 正则匹配（处理包含额外文本的情况，如 "No, I do not suggest..."）
    import re
    bool_pattern = re.compile(r'\b(yes|no|true|false|toxic|non[-\s]?toxic|safe|unsafe)\b', re.IGNORECASE)
    match = bool_pattern.search(text)
    if match:
        val = match.group(1).lower()
        if val in ("yes", "true", "toxic", "unsafe", "harmful"):
            return "yes"
        elif val in ("no", "false", "non-toxic", "non toxic", "nontoxic", "safe"):
            return "no"
    
    return None


def calculate_boolean_metrics(pred_text_list, gold_text_list):
    assert len(pred_text_list) == len(gold_text_list)
    num_all = len(pred_text_list)
    
    # 处理空列表
    if num_all == 0:
        return {
            'num_all': 0,
            'num_no_answer': 0,
            'num_invalid': 0,
            'num_correct': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mcc': 0.0,
        }
    
    metrics = {}
    metrics['num_all'] = num_all
    num_no_answer = 0
    num_invalid = 0
    num_correct = 0
    new_pred_text_list, new_gold_text_list = [], []
    
    for (pred_item, gold_item) in zip(pred_text_list, gold_text_list):
        if pred_item is None or pred_item == '':
            num_no_answer += 1
            continue
        assert len(pred_item) == 1
        assert len(gold_item) == 1
        
        # 使用规范化函数
        pred_normalized = _normalize_bool_to_yesno(pred_item[0])
        gold_normalized = _normalize_bool_to_yesno(gold_item[0])
        
        if pred_normalized is None:
            num_invalid += 1
            continue
        if gold_normalized is None:
            # Gold标准无效，跳过此样本
            continue
        
        pred_val = 1 if pred_normalized == 'yes' else 0
        gold_val = 1 if gold_normalized == 'yes' else 0
        new_pred_text_list.append(pred_val)
        new_gold_text_list.append(gold_val)
        if gold_val == pred_val:
            num_correct += 1

    metrics['num_no_answer'] = num_no_answer
    metrics['num_invalid'] = num_invalid
    metrics['num_correct'] = num_correct

    # 计算F1等指标
    if len(new_gold_text_list) == 0:
        # 没有有效预测
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1_score'] = 0.0
        metrics['mcc'] = 0.0
    else:
        new_gold_text_list = np.array(new_gold_text_list)
        new_pred_text_list = np.array(new_pred_text_list)
        
        # 使用binary分类，明确设置zero_division
        metrics['f1_score'] = f1_score(new_gold_text_list, new_pred_text_list, average='binary', zero_division=0)
        metrics['precision'] = precision_score(new_gold_text_list, new_pred_text_list, average='binary', zero_division=0)
        metrics['recall'] = recall_score(new_gold_text_list, new_pred_text_list, average='binary', zero_division=0)
        
        # MCC计算（需要将0转换为-1以避免MCC计算问题）
        new_gold_text_list_mcc = new_gold_text_list.copy()
        new_gold_text_list_mcc[new_gold_text_list_mcc == 0] = -1
        new_pred_text_list_mcc = new_pred_text_list.copy()
        new_pred_text_list_mcc[new_pred_text_list_mcc == 0] = -1
        metrics['mcc'] = matthews_corrcoef(new_gold_text_list_mcc, new_pred_text_list_mcc)

    return metrics