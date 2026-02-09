from __future__ import annotations

import re
import logging

import numpy as np
import torch
from torch.distributions.categorical import Categorical


def create_logger(logging_dir: str | None, rank: int) -> logging.Logger:
    """
    创建日志器。
    
    Args:
        logging_dir: 日志目录（仅 rank 0 有效）
        rank: 当前进程的 rank
        
    Returns:
        配置好的 Logger 实例
    """
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt") if logging_dir else logging.NullHandler()
            ]
        )
        return logging.getLogger(__name__)
    else:
        _logger = logging.getLogger(__name__)
        _logger.addHandler(logging.NullHandler())
        return _logger



@torch.no_grad()
def AE_SMILES_encode(sm, ae_model):
    if sm[0][:5] == "[CLS]":    sm = [s[5:] for s in sm]
    text_input = ae_model.tokenizer(sm).to(ae_model.device)
    text_input_ids = text_input
    text_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(text_input.device)
    if hasattr(ae_model.text_encoder2, 'bert'):
        output = ae_model.text_encoder2.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
    else:
        output = ae_model.text_encoder2(text_input_ids, attention_mask=text_attention_mask, return_dict=True).last_hidden_state

    if hasattr(ae_model, 'encode_prefix'):
        output = ae_model.encode_prefix(output)
        if ae_model.output_dim*2 == output.size(-1):
            mean, logvar = torch.chunk(output, 2, dim=-1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            output = mean + std * torch.randn_like(mean)
    return output


@torch.no_grad()
def generate(model, image_embeds, text, stochastic=True, prop_att_mask=None, k=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:   prop_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    token_output = model.text_encoder(text,
                                      attention_mask=text_atts,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=prop_att_mask,
                                      return_dict=True,
                                      is_decoder=True,
                                      return_logits=True,
                                      )[:, -1, :]  # batch*300
    if k:
        p = torch.softmax(token_output, dim=-1)
        if stochastic:
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
    if stochastic:
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    return token_output.unsqueeze(1)  # batch*1

@torch.no_grad()
def AE_SMILES_decode(pv, model, stochastic=False, k=2, max_length=150):
    if hasattr(model, 'decode_prefix'):
        pv = model.decode_prefix(pv)

    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError('Tokenizer is not defined')
    # test
    model.eval()
    candidate = []
    if k == 1:
        text_input = torch.tensor([tokenizer.cls_token_id]).expand(pv.size(0), 1).to(model.device)  # batch*1
        for _ in range(max_length):
            output = generate(model, pv, text_input, stochastic=False)
            if output.sum() == 0:
                break
            
            text_input = torch.cat([text_input, output], dim=-1)
        for i in range(text_input.size(0)):
            sentence = text_input[i]
            cdd = tokenizer.decode(sentence)[0]#newtkn
            candidate.append(cdd)
    else:
        for prop_embeds in pv:
            prop_embeds = prop_embeds.unsqueeze(0)
            product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(model.device)
            values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
            product_input = torch.cat([torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(model.device), indices.squeeze(0).unsqueeze(-1)], dim=-1)
            current_p = values.squeeze(0)
            final_output = []
            for _ in range(max_length):
                values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
                k2_p = current_p[:, None] + values
                product_input_k2 = torch.cat([product_input.unsqueeze(1).repeat(1, k, 1), indices.unsqueeze(-1)], dim=-1)
                if tokenizer.sep_token_id in indices:
                    ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                    for e in ends:
                        p = k2_p[e[0], e[1]].cpu().item()
                        final_output.append((p, product_input_k2[e[0], e[1]]))
                        k2_p[e[0], e[1]] = -1e5
                    if len(final_output) >= k ** 1:
                        break
                current_p, i = torch.topk(k2_p.flatten(), k)
                next_indices = torch.from_numpy(np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))).T
                product_input = torch.stack([product_input_k2[i[0], i[1]] for i in next_indices], dim=0)

            candidate_k = []
            final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
            for p, sentence in final_output:
                cdd = tokenizer.decode(sentence[:-1])[0]#newtkn
                candidate_k.append(cdd)
            if candidate_k == []:
                candidate.append("")
            else:
                candidate.append(candidate_k[0])
            # candidate.append(random.choice(candidate_k))
    return candidate


@torch.no_grad()
def qwen3_encode(descriptions, qwen, qwen_tokenizer, description_length, device):
    tokenized = qwen_tokenizer(
        descriptions,
        padding="max_length",
        truncation=True,
        max_length=description_length,
        return_tensors="pt",
    ).to(device)

    outputs = qwen(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    encoder_outputs = outputs.hidden_states[-1]
    return encoder_outputs, tokenized.attention_mask


class regexTokenizer:
    def __init__(self, vocab_path="./vocab_bpe_300_sc.txt", max_len=127):
        with open(vocab_path, "r", encoding="utf-8") as f:
            x = f.readlines()
            x = [xx.replace("##", "") for xx in x]
            x2 = x.copy()
        x2.sort(key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(token).strip()[:-1] for token in x2) + ")"
        self.rg = re.compile(pattern)

        self.idtotok = {cnt: i.strip() for cnt, i in enumerate(x)}
        self.vocab_size = len(self.idtotok)
        self.toktoid = {v: k for k, v in self.idtotok.items()}
        self.max_len = max_len
        self.cls_token_id = self.toktoid["[CLS]"]
        self.sep_token_id = self.toktoid["[SEP]"]
        self.pad_token_id = self.toktoid["[PAD]"]

    def decode_one(self, iter):
        if self.sep_token_id in iter:
            iter = iter[: (iter == self.sep_token_id).nonzero(as_tuple=True)[0][0].item()]
        return "".join([self.idtotok[i.item()] for i in iter[1:]])

    def decode(self, ids: torch.Tensor):
        if len(ids.shape) == 1:
            return [self.decode_one(ids)]
        smiles = []
        for i in ids:
            smiles.append(self.decode_one(i))
        return smiles

    def __len__(self):
        return self.vocab_size

    def __call__(self, smis: list, truncation="max_len"):
        tensors = []
        lengths = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            length, tensor = self.encode_one(i)
            tensors.append(tensor)
            lengths.append(length)
        output = torch.concat(tensors, dim=0)
        if truncation == "max_len":
            return output
        if truncation == "longest":
            return output[:, : max(lengths)]
        raise ValueError("truncation should be either max_len or longest")

    def encode_one(self, smi):
        smi = "[CLS]" + smi + "[SEP]"
        res = [self.toktoid[i] for i in self.rg.findall(smi)]
        token_length = len(res)
        if token_length < self.max_len:
            res += [self.pad_token_id] * (self.max_len - len(res))
        else:
            res = res[: self.max_len]
        return token_length, torch.LongTensor([res])

