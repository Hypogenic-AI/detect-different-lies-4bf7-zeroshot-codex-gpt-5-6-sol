#!/usr/bin/env python3
"""Semantically adjudicate answer/reference non-matches; saves every external judgment."""
import json, re, string, unicodedata
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def norm(s):
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower()
    s=re.sub(r"\([^)]*\)"," ",s); s=re.sub(r"\b(a|an|the)\b"," ",s)
    s="".join(c if c not in string.punctuation else " " for c in s)
    return " ".join(s.split())
def lexical(ans, aliases):
    a=norm(ans)
    if not a:return False
    for z in aliases:
        z=norm(z)
        if a==z:return True
        if len(z)>=4 and re.search(r"(?:^| )"+re.escape(z)+r"(?:$| )",a):return True
        if len(a)>=4 and re.search(r"(?:^| )"+re.escape(a)+r"(?:$| )",z):return True
    return False
def key(r): return f'{r["id"]}::{r["condition"]}::{r["replicate"]}'

def main():
    rows=[json.loads(x) for x in open("results/raw_generations.jsonl")]
    todo=[]; labels={}; seen={}
    for r in rows:
        k=key(r)
        if lexical(r["answer_text"],r["aliases"]): labels[k]={"correct":True,"source":"lexical"}
        else:
            signature=(r["question"],r["answer"],r["answer_text"])
            if signature not in seen:
                seen[signature]=len(todo)
                todo.append({"case_id":len(todo),"question":r["question"],"reference_answer":r["answer"],
                             "acceptable_aliases":r["aliases"],"candidate_answer":r["answer_text"]})
    model="Qwen/Qwen2.5-7B-Instruct (local semantic judge)"; judged={}; rawlog=[]
    instruction=("Grade whether the candidate is a correct short answer to the trivia question given the reference. "
      "Accept paraphrases, concise subsets that fully answer the question, and harmless extra specificity. Reject related "
      "but wrong answers, answers missing a required part, and answers with any contradiction. Be conservative. "
      "Reply with exactly CORRECT or INCORRECT.")
    tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",cache_dir="models",padding_side="left")
    tok.pad_token=tok.eos_token
    mdl=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",cache_dir="models",torch_dtype=torch.bfloat16,
                                             device_map="cuda",attn_implementation="sdpa").eval()
    for start in range(0,len(todo),24):
        batch=todo[start:start+24]; prompts=[]
        for x in batch:
            user=json.dumps({k:v for k,v in x.items() if k!="case_id"},ensure_ascii=False)
            prompts.append(tok.apply_chat_template([{"role":"system","content":instruction},{"role":"user","content":user}],
                                                    tokenize=False,add_generation_prompt=True))
        enc=tok(prompts,return_tensors="pt",padding=True).to(mdl.device); width=enc.input_ids.shape[1]
        with torch.inference_mode(): seq=mdl.generate(**enc,max_new_tokens=3,do_sample=False,pad_token_id=tok.pad_token_id)
        outs=[tok.decode(seq[i,width:],skip_special_tokens=True).strip() for i in range(len(batch))]
        for x,o in zip(batch,outs): judged[x["case_id"]]=o.upper().startswith("CORRECT")
        rawlog.append({"start":start,"model":model,"outputs":[{"case_id":x["case_id"],"output":o} for x,o in zip(batch,outs)]})
        print(f"judged {min(start+24,len(todo))}/{len(todo)}",flush=True)
    for r in rows:
        k=key(r)
        if k not in labels:
            i=seen[(r["question"],r["answer"],r["answer_text"])]
            labels[k]={"correct":judged[i],"source":"semantic_judge","case_id":i}
    out={"model":model,"instruction":instruction,"n_records":len(rows),"n_semantic_cases":len(todo),"labels":labels,"raw_batches":rawlog}
    Path("results/answer_judgments.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"records":len(rows),"semantic_cases":len(todo),"model":model}))
if __name__=="__main__":main()
