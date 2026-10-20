# docs/eval/eval_hybrid.py
"""混合检索：RRF 融合 + 加权融合，多组权重对比；与 BM25/向量同口径
W4D5 扩展：回答质量轴 = 生成答案 -> LLM 逐要点判覆盖 -> 归因 + 人工抽检导出
"""
import json
import os
import random
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# ---- 配置 ----
TOP_K = 5
CAND_K = 30
RRF_K = 60
COVER_THRESHOLD = 0.5
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
JUDGE_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SAMPLE_N = 5

CHUNKS_PATH = Path(__file__).parent.parent.parent / "docs" / "chunks" / "chunks_fixed.jsonl"
QUESTIONS_PATH = Path(__file__).parent / "questions.all2.jsonl"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "ls_rag_docs_clean_200"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"

STOP = {"的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们",
        "他们", "啊", "吗", "呢", "吧", "不", "有", "这", "那", "什么", "怎么",
        "一个", "一下", "要", "会", "可以", "能", "也", "都", "就", "又", "还",
        "很", "太", "才", "已", "已经", "将", "被", "对", "关于", "等", "之",
        "与", "及", "或", "并", "而", "各", "其", "该", "本", "此", "以下", "以上"}


def load_json(p):
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def load_questions(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[跳过] 第 {i} 行解析失败: {e}")
                continue
            if isinstance(obj, dict) and "question" in obj:
                out.append(obj)
    return out


def norm_src(s):
    s = str(s or "").strip().replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    if "/" in s or ":" in s:
        s = Path(s).name
    return s.lower()


def _get_kp(q):
    for k in ("expected_keypoints", "keypoints", "要点", "answer_points", "points"):
        v = q.get(k)
        if v:
            if isinstance(v, str):
                return [x.strip() for x in v.replace("；", "\n").replace(";", "\n").split("\n") if x.strip()]
            return [str(x).strip() for x in v if str(x).strip()]
    return []


def keypoint_coverage(hit_texts, keys):
    if not keys:
        return 0.0, 0
    joined = "".join(hit_texts)
    hit = 0.0
    for k in keys:
        raw = str(k).strip()
        if not raw:
            continue
        if len(raw) <= 4:
            hit += 1.0 if raw in joined else 0.0
        else:
            toks = [t for t in jieba.lcut(raw) if len(t) > 1 and t not in STOP]
            hit += (sum(1 for t in toks if t in joined) / len(toks)) if toks else (1.0 if raw in joined else 0.0)
    return hit / len(keys), len(keys)


def parse_sources(q):
    v = q.get("expected_source")
    if not v:
        return None
    if isinstance(v, str) and any(w in v for w in ["—", "拒答", "超出覆盖范围"]):
        return None
    if isinstance(v, list):
        vals = [str(x).strip() for x in v if str(x).strip()]
    else:
        vals = [x.strip() for x in str(v).split(",") if x.strip()]
    result = {norm_src(x) for x in vals if x}
    return result if result else None


# ---- W4D5: 生成答案 ----
def answer_one(question, hits):
    ctx = "\n\n".join(h.get("text", "") for h in hits[:TOP_K])
    prompt = f"""你是专升本政策问答助手。请根据以下材料回答问题，材料不足时明确说明。

材料：
{ctx}

问题：{question}

回答："""
    try:
        r = requests.post(
            JUDGE_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[生成失败: {e}]"


# ---- W4D5: LLM 逐要点判定覆盖 ----
def judge_coverage(question, keypoints, answer):
    if not keypoints:
        return []
    kp_lines = "\n".join(f"- {kp}" for kp in keypoints)
    prompt = f"""你是评测员。请逐条判断以下答案是否覆盖了每个要点。

问题：{question}
期望要点：
{kp_lines}
系统答案：
{answer}

请按以下 JSON 数组格式输出（只输出数组，不要其他文字）：
[
  {{"keypoint": "要点原文", "covered": true/false, "reason": "简短说明"}}
]
"""
    try:
        r = requests.post(
            JUDGE_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[判定失败: {e}]"


# ---- W4D5: 归因 ----
def attribute(keypoint, hits, answer):
    joined = "".join(h.get("text", "") for h in hits)
    raw = str(keypoint).strip()
    in_recall = len(raw) <= 4 and raw in joined or (
        len(raw) > 4 and any(t in joined for t in jieba.lcut(raw) if len(t) > 1 and t not in STOP)
    )
    in_answer = raw in answer
    if not in_recall:
        return "miss_recall"
    if in_recall and not in_answer:
        return "answer_drift"
    return "prompt_miss"


def judge_one(q, hits, exp):
    labeled = exp is not None
    got = {norm_src(h.get("source", "")) for h in hits}
    n_src = len((exp or set()) & got)
    keys = _get_kp(q)
    cov, total = keypoint_coverage([h.get("text", "") for h in hits], keys)
    if q.get("type") == "no_answer":
        v = "拒答"
    elif not labeled:
        v = "未标注"
    elif total == 0:
        v = "对" if n_src > 0 else "错"
    elif n_src > 0 and cov >= COVER_THRESHOLD:
        v = "对"
    elif n_src > 0 or cov > 0:
        v = "偏"
    else:
        v = "错"
    rec = 1.0 if (n_src > 0 and cov >= COVER_THRESHOLD) else (0.5 if (n_src > 0 or cov > 0) else 0.0)
    return dict(v=v, labeled=labeled, n_src=n_src, cov=cov, total=total, rec=rec, got=got)


def embed_one(text):
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": [text]},
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def run_bm25(chunks, bm, qs, k):
    out = []
    for q in qs:
        scores = bm.get_scores(list(jieba.cut_for_search(q["question"])))
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out.append([
            {
                "text": chunks[i]["text"],
                "source": chunks[i].get("source", chunks[i].get("file", "")),
                "page": chunks[i].get("page", None),
            }
            for i in idxs
        ])
    return out


def run_vec(qs, client, k):
    out = []
    for q in qs:
        qv = embed_one(q["question"])
        res = client.query_points(
            collection_name=COLLECTION, query=qv, limit=k,
            with_payload=["text", "source", "page"],
        ).points
        out.append([{"text": h.payload.get("text", ""),
                     "source": h.payload.get("source", ""),
                     "page": h.payload.get("page", "")} for h in res])
    return out


def _cid(c):
    return c.get("source", "") + "#" + str(c.get("page", ""))


def fuse_rrf(lists, k=TOP_K, rrf_k=RRF_K):
    score, order = {}, []
    for lst in lists:
        for rank, c in enumerate(lst):
            cid = _cid(c)
            if cid not in score:
                score[cid] = 0.0
                order.append((cid, c))
            score[cid] += 1.0 / (rrf_k + rank + 1)
    order.sort(key=lambda x: -score[x[0]])
    return [c for _, c in order[:k]]


def fuse_weighted(lists, weights, k=TOP_K):
    score, order = {}, []
    for lst, w in zip(lists, weights):
        if not lst:
            continue
        for rank, c in enumerate(lst):
            cid = _cid(c)
            if cid not in score:
                score[cid] = 0.0
                order.append((cid, c))
            score[cid] += w / (rank + 1)
    order.sort(key=lambda x: -score[x[0]])
    return [c for _, c in order[:k]]


def report(name, qs, hits_list):
    rows = [judge_one(q, hits, parse_sources(q)) for q, hits in zip(qs, hits_list)]
    lab = [r for r in rows if r["labeled"]]
    denom = max(len(lab), 1)
    num = sum(r["rec"] for r in lab)
    cnt = lambda vv: sum(1 for r in rows if r["v"] == vv)
    print(f"{name:24s} Recall@{TOP_K}={num/denom*100:5.1f}% | "
          f"对{cnt('对')} 偏{cnt('偏')} 错{cnt('错')} 未标注{cnt('未标注')} | "
          f"文件级{sum(1 for r in rows if r['n_src']>0)}/{len(rows)} "
          f"内容级{sum(1 for r in rows if r['cov']>0)}/{len(rows)} "
          f"(分母={denom})")
    return rows


# ---- W4D5: 回答质量轴评测 ----
def eval_answer_quality(qs, hits_list, config_name):
    print(f"\n=== {config_name} 回答质量轴 ===")
    results = []
    for qi, (q, hits) in enumerate(zip(qs, hits_list)):
        keys = _get_kp(q)
        if not keys:
            continue
        # hits_list 里每个元素是 dict，取 "hits" 键
        if isinstance(hits, dict):
            hits = hits.get("hits", [])
        ans = answer_one(q["question"], hits)
        raw = judge_coverage(q["question"], keys, ans)
        try:
            judgments = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            judgments = [{"keypoint": k, "covered": False, "reason": raw} for k in keys]
        covered = sum(1 for j in judgments if j.get("covered"))
        rate = covered / len(keys) if keys else 0
        attrs = [attribute(j["keypoint"], hits, ans) for j in judgments if not j.get("covered")]
        results.append(dict(
            qid=qi + 1, question=q["question"], answer=ans,
            total=len(keys), covered=covered, rate=rate,
            judgments=judgments, attrs=attrs,
        ))
        tag = "OK" if rate >= 1.0 else ("PARTIAL" if rate > 0 else "MISS")
        print(f"  Q{qi+1} [{tag}] 覆盖 {covered}/{len(keys)} ({rate*100:.0f}%) 归因 {attrs}")
    return results


def export_samples(all_results, out_path):
    flat = []
    for r in all_results:
        for j in r["judgments"]:
            flat.append(dict(
                qid=r["qid"], question=r["question"],
                keypoint=j["keypoint"], covered=j["covered"],
                reason=j.get("reason", ""), answer=r["answer"],
                attrs=r["attrs"],
            ))
    random.seed(42)
    sample = random.sample(flat, min(SAMPLE_N, len(flat)))
    out_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[抽检] 已导出 {len(sample)} 条样本 -> {out_path}")
    for s in sample:
        print(f"  Q{s['qid']} | 要点: {s['keypoint']} | 覆盖: {s['covered']} | 归因: {s['attrs']}")


def main():
    chunks = [c for c in load_json(CHUNKS_PATH) if c.get("text")]
    qs = load_questions(QUESTIONS_PATH)

    print(f"[自检] 题目字段: {list(qs[0].keys())}")
    print(f"[自检] expected_keypoints 样例: {qs[0].get('expected_keypoints')}")
    print(f"[自检] 参与计分题数: {sum(1 for q in qs if parse_sources(q) is not None)}")
    print(f"chunk={len(chunks)} 题目={len(qs)} CAND={CAND_K} TOP={TOP_K}\n")
    print("\n=== 被排除的 5 题 ===")
    for i, q in enumerate(qs):
        if parse_sources(q) is None:
            print(f"  Q{i+1}: {q['question'][:50]}  | expected_source={q.get('expected_source')}")

    toks = [list(jieba.cut_for_search(c["text"])) for c in chunks]
    bm = BM25Okapi(toks)
    client = QdrantClient(url=QDRANT_URL)

    bc = run_bm25(chunks, bm, qs, CAND_K)
    vc = run_vec(qs, client, CAND_K)

    baselines = {
        "BM25": [lst[:TOP_K] for lst in bc],
        "向量": [lst[:TOP_K] for lst in vc],
    }
    configs = [
        ("RRF",                lambda: [fuse_rrf([bc[i], vc[i]]) for i in range(len(qs))]),
        ("w(BM25=0.7,V=0.3)",  lambda: [fuse_weighted([bc[i], vc[i]], [0.7, 0.3]) for i in range(len(qs))]),
        ("w(BM25=0.5,V=0.5)",  lambda: [fuse_weighted([bc[i], vc[i]], [0.5, 0.5]) for i in range(len(qs))]),
        ("w(BM25=0.3,V=0.7)",  lambda: [fuse_weighted([bc[i], vc[i]], [0.3, 0.7]) for i in range(len(qs))]),
    ]
    fused = {name: fn() for name, fn in configs}

    print("=== 基线 ===")
    rb = report("BM25", qs, baselines["BM25"])
    rv = report("向量", qs, baselines["向量"])
    print("\n=== 混合 ===")
    rf = {name: report(name, qs, fused[name]) for name, _ in configs}

    print("\n=== 哪些题被混合救回来了 ===")
    base_both = [(rb[i]["rec"], rv[i]["rec"]) for i in range(len(qs))]
    for name, rows in rf.items():
        saved = []
        for i, (b, v) in enumerate(base_both):
            if rows[i]["rec"] > max(b, v):
                saved.append(f"Q{i+1}({b:.1f},{v:.1f}->{rows[i]['rec']:.1f}) {qs[i]['question'][:26]}")
        print(f"{name:24s} {len(saved)} 题 -> {saved if saved else '无'}")

    best = max(rf.items(), key=lambda kv: sum(r["rec"] for r in kv[1] if r["labeled"]))
    denom = sum(1 for r in rb if r["labeled"])
    print(f"\n最优配置: {best[0]} "
          f"(Recall@{TOP_K}={sum(r['rec'] for r in best[1] if r['labeled'])/max(denom,1)*100:.1f}%, 分母={denom})")
    print("注：权重在测试集上选取，存在选择偏差，结论仅描述性。")

    # ---- W4D5 回答质量轴（仅对最优配置跑，省时间） ----
    best_name, best_hits = best
    qa_results = eval_answer_quality(qs, best_hits, best_name)
    if qa_results:
        total_pts = sum(r["total"] for r in qa_results)
        total_cov = sum(r["covered"] for r in qa_results)
        print(f"\n要点覆盖率: {total_cov}/{total_pts} = {total_cov/total_pts*100:.1f}%")
        all_attrs = []
        for r in qa_results:
            all_attrs.extend(r["attrs"])
        from collections import Counter
        attr_dist = Counter(all_attrs)
        print(f"归因分布: {dict(attr_dist)}")
        out = Path(__file__).parent / f"samples_{best_name.replace('(', '').replace(')', '').replace(',', '').replace('=', '_')}.json"
        export_samples(qa_results, out)

    print("\n卡点记录: 检索命中却答错的题，主要卡在 answer_drift（召回到了但模型答偏）或 prompt_miss（信息在但没组织好）。")


if __name__ == "__main__":
    main()