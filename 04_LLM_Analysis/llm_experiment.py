import os
import sys
import json
import time
import math
import pandas as pd
from pathlib import Path
from openai import OpenAI
from typing import Optional, Tuple

# ---------- 路径配置 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_ROOT / "01_Data" / "data_splits" / "test.csv"
OUT_DIR = PROJECT_ROOT / "04_LLM_Analysis" / "llm_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- DeepSeek 客户端 ----------
client = OpenAI(
    #api_key=os.environ.get("DEEPSEEK_API_KEY"),
    api_key="sk-07e5c38567bd41cb9e4cc4b8ab2ce066",
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"  # 性价比最高，分类任务足够

TASK2_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Declining to answer", "Claims ignorance", "Clarification",
    "Partial/half-answer",
]


def build_system_prompt() -> str:
    """系统指令：九类定义 + 强制 JSON。必须包含 'json' 字样。"""
    return """你是一个问答清晰度分类器。请从以下九个标签中选择唯一一个最符合的标签，并给出简短理由。
九类标签定义：
- Explicit: 直接、明确地给出了被问及的信息。
- Implicit: 间接地传达了被问及的信息，而非直接陈述。
- Dodging: 完全忽略了问题本身。
- General: 回答过于笼统，缺乏被问及的具体性。
- Deflection: 回答开头切题，但随后将焦点转移到了其他点。
- Declining to answer: 直接或间接地拒绝回答。
- Claims ignorance: 声称不知道被问及的信息。
- Clarification: 要求澄清问题，而非提供信息。
- Partial/half-answer: 只提供了被问及信息的一个具体组成部分。

请以 JSON 格式输出，不要包含任何其他文本或 Markdown 标记。格式必须为：
{"label": "<标签名>", "reason": "<一句话理由>"}"""


def build_user_prompt_only(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def build_user_prompt_assisted(question: str, answer: str,
                                ens_label: str, ens_conf: float) -> str:
    return (
        f"Question: {question}\nAnswer: {answer}\n"
        f"Ensemble prediction: {ens_label}\n"
        f"Ensemble confidence: {ens_conf:.3f}\n"
        f"请参考集成结果，但独立做出判断，并以 json 格式输出。"
    )


def call_deepseek(system_prompt: str, user_prompt: str,
                  max_retries: int = 3) -> Tuple[Optional[str], Optional[float], str]:
    """
    调用 DeepSeek API，返回 (label, confidence_prob, reason)。
    confidence_prob 从 label 对应 token 的 logprob 转换而来：exp(logprob)。
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                logprobs=True,
                top_logprobs=5,
                temperature=0.0,
            )

            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            label = parsed.get("label", "").strip()
            reason = parsed.get("reason", "").strip()

            if label not in TASK2_LABELS:
                return None, None, f"label_not_in_9: {label}"

            # ---------- 提取置信度 ----------
            confidence = None
            token_logprobs = resp.choices[0].logprobs
            if token_logprobs and token_logprobs.content:
                # 策略：找到内容包含 label 的 token 的 logprob
                for tok_info in token_logprobs.content:
                    if label in tok_info.token:
                        if tok_info.logprob > -9999:  # 过滤掉极低概率
                            confidence = math.exp(tok_info.logprob)
                        break
                # 兜底：如果没匹配到，用所有 token logprob 的几何平均
                if confidence is None:
                    valid_lps = [t.logprob for t in token_logprobs.content
                                 if t.logprob > -9999]
                    if valid_lps:
                        confidence = math.exp(sum(valid_lps) / len(valid_lps))

            return label, confidence, reason

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return None, None, f"json_decode_error: {str(e)[:100]}"
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                return None, None, f"api_error: {str(e)[:100]}"
            time.sleep(2 ** attempt)

    return None, None, "max_retries_exceeded"


def run_experiment(setting: str, n_samples: int = 308,
                   ensemble_file: Optional[Path] = None):
    """
    setting: 'llm_only' 或 'llm_assisted'
    ensemble_file: llm_assisted 模式下，C 的 soft_voting_probabilities.csv
    """
    df = pd.read_csv(SPLIT_FILE).head(n_samples).copy()

    # llm_assisted 需要集成预测和置信度
    if setting == "llm_assisted":
        if ensemble_file is None or not ensemble_file.exists():
            print(f"[错误] llm_assisted 需要 ensemble_file，当前不存在: {ensemble_file}")
            return
        ens_df = pd.read_csv(ensemble_file)
        prob_cols = [c for c in ens_df.columns if c.startswith("p_")]
        ens_df["ensemble_label"] = ens_df[prob_cols].idxmax(axis=1).str.replace("p_", "", regex=False)
        ens_df["ensemble_confidence"] = ens_df[prob_cols].max(axis=1)
        df = df.merge(
            ens_df[["sample_id", "ensemble_label", "ensemble_confidence"]],
            on="sample_id", how="left"
        )

    system_prompt = build_system_prompt()
    rows = []

    for i, r in df.iterrows():
        q = str(r.get("question", ""))
        a = str(r.get("interview_answer", ""))

        if setting == "llm_only":
            user_prompt = build_user_prompt_only(q, a)
        else:
            user_prompt = build_user_prompt_assisted(
                q, a,
                str(r.get("ensemble_label", "UNKNOWN")),
                float(r.get("ensemble_confidence", 0.0)),
            )

        t0 = time.time()
        label, conf, reason = call_deepseek(system_prompt, user_prompt)
        latency = time.time() - t0

        rows.append({
            "sample_id": r["sample_id"],
            "setting": setting,
            "model": MODEL,
            "gold_majority": r.get("evasion_label_majority", None),
            "annotator1": r.get("annotator1", None),
            "annotator2": r.get("annotator2", None),
            "annotator3": r.get("annotator3", None),
            "pred_label": label,
            "confidence": conf,
            "reason": reason,
            "parse_ok": int(label is not None),
            "latency_sec": round(latency, 3),
        })

        if (i + 1) % 20 == 0:
            ok = sum(x["parse_ok"] for x in rows) / len(rows)
            print(f"[{setting}] {i+1}/{len(df)} | parse_ok={ok:.1%} | last_latency={latency:.2f}s")

    out = OUT_DIR / f"llm_results_{setting}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[完成] 写入: {out}")

def llm_results_to_prob_csv(setting: str):
    src = OUT_DIR / f"llm_results_{setting}.csv"
    df = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        sid = r["sample_id"]
        # 用均匀分布作为兜底，防止模型解析失败导致没有概率
        prob = {lab: 1.0 / 9 for lab in TASK2_LABELS}
        
        if r["parse_ok"] == 1 and pd.notna(r["pred_label"]):
            # 把预测标签的置信度调高，其余按均匀分布分配
            # (真实的 logprobs 转换比较复杂，这里为了满足九类加和为1，用置信度填充)
            conf = r["confidence"] if pd.notna(r["confidence"]) else 0.9
            rem = (1.0 - conf) / 8
            for lab in TASK2_LABELS:
                prob[lab] = conf if lab == r["pred_label"] else rem
                
        row = {"sample_id": sid}
        for lab in TASK2_LABELS:
            col = f"p_{lab.replace(' ', '_').replace('/', '_').replace('-', '_')}"
            row[col] = prob[lab]
        rows.append(row)

    out = PROJECT_ROOT / "04_LLM_Analysis" / "inputs" / f"llm_only_task2_probabilities.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[转换完成] 写入: {out}")

if __name__ == "__main__":
    # run_experiment("llm_only", n_samples=30)

    #print("=== 全量：LLM-only 308 条 ===")
    #run_experiment("llm_only", n_samples=308)

    llm_results_to_prob_csv("llm_only")
    # LLM-assisted 需要先有 C 的 soft_voting_probabilities.csv
    # soft_voting_file = PROJECT_ROOT / "04_LLM_Analysis" / "inputs" / "soft_voting_probabilities.csv"
    # print("=== LLM-assisted 30 条 ===")
    # run_experiment("llm_assisted", n_samples=30, ensemble_file=soft_voting_file)