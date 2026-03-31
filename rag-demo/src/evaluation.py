"""
Pipeline đánh giá chatbot pháp luật Lex
  Tầng 1 [RAGAS]        → context_precision, context_recall  (retriever)
  Tầng 2 [LLM-as-Judge] → accuracy, persona                  (generator)

Yêu cầu: pip install openai pymupdf python-docx ragas datasets
"""

import os, re, json, math, random
import fitz, docx
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
SEP = "─" * 55


# ════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════

def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _score_label(score) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "❌ Lỗi"
    if score >= 0.8: return "✅ Tốt"
    if score >= 0.6: return "⚠️  Trung bình"
    return "❌ Cần cải thiện"


def _print_scores(title: str, scores: dict[str, float]) -> None:
    print(f"\n{'═'*55}\n  {title}\n{'═'*55}")
    print(f"  {'Metric':<25} {'Điểm':>6}  Đánh giá")
    print(f"  {SEP}")
    for metric, score in scores.items():
        val = f"{score:>6.3f}" if isinstance(score, float) and not math.isnan(score) else "   N/A"
        print(f"  {metric:<25} {val}  {_score_label(score)}")
    print("═" * 55)


def _load_pairs(dataset_path: str) -> tuple[list[dict], str]:
    """Load qa_pairs từ flat hoặc merged JSON. Trả về (pairs, label)."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "datasets" in data:                          # merged format
        pairs = []
        for sub in data["datasets"]:
            for qa in sub.get("qa_pairs", []):
                qa.setdefault("law_name", sub.get("law_name", ""))
                pairs.append(qa)
        label = f"{data.get('total_laws', len(data['datasets']))} bộ luật"
    else:                                           # flat format
        pairs = data["qa_pairs"]
        label = data.get("law_name", "N/A")

    return pairs, label


def _call_llm(messages: list[dict], temperature: float = 0) -> str:
    """Gọi gpt-4o-mini và trả về text content."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict | None:
    """Strip markdown fences và parse JSON. Trả về None nếu lỗi."""
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error: {e}")
        return None


# ════════════════════════════════════════════════════════
# PHẦN 1: ĐỌC FILE & CHIA CHUNK
# ════════════════════════════════════════════════════════

def read_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return "".join(p.get_text() for p in fitz.open(file_path))
    if file_path.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(file_path).paragraphs if p.text.strip())
    raise ValueError(f"Không hỗ trợ định dạng: {file_path}")


def split_by_article(text: str, max_chunk_size: int = 3000) -> list[str]:
    """Cắt văn bản theo Điều. Fallback theo ký tự nếu không tìm được Điều."""
    matches = list(re.compile(r"\nĐiều\s+\d+\.\s", re.UNICODE).finditer(text))

    if not matches:                                 # fallback: cắt theo ký tự
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i + max_chunk_size].strip())
            i += max_chunk_size
        return [c for c in chunks if c]

    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[m.start():end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ════════════════════════════════════════════════════════
# PHẦN 2: SINH DATASET
# ════════════════════════════════════════════════════════

_QA_SYSTEM = """Bạn là chuyên gia pháp luật Việt Nam.
Sinh câu hỏi - đáp án loại trực tiếp (direct) từ đoạn văn bản pháp luật.

Yêu cầu:
- Chỉ dùng thông tin trong đoạn văn, không suy luận thêm
- Câu hỏi hỏi thẳng 1 thông tin cụ thể, đáp án rõ ràng
- Ghi rõ điều khoản nguồn (VD: Điều 5, Khoản 2)
- Trả về JSON duy nhất, không thêm text:
{"qa_pairs": [{"question": "...", "ground_truth": "...", "source_article": "Điều X"}]}"""

_QA_USER = "Văn bản từ {law_name}:\n\n{chunk}\n\nSinh tối đa {n} câu hỏi direct."


def _generate_qa_for_chunk(chunk: str, law_name: str, n: int) -> list[dict]:
    if n <= 0:
        return []
    try:
        raw = _call_llm([
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user",   "content": _QA_USER.format(law_name=law_name, chunk=chunk, n=n)},
        ], temperature=0.3)
        data = _parse_json(raw)
        pairs = data.get("qa_pairs", []) if data else []
        for p in pairs:
            p["type"] = "direct"
        return pairs
    except Exception as e:
        print(f"  ⚠️  API error: {e}")
        return []


def generate_dataset(
    file_path: str,
    law_name: str,
    output_path: str,
    total_questions: int = 30,
    max_chunks: int | None = None,
    seed: int | None = None,
) -> dict:
    """Sinh dataset câu hỏi. Bỏ qua nếu file đã tồn tại."""
    if os.path.exists(output_path):
        print(f"\n✅ Dataset đã có: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    text   = read_file(file_path)
    chunks = split_by_article(text)
    if seed is not None:
        random.seed(seed)
    random.shuffle(chunks)
    if max_chunks:
        chunks = chunks[:max_chunks]

    q_per_chunk = max(1, round(total_questions / len(chunks)))
    all_qa, remaining, current_id = [], total_questions, 1

    print(f"\n📄 {law_name} — {len(chunks)} Điều, mục tiêu {total_questions} câu")
    for i, chunk in enumerate(chunks):
        if remaining <= 0:
            break
        pairs = _generate_qa_for_chunk(chunk, law_name, min(q_per_chunk, remaining))[:remaining]
        for qa in pairs:
            qa["id"] = current_id
            current_id += 1
        all_qa.extend(pairs)
        remaining -= len(pairs)
        print(f"  Điều {i+1}/{len(chunks)} → +{len(pairs)} câu (tổng {len(all_qa)})")

    dataset = {"law_name": law_name, "file_path": file_path,
               "total_questions": len(all_qa), "qa_pairs": all_qa}
    _save_json(output_path, dataset)
    print(f"\n✅ {len(all_qa)} câu → {output_path}")
    return dataset


# ════════════════════════════════════════════════════════
# PHẦN 3: LLM-AS-A-JUDGE
# ════════════════════════════════════════════════════════

_ACCURACY_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam kiểm tra độ chính xác chatbot.

Câu hỏi       : {question}
Đáp án chuẩn  : {ground_truth}
Câu trả lời   : {answer}

Tách đáp án chuẩn thành các facts nhỏ. Kiểm tra từng fact có trong câu trả lời không.
Bỏ qua phần chào hỏi (⚖️), Lời nhắc Lex, Khuyến nghị.

score = facts đúng / tổng facts | PASS ≥ 0.8 | PARTIAL ≥ 0.5 | FAIL < 0.5

Trả về JSON duy nhất:
{{
  "facts": [{{ "fact": "...", "covered": bool, "correct": bool, "note": "..." }}],
  "score": 0.0-1.0, 
  "verdict": "PASS/PARTIAL/FAIL", 
  "summary": "..."
}}"""

_PERSONA_PROMPT = """Bạn là giám khảo đánh giá chatbot pháp luật "Lex — Đại sứ Pháp lý Số".

Câu trả lời: {answer}

Chấm 4 tiêu chí:
1. signature : Bắt đầu ⚖️ VÀ kết thúc "Lời nhắc Lex"?
2. tone      : Điềm tĩnh, nghiêm minh? Xưng "Tôi", gọi "Quý khách"?
3. structure : Có đủ Căn cứ pháp lý / Phân tích / Khuyến nghị?
4. citation  : Đúng định dạng "Tên văn bản — Điều X, Khoản Y"?

score = tiêu chí pass / 4 | PASS ≥ 0.75 | PARTIAL ≥ 0.5 | FAIL < 0.5

Trả về JSON duy nhất:
{{
  "signature": {{ "pass": bool, "note": "..." }}, 
  "tone": {{ "pass": bool, "note": "..." }},
  "structure": {{ "pass": bool, "note": "..." }}, 
  "citation": {{ "pass": bool, "note": "..." }},
  "score": 0.0-1.0, 
  "verdict": "PASS/PARTIAL/FAIL"
}}"""


def _judge(prompt: str) -> dict:
    """Gọi judge, parse JSON. Trả về dict với verdict=ERROR nếu thất bại."""
    try:
        raw  = _call_llm([{"role": "user", "content": prompt}])
        data = _parse_json(raw)
        return data if data else {"score": None, "verdict": "ERROR"}
    except Exception as e:
        print(f"  ⚠️  Judge error: {e}")
        return {"score": None, "verdict": "ERROR"}


def run_llm_judge(
    dataset_path: str,
    chatbot_fn,
    output_path: str = "judge_result.json",
) -> dict | None:
    """Chạy LLM-as-Judge: accuracy (fact-level) + persona (4 tiêu chí)."""
    pairs, label = _load_pairs(dataset_path)
    print(f"\n⚖️  LLM-as-Judge — {len(pairs)} câu | {label}\n{SEP}")

    results, acc_scores, per_scores = [], [], []
    verdicts = {"accuracy": {}, "persona": {}}      # count PASS/PARTIAL/FAIL/ERROR

    for i, qa in enumerate(pairs):
        print(f"  [{i+1:>2}/{len(pairs)}] {qa['question'][:50]}...", end=" ")

        try:
            answer, _ = chatbot_fn(qa["question"])
        except Exception as e:
            print(f"chatbot lỗi: {e}")
            results.append({**qa, "answer": "", "accuracy": {"verdict": "ERROR"},
                             "persona": {"verdict": "ERROR"}})
            continue

        acc = _judge(_ACCURACY_PROMPT.format(
            question=qa["question"], ground_truth=qa["ground_truth"], answer=answer))
        per = _judge(_PERSONA_PROMPT.format(answer=answer))

        if acc.get("score") is not None: acc_scores.append(acc["score"])
        if per.get("score") is not None: per_scores.append(per["score"])

        for key, result in [("accuracy", acc), ("persona", per)]:
            v = result.get("verdict", "ERROR")
            verdicts[key][v] = verdicts[key].get(v, 0) + 1

        results.append({**qa, "answer": answer, "accuracy": acc, "persona": per})
        print(f"acc={acc.get('score', 'ERR'):.2f}[{acc.get('verdict')}]  "
              f"persona={per.get('score', 'ERR'):.2f}[{per.get('verdict')}]"
              if acc.get("score") is not None else
              f"acc=[{acc.get('verdict')}]  persona=[{per.get('verdict')}]")

    avg_acc = sum(acc_scores) / len(acc_scores) if acc_scores else None
    avg_per = sum(per_scores) / len(per_scores) if per_scores else None

    _print_scores("KẾT QUẢ LLM-AS-JUDGE", {"accuracy": avg_acc, "persona": avg_per})
    for key in ["accuracy", "persona"]:
        print(f"  {key:10}: {verdicts[key]}")

    # Top 5 câu accuracy thấp nhất
    worst = sorted([r for r in results if r["accuracy"].get("score") is not None],
                   key=lambda r: r["accuracy"]["score"])[:5]
    if worst:
        print("\n  Top 5 accuracy thấp:")
        for r in worst:
            print(f"    [{r['accuracy']['score']:.2f}] {r['question'][:55]}")

    overall = {"accuracy": round(avg_acc, 3) if avg_acc is not None else None,
               "persona":  round(avg_per, 3)  if avg_per  is not None else None,
               "verdicts": verdicts, "total": len(pairs)}
    output = {"overall": overall, "results": results}
    _save_json(output_path, output)
    _save_json(output_path.replace(".json", "_summary.json"), overall)
    print(f"\n  Chi tiết → {output_path}")
    return output


# ════════════════════════════════════════════════════════
# PHẦN 4: RAGAS — RETRIEVER
# ════════════════════════════════════════════════════════

def run_ragas_evaluation(
    dataset_path: str,
    chatbot_fn,
    output_path: str = "ragas_result.csv",
) -> dict | None:
    """Chạy RAGAS: context_precision + context_recall (retriever only)."""
    try:
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import context_precision, context_recall
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings as RagasEmbeddings
        from openai import AsyncOpenAI
        from datasets import Dataset
    except ImportError:
        print("Thiếu thư viện: pip install ragas datasets")
        return None

    pairs, label = _load_pairs(dataset_path)
    print(f"\n🔍 RAGAS Retriever — {len(pairs)} câu | {label}\n{SEP}")

    records, errors = [], 0
    for i, qa in enumerate(pairs):
        print(f"  [{i+1:>2}/{len(pairs)}] {qa['question'][:50]}...", end=" ")
        try:
            answer, contexts = chatbot_fn(qa["question"])
            if not contexts:
                print("contexts rỗng, bỏ qua"); errors += 1; continue
            records.append({"question": qa["question"], "answer": answer,
                            "contexts": contexts, "ground_truth": qa["ground_truth"],
                            "type": qa.get("type", ""), "id": qa.get("id")})
            print("✅")
        except Exception as e:
            print(f"Lỗi: {e}"); errors += 1

    if not records:
        print(f"Không có câu nào thành công ({errors} lỗi)."); return None
    if errors:
        print(f"  {errors} câu lỗi, đánh giá {len(records)} câu còn lại.")

    judge_llm    = llm_factory("gpt-4o-mini", client=client, n=1)
    vi_embeddings = RagasEmbeddings(
        client=AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        model="text-embedding-3-small",
    )

    print("\n  Đang chạy RAGAS...")
    ragas_keys = ["question", "answer", "contexts", "ground_truth"]
    df = evaluate(
        dataset=Dataset.from_list([{k: r[k] for k in ragas_keys} for r in records]),
        metrics=[context_precision, context_recall],
        llm=judge_llm,
        embeddings=vi_embeddings,
        run_config=RunConfig(timeout=180, max_retries=5),
    ).to_pandas()

    df["type"] = [r["type"] for r in records[:len(df)]]
    df["id"]   = [r["id"]   for r in records[:len(df)]]

    overall = {col: round(df[col].mean(), 3) for col in ["context_precision", "context_recall"]}
    _print_scores("KẾT QUẢ RAGAS — RETRIEVER", overall)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    result = {"overall": overall, "detail_csv": output_path}
    _save_json(output_path.replace(".csv", "_summary.json"), result)
    print(f"\n  Chi tiết → {output_path}")
    return result


# ════════════════════════════════════════════════════════
# PHẦN 5: PIPELINE TỔNG HỢP
# ════════════════════════════════════════════════════════

def run_full_evaluation(
    dataset_path: str,
    chatbot_fn,
    output_dir: str = "eval_output",
) -> dict:
    """Chạy cả 2 tầng và in báo cáo tổng hợp."""
    os.makedirs(output_dir, exist_ok=True)

    ragas_result = run_ragas_evaluation(
        dataset_path, chatbot_fn,
        output_path=os.path.join(output_dir, "ragas_result.csv"),
    )
    judge_result = run_llm_judge(
        dataset_path, chatbot_fn,
        output_path=os.path.join(output_dir, "judge_result.json"),
    )

    # Báo cáo tổng hợp
    print(f"\n{'█'*55}\n  TỔNG HỢP\n{'█'*55}")
    print(f"  {'Tầng':<12} {'Metric':<25} {'Điểm':>6}")
    print(f"  {SEP}")
    if ragas_result:
        for k, v in ragas_result["overall"].items():
            print(f"  {'[Retriever]':<12} {k:<25} {v:>6.3f}")
    if judge_result:
        for k in ["accuracy", "persona"]:
            v = judge_result["overall"].get(k)
            print(f"  {'[Generator]':<12} {k:<25} {f'{v:>6.3f}' if v else '   N/A'}")
    print("█" * 55)

    combined = {"ragas": ragas_result["overall"] if ragas_result else None,
                "judge": judge_result["overall"]  if judge_result  else None}
    path = os.path.join(output_dir, "evaluation_combined.json")
    _save_json(path, combined)
    print(f"\n  Kết quả tổng hợp → {path}\n")
    return combined

import pandas as pd

def rerun_judge_from_ragas_file(ragas_csv_path: str, output_path: str = "eval_output/judge_result_fixed.json"):
    # Đọc file CSV
    df = pd.read_csv(ragas_csv_path)
    print(f"🚀 Re-running Judge cho {len(df)} câu hỏi từ {ragas_csv_path}...")
    
    results, acc_scores, per_scores = [], [], []
    verdicts = {"accuracy": {}, "persona": {}}

    for i, row in df.iterrows():
        # Mapping chính xác theo header file CSV của bạn
        q = row['user_input']
        a = row['response']
        gt = row['reference']
        
        print(f"  [{i+1}] Đang chấm: {str(q)[:50]}...", end=" ")

        # Gọi judge (Đảm bảo Prompt đã được sửa dấu {{ }} ở bước trước)
        acc = _judge(_ACCURACY_PROMPT.format(question=q, ground_truth=gt, answer=a))
        per = _judge(_PERSONA_PROMPT.format(answer=a))

        if acc.get("score") is not None: acc_scores.append(acc["score"])
        if per.get("score") is not None: per_scores.append(per["score"])

        for key, result in [("accuracy", acc), ("persona", per)]:
            v = result.get("verdict", "ERROR")
            verdicts[key][v] = verdicts[key].get(v, 0) + 1

        results.append({
            "id": row.get('id'),
            "question": q, 
            "ground_truth": gt, 
            "answer": a, 
            "accuracy": acc, 
            "persona": per
        })
        print(f"Done! (Acc: {acc.get('score', 'ERR')})")

    avg_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0
    avg_per = sum(per_scores) / len(per_scores) if per_scores else 0

    overall = {
        "accuracy": round(avg_acc, 3), 
        "persona": round(avg_per, 3), 
        "verdicts": verdicts,
        "total": len(df)
    }
    
    _save_json(output_path, {"overall": overall, "results": results})
    print(f"\n✅ Hoàn tất! Kết quả tổng hợp: {output_path}")
    _print_scores("KẾT QUẢ TỔNG HỢP SAU KHI FIX", {"accuracy": avg_acc, "persona": avg_per})




if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    sys.path.insert(0, PROJECT_SRC)

    from generator.llm_generator import LLMGenerator
    from retrieval.retriever import Retriever
    from retrieval.reranker import Reranker
    from embedding.embedding import EmbeddingService
    from embedding.bm25_en import BM25Encoder
    from ingestion.qdrant_store import QdrantVectorStore

    # ── Khởi tạo các service ─────────────────────────────
    BM25_PATH = os.getenv("BM25_VOCAB_PATH",
                          os.path.join(PROJECT_SRC, "data", "bm25_vocab.json"))

    embedding_service = EmbeddingService()
    bm25_encoder      = BM25Encoder(vocab_path=BM25_PATH)
    bm25_encoder.load(BM25_PATH)

    vector_store = QdrantVectorStore(
        collection_name=os.getenv("QDRANT_COLLECTION", "legal_documents"),
        dimension=embedding_service.dimension,
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    retriever = Retriever(
        embedding_service=embedding_service, bm25_encoder=bm25_encoder,
        vector_store=vector_store, reranker=Reranker(),
        initial_top_k=40, final_top_n=5, use_reranker=True,
    )
    generator = LLMGenerator(
        simple_model=os.getenv("SIMPLE_MODEL", "gpt-4o-mini"),
        reasoning_model=os.getenv("REASONING_MODEL", "gpt-4o-mini"),
    )

    # ── Chatbot function ──────────────────────────────────
    def my_chatbot(question: str) -> tuple[str, list[str]]:
        chunks  = retriever.retrieve(question)
        answer  = generator.generate(query=question, context_chunks=chunks)
        contexts = [c.get("parent_content") or c.get("text", "") for c in chunks]
        return answer, contexts

    # ── Sinh dataset (bỏ qua nếu đã có) ─────────────────
    generate_dataset(
        file_path=("/home/phattl/code/fsds_final_proj/aide4/"
                   "Legal-AI-Lex-Assistant/rag-demo/documents/Luat_Honnhan_Giadinh.pdf"),
        law_name="Luật Hôn Nhân và Gia Đình 2014",
        output_path="luat_merged_dataset.json",
        total_questions=5,
    )

    # ── Chạy toàn bộ evaluation ───────────────────────────
    rerun_judge_from_ragas_file(
        ragas_csv_path="eval_output/ragas_result.csv", # Đường dẫn file csv RAGAS đã sinh ra
        output_path="eval_output/judge_result_fixed.json"
    )

