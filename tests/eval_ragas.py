# tests/eval_ragas.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS RAGAS?
#   RAGAS = Retrieval Augmented Generation Assessment
#   A framework that measures HOW GOOD your LLM/agent answers are.
#   It gives you numbers (0.0 → 1.0) for specific quality dimensions.
#   Without RAGAS: you test your app by "reading the output and guessing."
#   With RAGAS: you get reproducible scores you can track over time.
#
# WHY DO WE NEED IT?
#   Your travel planner has THREE steps that can each fail silently:
#     1. Orchestrator: did it route to the right agent?
#     2. Researcher:   did it find accurate web data?
#     3. Planner:      did it use that data correctly in the final answer?
#   RAGAS measures step 3 (answer quality) and indirectly step 2 (context usage).
#   If your score drops after a code change → you broke something.
#
# THE THREE METRICS WE USE:
#
#   ┌─────────────────────┬──────────────────────────────────────────────────┐
#   │ Metric              │ What it measures                                 │
#   ├─────────────────────┼──────────────────────────────────────────────────┤
#   │ Faithfulness        │ Is the answer grounded in the retrieved context? │
#   │                     │ High = answer uses research_findings, not LLM    │
#   │                     │ imagination. Low = hallucination.                │
#   ├─────────────────────┼──────────────────────────────────────────────────┤
#   │ Answer Relevancy    │ Does the answer actually address the question?   │
#   │                     │ High = focused on the travel query.              │
#   │                     │ Low = tangential, verbose, off-topic.            │
#   ├─────────────────────┼──────────────────────────────────────────────────┤
#   │ Context Precision   │ Was the retrieved context (research_findings)    │
#   │                     │ relevant to answering the question?              │
#   │                     │ High = researcher found useful data.             │
#   │                     │ Low = researcher fetched irrelevant results.     │
#   └─────────────────────┴──────────────────────────────────────────────────┘
#
# HOW RAGAS WORKS INTERNALLY:
#   RAGAS is itself an LLM judge. It calls an LLM (GPT-4o by default) and asks:
#     "Given this question, context, and answer — how faithful is the answer? (0-1)"
#   This is called "LLM-as-judge" evaluation.
#   It's not perfect but it's far better than reading 100 outputs manually.
#
# HOW TO RUN:
#   1. Set environment variables (your .env works):
#      OPENAI_API_KEY, FASTAPI_URL (optional — if using live API mode)
#
#   2. Install ragas:
#      pip install ragas==0.2.14
#
#   3. Run:
#      cd multi_agent_architecture
#      python tests/eval_ragas.py
#
#   4. Read the score table printed at the end.
#      Target: faithfulness > 0.8, answer_relevancy > 0.8
#
# TWO EVALUATION MODES:
#   MODE 1: OFFLINE (default, no FastAPI needed)
#     Uses hardcoded test cases — fast, reproducible, good for CI.
#   MODE 2: LIVE API (uncomment the call at the bottom)
#     Calls your actual FastAPI endpoint — tests the full pipeline end-to-end.
#     Requires `docker compose up` or local FastAPI running.
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import requests
from dotenv import load_dotenv

# ── RAGAS imports ─────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Dataset builder ───────────────────────────────────────────────────────────
from datasets import Dataset   # pip install datasets (comes with ragas)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# RAGAS NEEDS AN LLM TO JUDGE ANSWERS
# WHY wrap in LangchainLLMWrapper?
#   RAGAS internally uses its own LLM abstraction. LangchainLLMWrapper bridges
#   your LangChain ChatOpenAI model into RAGAS's expected interface.
#   This means you use the same model for generation AND evaluation.
# ─────────────────────────────────────────────────────────────────────────────
judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gpt-4o",
        temperature=0,          # temperature=0 for evaluation = deterministic scores
        api_key=os.getenv("OPENAI_API_KEY")
    )
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST DATASET — hardcoded golden examples
#
# RAGAS expects a dataset with these columns:
#   question   → the user's original input
#   answer     → what your agent actually returned
#   contexts   → list of context strings (our research_findings)
#   ground_truth (optional) → ideal answer for reference metrics
#
# HOW TO BUILD A GOOD TEST DATASET:
#   1. Run your agent on 10-20 real travel queries.
#   2. Copy the output (question, research_findings, final_answer) here.
#   3. Add a ground_truth for each (write what the ideal answer should cover).
#   This becomes your "golden set" — run it every release to catch regressions.
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SAMPLES = [
    {
        "question": "Plan a 3-day trip to Paris with hotel recommendations and must-see attractions.",
        # contexts = what the researcher found (Tavily results)
        # In a real eval pipeline you'd capture this from the agent run.
        # Here we provide representative context that matches what Tavily would return.
        "contexts": [
            """Paris top hotels 2025:
            Le Marais Boutique Hotel - 4 stars, €180/night, located in the historic Marais district.
            Hotel des Arts Montmartre - 3 stars, €120/night, near Sacré-Cœur.
            Hôtel Lutetia - 5 stars, €450/night, luxury option on the Left Bank.
            Must-see attractions: Eiffel Tower (open 9am-midnight, €29 adult ticket),
            Louvre Museum (closed Tuesdays, €17), Notre-Dame Cathedral (exterior only, under repair),
            Montmartre neighborhood (free), Seine River cruise (€15, 1 hour)."""
        ],
        "answer": """Here's your 3-day Paris itinerary:

Day 1 - Iconic Paris:
- Morning: Eiffel Tower (open 9am, €29 adult ticket — book online to skip queues)
- Afternoon: Seine River cruise (€15, departs hourly near Trocadéro)
- Evening: Dinner in the 7th arrondissement

Day 2 - Art and Culture:
- Morning: Louvre Museum (€17, note: closed Tuesdays — plan accordingly)
- Afternoon: Explore the Marais district (free, great street food)
- Evening: Walk along the Seine

Day 3 - Montmartre:
- Morning: Sacré-Cœur and Montmartre neighborhood (free)
- Afternoon: Shopping on Rue des Abbesses
- Evening: Farewell dinner at a bistro

Hotel Recommendations:
- Budget: Hotel des Arts Montmartre — €120/night, near Sacré-Cœur
- Mid-range: Le Marais Boutique Hotel — €180/night, central location
- Luxury: Hôtel Lutetia — €450/night, Left Bank elegance""",
        "ground_truth": "A 3-day Paris itinerary covering Eiffel Tower, Louvre, Montmartre with hotel options at budget/mid/luxury price points and accurate opening hours and prices."
    },
    {
        "question": "What are the best beach destinations in Thailand for February?",
        "contexts": [
            """Thailand beaches in February:
            February is peak dry season — excellent beach weather, no rain.
            Koh Samui: avg 30°C, flights from Bangkok ~€80 return, beach season.
            Koh Lanta: quieter than Phuket, best snorkeling in February.
            Phuket: busiest in February, Patong beach gets crowded, prices 20% higher.
            Krabi: Railay Beach accessible only by boat, stunning limestone cliffs.
            Budget: Koh Lipe offers cheaper guesthouses ~$30/night.
            Activities: Full Moon Party on Koh Phangan, diving off Koh Tao (best visibility in Feb)."""
        ],
        "answer": """February is one of the best months to visit Thailand's beaches — it's dry season across the Gulf Coast and Andaman Sea.

Top recommendations:

**Koh Samui** - Best overall for February
- Perfect weather: 30°C average, virtually no rain
- Great for: families, couples, first-time visitors
- Flights from Bangkok: ~€80 return

**Krabi / Railay Beach** - Most scenic
- Accessible only by longtail boat (adds to the adventure)
- Stunning limestone cliffs, emerald water
- Good snorkeling and rock climbing

**Koh Lanta** - Best for tranquility
- Much less crowded than Phuket in February
- Best snorkeling visibility of the year
- Budget-friendly: guesthouses from $40/night

**Koh Tao** - Best for diving
- February = best underwater visibility of the year
- PADI courses available if you want to get certified

Avoid: Phuket's Patong Beach in February — peak prices (20% more) and very crowded.""",
        "ground_truth": "Thailand beach recommendations for February: Koh Samui, Krabi, Koh Lanta with reasons why February is ideal (dry season) and specific details like prices and activities."
    },
    {
        "question": "How much does a week in Tokyo cost including flights from London?",
        "contexts": [
            """Tokyo travel costs 2025:
            Flights London to Tokyo: £650-£900 return (budget airlines like JAL, ANA).
            Best booking window: 2-3 months ahead for cheapest fares.
            Accommodation: Budget hostel £25-40/night, mid-range hotel £80-130/night,
            luxury hotel £250+/night. 7 nights mid-range hotel = £700-£910.
            Daily food budget: £30-50/day covering convenience store meals, ramen shops,
            izakayas. Tokyo Metro day pass: £6. Attractions: TeamLab Planets £30,
            Senso-ji temple free, Robot Restaurant £70.
            Total for 7 days: Budget traveler ~£1,500, Mid-range ~£2,500, Luxury £4,000+"""
        ],
        "answer": """Here's a realistic cost breakdown for 1 week in Tokyo from London (2025):

**Flights (return, per person):**
- Budget: £650 (book 2-3 months ahead, JAL/ANA/BA)
- Average: £800
- Peak season: £900+

**Accommodation (7 nights):**
- Budget hostel: £175-280 (£25-40/night)
- Mid-range hotel: £560-910 (£80-130/night)
- Luxury: £1,750+ (£250+/night)

**Daily expenses (per person/day):**
- Food: £30-50 (convenience stores + ramen = cheap; izakayas = mid)
- Transport: ~£10-15 (day pass: £6)
- Attractions: £20-30/day average

**Total per person for 7 days:**
- Budget backpacker: ~£1,500 all-in
- Mid-range traveler: ~£2,500 all-in
- Luxury: £4,000+

**Money-saving tips:**
- 7-Eleven and Lawson convenience stores = best value meals (£3-5)
- IC card (Suica) for metro — no need for individual tickets
- Many temples and shrines are free (Senso-ji, Meiji Jingu)""",
        "ground_truth": "Cost breakdown for London-Tokyo week: flights £650-900, accommodation ranges, daily spend, total estimates for budget/mid/luxury travelers with money-saving tips."
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE EVALUATION (uses hardcoded EVAL_SAMPLES above)
# ─────────────────────────────────────────────────────────────────────────────

def run_offline_eval() -> dict:
    """
    Run RAGAS evaluation on the hardcoded test dataset.

    Returns a dict of metric scores, e.g.:
    {
        "faithfulness": 0.91,
        "answer_relevancy": 0.88,
        "context_precision": 0.85
    }

    WHY 'contexts' is a list of lists?
      RAGAS expects multiple context chunks per question (like RAG retrieval returns N chunks).
      Our agent returns ONE research_findings string, so we wrap it in a list: [findings_text].
    """
    print("\n" + "="*60)
    print("  RAGAS OFFLINE EVALUATION — TRAVEL PLANNER AGENT")
    print("="*60)
    print(f"  Evaluating {len(EVAL_SAMPLES)} test cases...")
    print(f"  Judge model: gpt-4o (temperature=0)")
    print("="*60 + "\n")

    dataset = Dataset.from_dict({
        "question":    [s["question"]    for s in EVAL_SAMPLES],
        "answer":      [s["answer"]      for s in EVAL_SAMPLES],
        "contexts":    [[s["contexts"][0]] for s in EVAL_SAMPLES],  # wrap in list
        "ground_truth":[s["ground_truth"] for s in EVAL_SAMPLES],
    })

    # ── Run evaluation ────────────────────────────────────────────────────────
    # WHAT EACH METRIC DOES INTERNALLY:
    #   faithfulness:      LLM checks: "Can each sentence in the answer be
    #                      inferred from the context?" → fraction of sentences
    #                      that are context-grounded.
    #   answer_relevancy:  Generates N questions from the answer, checks if they
    #                      match the original question using cosine similarity.
    #                      (Uses embeddings model)
    #   context_precision: Checks if the ground truth answer could be derived
    #                      from the provided context.
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    scores = result.to_pandas()[["faithfulness", "answer_relevancy", "context_precision"]]

    print("\n📊 RESULTS PER QUESTION:")
    print("-" * 60)
    for i, row in scores.iterrows():
        q = EVAL_SAMPLES[i]["question"][:60] + "..."
        print(f"\nQ{i+1}: {q}")
        print(f"  Faithfulness     : {row['faithfulness']:.3f}  {'✅' if row['faithfulness'] > 0.8 else '⚠️'}")
        print(f"  Answer Relevancy : {row['answer_relevancy']:.3f}  {'✅' if row['answer_relevancy'] > 0.8 else '⚠️'}")
        print(f"  Context Precision: {row['context_precision']:.3f}  {'✅' if row['context_precision'] > 0.8 else '⚠️'}")

    avg_scores = {
        "faithfulness":      float(scores["faithfulness"].mean()),
        "answer_relevancy":  float(scores["answer_relevancy"].mean()),
        "context_precision": float(scores["context_precision"].mean()),
    }

    print("\n" + "="*60)
    print("  AVERAGE SCORES")
    print("="*60)
    for metric, score in avg_scores.items():
        status = "✅ PASS" if score > 0.8 else "⚠️  LOW"
        print(f"  {metric:<25} {score:.3f}   {status}")
    print("="*60)
    print("\n  Target: all metrics > 0.80")
    print("  Interpret:")
    print("    0.9+  = Excellent — answers are grounded and relevant")
    print("    0.8+  = Good     — minor improvements possible")
    print("    0.7+  = Fair     — review prompts and researcher quality")
    print("    <0.7  = Poor     — hallucinations or irrelevant content detected")
    print()

    return avg_scores


# ─────────────────────────────────────────────────────────────────────────────
# LIVE API EVALUATION (calls your running FastAPI endpoint)
#
# HOW IT WORKS:
#   1. For each question in EVAL_SAMPLES, calls POST /user (blocking endpoint)
#   2. Uses the returned final_answer as the "answer"
#   3. NOTE: We don't get research_findings from /user — only the final answer.
#      For full evaluation you'd need to add a /debug endpoint that returns state.
#   4. Uses original context from EVAL_SAMPLES for context metrics.
#
# WHEN TO USE:
#   - After `docker compose up` — to test the FULL pipeline end to end
#   - Before a release — to confirm nothing broke
#   - NOT in CI (too slow, requires running services)
# ─────────────────────────────────────────────────────────────────────────────

def run_live_api_eval(fastapi_url: str = "http://localhost:8000") -> dict:
    """
    Calls the real FastAPI agent for each test case, then evaluates the live outputs.

    Args:
        fastapi_url: Base URL of your FastAPI service (default: localhost)
    """
    print(f"\n🌐 LIVE API EVAL — calling {fastapi_url}/user")
    print("   This requires FastAPI to be running (`docker compose up` or uvicorn)\n")

    live_samples = []

    for i, sample in enumerate(EVAL_SAMPLES):
        print(f"   Calling agent for Q{i+1}: {sample['question'][:60]}...")
        try:
            resp = requests.post(
                f"{fastapi_url}/user",
                json={"user_input": sample["question"]},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            live_answer = data.get("final_answer", "")
            if not live_answer or live_answer == "Failed after retries":
                print(f"   ⚠️  Q{i+1} got empty/failed answer — using placeholder")
                live_answer = "Unable to generate answer."
        except Exception as e:
            print(f"   ❌ Q{i+1} API call failed: {e}")
            live_answer = "API call failed."

        live_samples.append({
            "question":    sample["question"],
            "answer":      live_answer,
            "contexts":    sample["contexts"],   # keep original research context
            "ground_truth": sample["ground_truth"]
        })

    # Now run RAGAS on the live answers
    dataset = Dataset.from_dict({
        "question":    [s["question"]    for s in live_samples],
        "answer":      [s["answer"]      for s in live_samples],
        "contexts":    [[s["contexts"][0]] for s in live_samples],
        "ground_truth":[s["ground_truth"] for s in live_samples],
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    scores = result.to_pandas()[["faithfulness", "answer_relevancy", "context_precision"]]
    avg = {k: float(scores[k].mean()) for k in ["faithfulness", "answer_relevancy", "context_precision"]}

    print("\n  LIVE API SCORES:")
    for metric, score in avg.items():
        print(f"  {metric:<25} {score:.3f}   {'✅' if score > 0.8 else '⚠️'}")

    return avg


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    # MODE 1: Offline eval (no FastAPI needed)
    offline_scores = run_offline_eval()

    # MODE 2: Live API eval (uncomment when FastAPI is running)
    # live_scores = run_live_api_eval(fastapi_url="http://localhost:8000")

    # Save scores to JSON for CI tracking / trend analysis
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({"offline": offline_scores}, f, indent=2)
    print(f"📁 Scores saved to {output_path}")
