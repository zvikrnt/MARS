import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "tmp_master_merged.json"
OUTPUT_CSV = OUT_DIR / "fixedelo_bot_synthetic_dataset.csv"
STUDENTS_CSV = OUT_DIR / "fixedelo_student_personas.csv"
PERSONA_SUMMARY_CSV = OUT_DIR / "fixedelo_persona_summary.csv"
SUMMARY_JSON = OUT_DIR / "fixedelo_dataset_summary.json"

RNG = random.Random(20260602)
FIXED_ELO_K = 32

COLUMNS = [
    "student_id",
    "day_index",
    "date",
    "timestamp",
    "question_id",
    "subject",
    "topic",
    "difficulty",
    "question_elo",
    "question_elo_category",
    "centrality",
    "question_tags",
    "graph_mode",
    "remediation_for_question_id",
    "remediation_step",
    "graph_edge_kind",
    "graph_edge_weight",
    "recommendation_reason",
    "R_before",
    "rating_level_before",
    "v_before",
    "n_before",
    "sigma2_before",
    "k_before",
    "m_before",
    "W_before",
    "Qans_before_count",
    "Qsrv_before_count",
    "pi_rem_before",
    "pred_prob",
    "response_quality",
    "correct",
    "response_time_sec",
    "Ki",
    "phi",
    "Ei",
    "delta_Rraw",
    "v_after",
    "rating_level_after",
    "R_after",
    "sigma2_after",
    "k_after",
    "m_after",
    "W_after",
    "Qans_after_count",
    "Qsrv_after_count",
    "pi_rem_after",
]

PERSONAS = [
    {
        "name": "Topper",
        "students": 30,
        "start": (1840, 2100),
        "skill": 2100,
        "growth": 2.8,
        "care": 0.10,
        "speed": 0.62,
        "attempts": (1900, 2350),
        "challenge": 120,
        "volatility": 0.035,
        "accuracy_bias": 0.22,
    },
    {
        "name": "High Achiever",
        "students": 30,
        "start": (1660, 1880),
        "skill": 1880,
        "growth": 2.3,
        "care": 0.13,
        "speed": 0.72,
        "attempts": (1750, 2200),
        "challenge": 90,
        "volatility": 0.045,
        "accuracy_bias": 0.17,
    },
    {
        "name": "Consistent Improver",
        "students": 30,
        "start": (1450, 1680),
        "skill": 1600,
        "growth": 3.0,
        "care": 0.17,
        "speed": 0.88,
        "attempts": (1600, 2050),
        "challenge": 35,
        "volatility": 0.055,
        "accuracy_bias": 0.10,
    },
    {
        "name": "Exam Sprinter",
        "students": 30,
        "start": (1380, 1620),
        "skill": 1540,
        "growth": 1.4,
        "care": 0.20,
        "speed": 0.70,
        "attempts": (1500, 1950),
        "challenge": 70,
        "volatility": 0.08,
        "accuracy_bias": 0.06,
    },
    {
        "name": "Average Steady",
        "students": 30,
        "start": (1220, 1480),
        "skill": 1360,
        "growth": 1.7,
        "care": 0.24,
        "speed": 1.0,
        "attempts": (1350, 1750),
        "challenge": 0,
        "volatility": 0.07,
        "accuracy_bias": 0.01,
    },
    {
        "name": "Inconsistent Worker",
        "students": 30,
        "start": (1160, 1430),
        "skill": 1300,
        "growth": 1.0,
        "care": 0.29,
        "speed": 0.82,
        "attempts": (1250, 1700),
        "challenge": 20,
        "volatility": 0.13,
        "accuracy_bias": -0.04,
    },
    {
        "name": "Overconfident Fast",
        "students": 30,
        "start": (1180, 1480),
        "skill": 1280,
        "growth": 0.8,
        "care": 0.36,
        "speed": 0.48,
        "attempts": (1350, 1800),
        "challenge": 160,
        "volatility": 0.11,
        "accuracy_bias": -0.09,
    },
    {
        "name": "Slow Remediator",
        "students": 30,
        "start": (980, 1240),
        "skill": 1130,
        "growth": 1.8,
        "care": 0.22,
        "speed": 1.35,
        "attempts": (1050, 1450),
        "challenge": -60,
        "volatility": 0.075,
        "accuracy_bias": -0.06,
    },
    {
        "name": "Anxious Guesser",
        "students": 30,
        "start": (850, 1120),
        "skill": 980,
        "growth": 0.7,
        "care": 0.42,
        "speed": 0.55,
        "attempts": (900, 1300),
        "challenge": 40,
        "volatility": 0.14,
        "accuracy_bias": -0.14,
    },
    {
        "name": "Disengaged Beginner",
        "students": 30,
        "start": (650, 940),
        "skill": 790,
        "growth": 0.25,
        "care": 0.48,
        "speed": 0.78,
        "attempts": (650, 1100),
        "challenge": -20,
        "volatility": 0.12,
        "accuracy_bias": -0.22,
    },
]

TOPIC_TAGS = {
    "la-matrices": "matrices|determinants|rank|linear-transformations",
    "la-eigenvalues": "eigenvalues|eigenvectors|diagonalization|spectral-theorem",
    "probability-distributions": "normal-distribution|binomial-distribution|expected-value|variance",
    "statistics": "estimation|hypothesis-testing|confidence-intervals|regression",
    "ml-supervised": "linear-regression|logistic-regression|decision-trees|svm|ensembles",
    "ml-unsupervised": "clustering|pca|k-means|dimensionality-reduction",
    "dbms-normalization": "functional-dependencies|normal-forms|lossless-join|bcnf",
    "algorithms": "time-complexity|sorting|graphs|dynamic-programming",
    "python-programming": "python|numpy|pandas|control-flow|functions",
    "verbal-reasoning": "english|grammar|reading-comprehension|word-usage",
}


def expected_score(student_rating, item_rating):
    return 1.0 / (1.0 + math.pow(10.0, (item_rating - student_rating) / 400.0))


def fixed_elo_update(student_rating, item_rating, correct):
    expected = expected_score(student_rating, item_rating)
    actual = 1 if correct else 0
    delta = round(FIXED_ELO_K * (actual - expected))
    return expected, delta, max(0, round(student_rating + delta))


def rating_level(rating):
    if rating < 1000:
        return "Easy"
    if rating < 1400:
        return "Medium"
    if rating < 1750:
        return "Hard"
    return "Expert"


def elo_category(elo):
    if elo < 1000:
        return "Easy"
    if elo < 1400:
        return "Medium"
    if elo < 1750:
        return "Hard"
    return "Expert"


def clamp(value, low, high):
    return max(low, min(high, value))


def load_questions():
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)

    questions = []
    for q in raw:
        elo = int(q.get("eloRating") or 1200)
        topic = str(q.get("topicId") or "general")
        tags = q.get("tags") or TOPIC_TAGS.get(topic) or topic.replace("-", "|")
        if isinstance(tags, list):
            tags = "|".join(str(tag) for tag in tags[:10])
        questions.append(
            {
                "id": str(q.get("id")),
                "subject": str(q.get("subjectId") or "general"),
                "topic": topic,
                "difficulty": str(q.get("difficulty") or "medium").lower(),
                "elo": elo,
                "tags": tags,
                "centrality": round(RNG.betavariate(2.2, 2.0), 3),
            }
        )

    if not questions:
        raise RuntimeError(f"No questions found in {QUESTIONS_PATH}")
    return questions


def choose_question(questions, target_rating, preferred_topic=None, easier_than=None):
    pool = questions
    if preferred_topic:
        same_topic = [q for q in questions if q["topic"] == preferred_topic]
        if same_topic:
            pool = same_topic
    if easier_than is not None:
        easier = [q for q in pool if q["elo"] <= easier_than]
        if easier:
            pool = easier

    candidates = RNG.sample(pool, min(len(pool), 80))
    return min(candidates, key=lambda q: abs(q["elo"] - target_rating) + RNG.uniform(0, 90))


def response_time(question_elo, rating, correct, persona):
    gap = question_elo - rating
    base = 28 + max(0, gap) * 0.035 + RNG.gammavariate(2.4, 8.5)
    if not correct:
        base += RNG.uniform(4, 24)
    if persona["name"] == "Overconfident Fast":
        base *= RNG.uniform(0.45, 0.78)
    elif persona["name"] == "Anxious Guesser":
        base *= RNG.choice([RNG.uniform(0.35, 0.7), RNG.uniform(1.15, 1.75)])
    else:
        base *= persona["speed"] * RNG.uniform(0.78, 1.24)
    return round(clamp(base, 3.5, 210.0), 6)


def make_pi_rem(unresolved):
    if not unresolved:
        return "{}"
    compact = {}
    for qid, data in list(unresolved.items())[-6:]:
        compact[qid] = {
            "topic": data["topic"],
            "tags": data["tags"],
            "rem_attempts": data["rem_attempts"],
            "rem_correct": data["rem_correct"],
            "retry_attempts": data["retry_attempts"],
        }
    return repr(compact)


def build_students():
    students = []
    sid = 1
    for persona_index, persona in enumerate(PERSONAS, start=1):
        for _ in range(persona["students"]):
            start_rating = RNG.randint(*persona["start"])
            target_attempts = RNG.randint(*persona["attempts"])
            active_days = RNG.randint(80, 132)
            students.append(
                {
                    "student_id": sid,
                    "persona": persona["name"],
                    "persona_rank": persona_index,
                    "start_rating": start_rating,
                    "latent_skill": persona["skill"] + RNG.gauss(0, 55),
                    "target_attempts": target_attempts,
                    "active_days": active_days,
                    "persona_config": persona,
                }
            )
            sid += 1
    return students


def generate():
    OUT_DIR.mkdir(exist_ok=True)
    questions = load_questions()
    students = build_students()
    start_date = datetime(2026, 1, 29, 8, 15, 0)

    row_count = 0
    persona_counts = Counter()
    persona_correct = Counter()
    persona_attempts = Counter()
    final_ratings = {}
    student_rows = []

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for student in students:
            sid = student["student_id"]
            persona = student["persona_config"]
            rating = float(student["start_rating"])
            latent_skill = float(student["latent_skill"])
            total_attempts = student["target_attempts"]
            active_days = student["active_days"]
            attempts_done = 0
            n = 0
            velocity = 0.0
            sigma2 = 55.0 + (10 - student["persona_rank"]) * 2.5
            streak = 0
            weak_topic = ""
            current_topic = ""
            unresolved = {}
            topic_counts = defaultdict(int)
            student_correct = 0
            date_offset = RNG.randint(0, 10)

            daily_weights = [max(0.1, RNG.lognormvariate(0.0, 0.55)) for _ in range(active_days)]
            skipped = set(RNG.sample(range(active_days), k=RNG.randint(3, 16)))
            weight_sum = sum(w for i, w in enumerate(daily_weights) if i not in skipped)

            for day in range(active_days):
                if attempts_done >= total_attempts:
                    break
                if day in skipped:
                    continue

                expected_today = total_attempts * daily_weights[day] / weight_sum
                session_attempts = max(1, int(RNG.gauss(expected_today, max(1.2, expected_today * 0.25))))
                if RNG.random() < 0.12:
                    session_attempts += RNG.randint(3, 10)
                session_attempts = min(session_attempts, total_attempts - attempts_done)

                day_date = start_date + timedelta(days=day + date_offset)
                session_start = day_date.replace(
                    hour=RNG.randint(6, 23),
                    minute=RNG.randint(0, 59),
                    second=RNG.randint(0, 59),
                    microsecond=RNG.randint(0, 999999),
                )
                seconds_cursor = 0

                for _ in range(session_attempts):
                    if attempts_done >= total_attempts:
                        break

                    R_before = rating
                    v_before = velocity
                    sigma_before = sigma2
                    k_before = streak
                    m_before = "high" if streak >= 4 else "low"
                    W_before = weak_topic
                    Qans_before = n
                    Qsrv_before = n
                    pi_before = make_pi_rem(unresolved)

                    graph_mode = "standard"
                    remediation_for = ""
                    remediation_step = 0
                    edge_kind = RNG.choice(["bridge", "same-topic", "prerequisite", "start"])
                    edge_weight = round(RNG.uniform(0.05, 0.95), 4)
                    reason = "standard graph-neighbourhood recommendation"

                    if unresolved and RNG.random() < (0.24 + persona["care"] * 0.35):
                        remediation_for = RNG.choice(list(unresolved.keys()))
                        rem = unresolved[remediation_for]
                        graph_mode = "remediation"
                        remediation_step = rem["rem_attempts"] + 1
                        edge_kind = "bridge"
                        edge_weight = round(RNG.uniform(0.52, 0.86), 4)
                        target = rating - RNG.uniform(80, 230)
                        question = choose_question(
                            questions,
                            target,
                            preferred_topic=rem["topic"],
                            easier_than=rem["elo"] - RNG.randint(20, 120),
                        )
                        reason = "tag/topic related easier remediation for unresolved miss"
                    elif unresolved and RNG.random() < 0.08:
                        remediation_for = RNG.choice(list(unresolved.keys()))
                        rem = unresolved[remediation_for]
                        graph_mode = "retry"
                        remediation_step = rem["retry_attempts"] + 1
                        edge_kind = "same-topic"
                        edge_weight = round(RNG.uniform(0.005, 0.18), 4)
                        target = rem["elo"] + RNG.uniform(-25, 25)
                        question = choose_question(questions, target, preferred_topic=rem["topic"])
                        reason = "retry original missed item after related remediation success"
                    else:
                        target = rating + persona["challenge"] + RNG.gauss(0, 115)
                        if current_topic and RNG.random() < 0.42:
                            question = choose_question(questions, target, preferred_topic=current_topic)
                            edge_kind = "same-topic"
                        else:
                            question = choose_question(questions, target)

                    drift = persona["growth"] * (attempts_done / max(1, total_attempts)) * 120
                    effective_skill = latent_skill + drift + RNG.gauss(0, 95 * persona["volatility"])
                    base_prob = expected_score(effective_skill, question["elo"])
                    fatigue = -0.035 if attempts_done % 70 > 58 else 0.0
                    careless = persona["care"] * RNG.random()
                    p_correct = clamp(
                        base_prob + persona["accuracy_bias"] + fatigue - careless + RNG.gauss(0, 0.025),
                        0.03,
                        0.97,
                    )
                    correct = 1 if RNG.random() < p_correct else 0

                    pred_prob, delta, next_rating = fixed_elo_update(R_before, question["elo"], bool(correct))
                    quality = clamp((0.58 * correct) + (0.42 * p_correct) + RNG.gauss(0, 0.04), 0, 1)
                    rt = response_time(question["elo"], R_before, bool(correct), persona)

                    velocity = round((0.72 * velocity) + (0.28 * delta), 3)
                    sigma2 = round(clamp((sigma2 * 0.987) + abs(delta) * 0.18 + RNG.uniform(-0.35, 0.25), 8.0, 70.0), 4)
                    streak = streak + 1 if correct else -1
                    if correct:
                        if graph_mode == "remediation" and remediation_for in unresolved:
                            unresolved[remediation_for]["rem_correct"] += 1
                        if graph_mode == "retry" and remediation_for in unresolved:
                            unresolved[remediation_for]["retry_attempts"] += 1
                            if unresolved[remediation_for]["retry_attempts"] >= 1:
                                unresolved.pop(remediation_for, None)
                    else:
                        weak_topic = question["topic"]
                        unresolved[question["id"]] = {
                            "topic": question["topic"],
                            "tags": question["tags"],
                            "elo": question["elo"],
                            "rem_attempts": 0,
                            "rem_correct": 0,
                            "retry_attempts": 0,
                        }

                    if graph_mode == "remediation" and remediation_for in unresolved:
                        unresolved[remediation_for]["rem_attempts"] += 1

                    n += 1
                    attempts_done += 1
                    topic_counts[question["topic"]] += 1
                    current_topic = question["topic"] if RNG.random() < 0.7 else current_topic
                    rating = float(next_rating)
                    seconds_cursor += int(rt + RNG.uniform(5, 75))
                    ts = session_start + timedelta(seconds=seconds_cursor)

                    row = {
                        "student_id": sid,
                        "day_index": day,
                        "date": ts.date().isoformat(),
                        "timestamp": ts.isoformat(),
                        "question_id": question["id"],
                        "subject": question["subject"],
                        "topic": question["topic"],
                        "difficulty": question["difficulty"],
                        "question_elo": question["elo"],
                        "question_elo_category": elo_category(question["elo"]),
                        "centrality": question["centrality"],
                        "question_tags": question["tags"],
                        "graph_mode": graph_mode,
                        "remediation_for_question_id": remediation_for,
                        "remediation_step": remediation_step,
                        "graph_edge_kind": edge_kind,
                        "graph_edge_weight": edge_weight,
                        "recommendation_reason": reason,
                        "R_before": round(R_before, 3),
                        "rating_level_before": rating_level(R_before),
                        "v_before": round(v_before, 3),
                        "n_before": Qans_before,
                        "sigma2_before": round(sigma_before, 3),
                        "k_before": k_before,
                        "m_before": m_before,
                        "W_before": W_before,
                        "Qans_before_count": Qans_before,
                        "Qsrv_before_count": Qsrv_before,
                        "pi_rem_before": pi_before,
                        "pred_prob": round(pred_prob, 4),
                        "response_quality": round(quality, 6),
                        "correct": correct,
                        "response_time_sec": rt,
                        "Ki": FIXED_ELO_K,
                        "phi": 1.0,
                        "Ei": round(pred_prob, 4),
                        "delta_Rraw": delta,
                        "v_after": velocity,
                        "rating_level_after": rating_level(rating),
                        "R_after": round(rating, 3),
                        "sigma2_after": sigma2,
                        "k_after": streak,
                        "m_after": "high" if streak >= 4 else "low",
                        "W_after": weak_topic,
                        "Qans_after_count": n,
                        "Qsrv_after_count": n,
                        "pi_rem_after": make_pi_rem(unresolved),
                    }
                    writer.writerow(row)

                    row_count += 1
                    persona_counts[student["persona"]] += 1
                    persona_attempts[student["persona"]] += 1
                    persona_correct[student["persona"]] += correct
                    student_correct += correct

            final_ratings[sid] = round(rating, 3)
            student_rows.append(
                {
                    "student_id": sid,
                    "persona": student["persona"],
                    "persona_rank": student["persona_rank"],
                    "start_rating": student["start_rating"],
                    "final_rating": round(rating, 3),
                    "attempts": attempts_done,
                    "accuracy": round(student_correct / max(1, attempts_done), 4),
                    "active_days": active_days,
                    "top_topic": max(topic_counts, key=topic_counts.get) if topic_counts else "",
                }
            )

    with STUDENTS_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "student_id",
            "persona",
            "persona_rank",
            "start_rating",
            "final_rating",
            "attempts",
            "accuracy",
            "active_days",
            "top_topic",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(student_rows)

    persona_student_rows = defaultdict(list)
    for row in student_rows:
        persona_student_rows[row["persona"]].append(row)

    persona_summary_rows = []
    for persona in PERSONAS:
        name = persona["name"]
        rows = persona_student_rows[name]
        attempts = sum(int(row["attempts"]) for row in rows)
        persona_summary_rows.append(
            {
                "persona_rank": next(row["persona_rank"] for row in rows),
                "persona": name,
                "students": len(rows),
                "attempt_rows": attempts,
                "mean_start_rating": round(sum(float(row["start_rating"]) for row in rows) / len(rows), 3),
                "mean_final_rating": round(sum(float(row["final_rating"]) for row in rows) / len(rows), 3),
                "mean_student_accuracy": round(sum(float(row["accuracy"]) for row in rows) / len(rows), 4),
                "min_final_rating": min(float(row["final_rating"]) for row in rows),
                "max_final_rating": max(float(row["final_rating"]) for row in rows),
            }
        )

    with PERSONA_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "persona_rank",
            "persona",
            "students",
            "attempt_rows",
            "mean_start_rating",
            "mean_final_rating",
            "mean_student_accuracy",
            "min_final_rating",
            "max_final_rating",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(persona_summary_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_question_bank": str(QUESTIONS_PATH.relative_to(ROOT)),
        "model": "FixedElo",
        "k": FIXED_ELO_K,
        "formula": "delta = round(K * (actual - 1 / (1 + 10 ** ((question_elo - R_before) / 400))))",
        "students": len(students),
        "rows": row_count,
        "columns": len(COLUMNS),
        "output_csv": OUTPUT_CSV.name,
        "students_csv": STUDENTS_CSV.name,
        "persona_summary_csv": PERSONA_SUMMARY_CSV.name,
        "persona_rows": dict(persona_counts),
        "persona_accuracy": {
            name: round(persona_correct[name] / max(1, persona_attempts[name]), 4)
            for name in persona_counts
        },
        "persona_summary": persona_summary_rows,
        "rating_min": min(final_ratings.values()),
        "rating_max": max(final_ratings.values()),
        "rating_mean": round(sum(final_ratings.values()) / len(final_ratings), 3),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = generate()
    print(json.dumps(result, indent=2))
