import json
import re
from typing import Any

from config import EVALUATION_MODEL_PATH
from agents.hf_chat import HuggingFaceChatModel
from prompts import EVALUATOR_SYSTEM_PROMPT, EVALUATOR_USER_PROMPT


class EvaluationAgent:
    def __init__(self, model: str = EVALUATION_MODEL_PATH):
        self.model = model
        self.llm = HuggingFaceChatModel(model)

    async def evaluate(
        self,
        question: str,
        answer: str,
        framework: str,
        steps: list[str],
    ) -> dict[str, Any]:
        result = await self._try_model(question, answer, framework, steps)
        if not result or self._model_result_looks_invalid(result, answer):
            result = self._heuristic(question, answer, framework, steps)

        steps_covered = result.get("steps_covered", [False] * len(steps))
        band = self._normalize_band(result.get("band"))
        score = self._band_to_score(band)
        result["band"] = band
        feedback = self._format_feedback(result)
        return {
            "steps_covered": [bool(item) for item in steps_covered[: len(steps)]],
            "score": score,
            "feedback": feedback,
        }

    async def _try_model(
        self,
        question: str,
        answer: str,
        framework: str,
        steps: list[str],
    ) -> dict[str, Any] | None:
        prompt = EVALUATOR_USER_PROMPT.format(
            question=question,
            answer=answer,
            framework=framework,
            steps="\n".join(f"- {step}" for step in steps),
        )
        response = await self.llm.generate(
            EVALUATOR_SYSTEM_PROMPT,
            prompt,
            max_new_tokens=768,
        )
        if not response:
            return None
        try:
            return json.loads(self._extract_json(response))
        except Exception:
            return None

    def _model_result_looks_invalid(self, result: dict[str, Any], answer: str) -> bool:
        answer = answer.strip()
        if not answer:
            return self._normalize_band(result.get("band")) != "No hire"

        feedback_text = " ".join(
            str(result.get(key, ""))
            for key in ("strong_points", "weak_points", "critical_gaps")
        ).lower()
        benchmark = str(result.get("benchmark_answer", "")).lower()

        generic_phrases = (
            "should clarify",
            "should include",
            "should explain",
            "strong answer should",
            "benchmark answer",
            "perfect answer",
        )
        if any(phrase in feedback_text for phrase in generic_phrases):
            return True

        if benchmark and feedback_text and self._text_similarity(feedback_text, benchmark) > 0.55:
            return True

        strong_points = result.get("strong_points", [])
        if isinstance(strong_points, list) and strong_points and answer:
            answer_words = {
                word
                for word in re.findall(r"[a-zA-Z]+", answer.lower())
                if len(word) > 3
            }
            strong_words = {
                word
                for item in strong_points
                for word in re.findall(r"[a-zA-Z]+", str(item).lower())
                if len(word) > 3
            }
            if strong_words and len(answer_words.intersection(strong_words)) == 0:
                return True

        return False

    def _text_similarity(self, left: str, right: str) -> float:
        left_words = {
            word
            for word in re.findall(r"[a-zA-Z]+", left.lower())
            if len(word) > 3
        }
        right_words = {
            word
            for word in re.findall(r"[a-zA-Z]+", right.lower())
            if len(word) > 3
        }
        if not left_words or not right_words:
            return 0.0
        return len(left_words.intersection(right_words)) / len(left_words.union(right_words))

    def _heuristic(self, question: str, answer: str, framework: str, steps: list[str]) -> dict[str, Any]:
        benchmark = self._benchmark_answer(question, framework)
        if not answer.strip():
            return {
                "benchmark_answer": benchmark,
                "steps_covered": [False] * len(steps),
                "band": "No hire",
                "strong_points": ["Nil"],
                "weak_points": ["Candidate did not give any answer."],
                "critical_gaps": ["No candidate answer was captured, so structured thinking could not be assessed."],
            }

        answer_words = set(re.findall(r"[a-zA-Z]+", answer.lower()))
        covered = []
        for step in steps:
            step_words = set(re.findall(r"[a-zA-Z]+", step.lower()))
            covered.append(bool(answer_words.intersection(step_words)))

        assessment = self._assess_structured_thinking(answer, covered, framework)

        return {
            "benchmark_answer": benchmark,
            "steps_covered": covered,
            "band": assessment["band"],
            "strong_points": assessment["strong_points"],
            "weak_points": assessment["weak_points"],
            "critical_gaps": assessment["critical_gaps"],
        }

    def _assess_structured_thinking(
        self,
        answer: str,
        covered: list[bool],
        framework: str,
    ) -> dict[str, Any]:
        text = answer.lower()
        structure_signals = [
            bool(re.search(r"\b(first|second|third|next|then|finally|step|approach)\b", text)),
            bool(re.search(r"\b(assume|requirement|constraint|clarify)\b", text)),
            bool(re.search(r"\b(tradeoff|however|but|risk|limitation)\b", text)),
            bool(re.search(r"\b(metric|evaluate|validate|test|monitor)\b", text)),
            bool(re.search(r"\b(example|for instance|such as)\b", text)),
        ]
        covered_count = sum(covered)
        structure_count = sum(structure_signals)
        word_count = len(answer.split())

        if covered_count >= 4 and structure_count >= 3 and word_count >= 80:
            band = "Strong hire"
        elif covered_count >= 3 or (structure_count >= 3 and word_count >= 60):
            band = "Hire"
        elif covered_count >= 1 or structure_count >= 1 or word_count >= 25:
            band = "Borderline"
        else:
            band = "No hire"

        strong_points = []
        if covered_count >= 3:
            strong_points.append("Covered most of the expected framework areas.")
        elif covered_count > 0:
            strong_points.append("Covered at least part of the expected framework.")
        if structure_count >= 2:
            strong_points.append("Showed some structured thinking instead of only giving isolated facts.")
        if word_count >= 60:
            strong_points.append("Provided enough detail to understand the direction of the answer.")

        weak_points = []
        if covered_count < 3:
            weak_points.append("Did not clearly cover enough key areas for a confident pass.")
        if structure_count < 2:
            weak_points.append("The reasoning process was not explicit enough.")
        if word_count < 40:
            weak_points.append("The answer was brief, so the interviewer has limited evidence of depth.")

        critical_gaps = []
        if framework == "System Design" and not re.search(r"\b(requirement|constraint|scale|latency|tradeoff|failure)\b", text):
            critical_gaps.append("For system design, the answer needs clearer requirements, constraints, scale, and tradeoff reasoning.")
        if framework == "Technical" and not re.search(r"\b(example|metric|evaluate|tradeoff|assumption|why)\b", text):
            critical_gaps.append("For a technical answer, the candidate should explain why the concept works and give an example or validation angle.")
        if not critical_gaps:
            critical_gaps.append("Nil")

        return {
            "band": band,
            "strong_points": strong_points or ["Nil"],
            "weak_points": weak_points or ["Nil"],
            "critical_gaps": critical_gaps,
        }

    def _compare_to_benchmark(self, answer: str, benchmark: str) -> dict[str, Any]:
        concepts = self._benchmark_concepts(benchmark)
        answer_text = answer.lower()
        matched = [concept for concept in concepts if self._concept_is_covered(concept, answer_text)]
        missed = [concept for concept in concepts if concept not in matched]

        coverage = len(matched) / max(1, len(concepts))
        score = max(1, min(5, round(coverage * 5)))
        if len(answer.split()) < 30 and score > 2:
            score = 2

        strong_points = [f"Covered {concept}." for concept in matched[:4]] or ["Nil"]
        weak_points = [f"Did not clearly explain {concept}." for concept in missed[:4]] or ["Nil"]
        critical_gaps = [f"Missing benchmark concept: {concept}." for concept in missed[:5]] or ["Nil"]

        if len(answer.split()) < 30:
            weak_points.append("The answer is too brief to demonstrate the reasoning expected by the benchmark.")
        return {
            "score": score,
            "strong_points": strong_points,
            "weak_points": weak_points,
            "critical_gaps": critical_gaps,
        }

    def _benchmark_concepts(self, benchmark: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", benchmark)
        concepts = []
        for sentence in sentences:
            clean = sentence.strip()
            if not clean:
                continue
            if ":" in clean and len(clean.split(":")[0].split()) <= 5:
                label, detail = clean.split(":", 1)
                clean = f"{label.strip()} ({detail.strip()})"
            concepts.append(clean.rstrip("."))
        return concepts[:10]

    def _concept_is_covered(self, concept: str, answer_text: str) -> bool:
        concept_words = {
            word
            for word in re.findall(r"[a-zA-Z]+", concept.lower())
            if len(word) > 3 and word not in self._stopwords()
        }
        if not concept_words:
            return False
        answer_words = set(re.findall(r"[a-zA-Z]+", answer_text))
        overlap = concept_words.intersection(answer_words)
        required = 1 if len(concept_words) <= 3 else max(2, min(4, len(concept_words) // 3))
        return len(overlap) >= required

    def _stopwords(self) -> set[str]:
        return {
            "correct",
            "answer",
            "should",
            "include",
            "explain",
            "mention",
            "strong",
            "typical",
            "common",
            "uses",
            "such",
            "that",
            "with",
            "from",
            "into",
            "where",
            "when",
            "while",
            "also",
            "mainly",
            "useful",
        }

    def _benchmark_answer(self, question: str, framework: str) -> str:
        text = question.lower()
        if "supervised" in text and "unsupervised" in text:
            return (
                "A correct answer should explain that supervised learning uses labeled training data, where each example has input features "
                "and a known target label or value. The model learns a mapping from inputs to outputs and is evaluated against ground truth. "
                "Typical supervised tasks include classification, such as spam detection or churn prediction, and regression, such as predicting "
                "house prices. Common algorithms include linear regression, logistic regression, decision trees, random forests, gradient boosting, "
                "support vector machines, and neural networks. Unsupervised learning uses unlabeled data and tries to discover structure, patterns, "
                "or representations without a target label. Typical tasks include clustering, dimensionality reduction, anomaly detection, and "
                "association discovery. Common algorithms include k-means, hierarchical clustering, DBSCAN, PCA, t-SNE or UMAP for visualization, "
                "autoencoders, and Gaussian mixture models. A strong answer should also mention evaluation differences: supervised models can use "
                "metrics like accuracy, precision, recall, F1, RMSE, or MAE, while unsupervised models are harder to evaluate and may use silhouette "
                "score, reconstruction error, downstream task performance, or human/business validation."
            )

        if "linear regression" in text and "assumption" in text:
            return (
                "A correct answer should state the main assumptions of linear regression: "
                "1. Linearity: the expected target is a linear combination of the predictors. "
                "2. Independence: observations and residual errors are independent, with no autocorrelation. "
                "3. Homoscedasticity: residuals have constant variance across predicted values. "
                "4. No perfect multicollinearity: predictors are not exact linear combinations of each other. "
                "5. Exogeneity: errors have mean zero and are not correlated with the predictors. "
                "6. Normality of residuals is mainly needed for small-sample hypothesis tests and confidence intervals, "
                "not for unbiased coefficient estimates. A strong answer should also mention checking residual plots, "
                "variance inflation factor for multicollinearity, and transformations or robust standard errors when assumptions fail."
            )

        if "logistic regression" in text and "assumption" in text:
            return (
                "A correct answer should explain that logistic regression assumes independent observations, "
                "a linear relationship between predictors and the log-odds of the target, no severe multicollinearity, "
                "adequate sample size, correctly specified features and interactions, and limited influence from extreme outliers. "
                "It should also mention that the target is binary or modeled as binomial, and that calibration, ROC-AUC, precision, recall, "
                "and confusion-matrix tradeoffs are useful evaluation checks."
            )

        if "recommendation" in text or "recommender" in text:
            return (
                "A strong benchmark answer should clarify users, items, goals, constraints, and success metrics such as CTR, conversion, "
                "retention, NDCG, recall@K, and diversity. It should propose candidate generation using collaborative filtering, "
                "content-based retrieval, or embeddings; ranking using a learned model with user, item, and context features; "
                "and feedback loops from clicks, ratings, purchases, skips, and dwell time. It should cover cold start, popularity bias, "
                "exploration versus exploitation, freshness, latency, offline and online evaluation, A/B testing, monitoring drift, "
                "and abuse or privacy concerns."
            )

        if "overfitting" in text or "underfitting" in text:
            return (
                "A correct answer should define overfitting as low training error but poor generalization, and underfitting as poor performance "
                "on both train and validation data. It should mention causes such as excessive model complexity, noisy features, data leakage, "
                "or insufficient regularization for overfitting, and overly simple models or insufficient features for underfitting. "
                "It should include fixes such as cross-validation, regularization, more data, feature selection, early stopping, pruning, "
                "simpler or richer models as appropriate, and monitoring train-validation learning curves."
            )

        if "bias" in text and "variance" in text:
            return (
                "A correct answer should explain bias as error from overly restrictive assumptions and variance as sensitivity to training data. "
                "High bias causes underfitting; high variance causes overfitting. The answer should discuss the tradeoff, how model complexity "
                "affects each side, and practical diagnosis using train and validation errors. It should mention remedies such as adding features "
                "or model capacity for high bias, and regularization, more data, ensembling, or simpler models for high variance."
            )

        if ("rate limit" in text or "rate limiting" in text) and (
            "token" in text or "consumption" in text or "aggregate" in text or "aggregates" in text
        ):
            return (
                "A strong benchmark answer should design a low-latency token usage metering and rate-limiting service. "
                "First clarify requirements: limit by user, API key, organization, model, endpoint, or time window; support per-minute, "
                "daily, and monthly quotas; handle burst limits; provide accurate enough enforcement with very low request-path latency; "
                "and expose usage dashboards and audit logs. The request path should call a Rate Limit service before or during inference. "
                "That service should use Redis or another fast distributed counter store for hot-window counters, commonly with token bucket, "
                "leaky bucket, or sliding-window counters keyed by tenant and model. Estimated input tokens can be checked before admission, "
                "then final actual input plus output tokens should be committed after completion. For streaming responses, token usage can be "
                "reserved up front, incrementally updated, or reconciled at stream end. The system should write durable usage events to Kafka, "
                "Kinesis, or a log table, then aggregate asynchronously into OLAP/storage such as ClickHouse, BigQuery, or partitioned Postgres "
                "tables for reporting. The design should cover idempotency with request IDs, atomic counter updates, TTLs for window counters, "
                "clock/window boundary handling, refunds for failed requests, backpressure behavior, multi-region consistency tradeoffs, "
                "eventual reconciliation between Redis and durable aggregates, and observability metrics such as allowed/blocked requests, "
                "counter latency, aggregation lag, dropped events, and quota accuracy."
            )

        if framework == "System Design":
            return (
                "A strong answer should clarify functional and non-functional requirements, define APIs and data entities, propose a high-level "
                "architecture, explain storage and serving choices, discuss scaling, caching, reliability, observability, latency, throughput, "
                "and failure modes, then close with tradeoffs and validation metrics."
            )

        if framework == "Technical":
            return (
                "A strong technical answer should define the concept accurately, state assumptions, explain the mechanism or formula where relevant, "
                "give practical examples, discuss edge cases and tradeoffs, and mention how to validate the approach in production or experiments."
            )

        return (
            "A strong answer should directly answer the question, define key terms, provide a structured explanation, include concrete examples, "
            "discuss tradeoffs or limitations, and close with how the answer would be validated or applied in practice."
        )

    def _normalize_score(self, value: Any) -> int:
        try:
            score = int(float(value))
        except (TypeError, ValueError):
            score = 3
        return max(0, min(5, score))

    def _normalize_band(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        labels = {
            "strong hire": "Strong hire",
            "hire": "Hire",
            "borderline": "Borderline",
            "no hire": "No hire",
        }
        if text in labels:
            return labels[text]

        score = self._normalize_score(value)
        if score >= 5:
            return "Strong hire"
        if score >= 4:
            return "Hire"
        if score >= 2:
            return "Borderline"
        return "No hire"

    def _band_to_score(self, band: str) -> int:
        return {
            "Strong hire": 5,
            "Hire": 4,
            "Borderline": 2,
            "No hire": 0,
        }.get(band, 2)

    def _format_feedback(self, result: dict[str, Any]) -> str:
        benchmark = str(
            result.get("benchmark_answer")
            or result.get("baseline_answer")
            or "No benchmark answer was generated."
        ).strip()
        band = self._normalize_band(result.get("band") or result.get("score", 2))
        strong_points = self._format_list(result.get("strong_points") or result.get("strengths") or ["Nil"])
        weak_points = self._format_list(result.get("weak_points") or ["Nil"])
        critical_gaps = self._format_list(
            result.get("critical_gaps")
            or result.get("gaps")
            or result.get("improvements")
            or ["Nil"]
        )
        return (
            f"Agent benchmark answer:\n{benchmark}\n\n"
            f"Evaluation band: {band}\n\n"
            f"Strong points:\n{strong_points}\n\n"
            f"Weak points:\n{weak_points}\n\n"
            f"Critical gaps:\n{critical_gaps}"
        )

    def _format_list(self, value: Any) -> str:
        if not isinstance(value, list):
            return str(value).strip()
        return "\n".join(f"- {item}" for item in value if str(item).strip())

    def _extract_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else text
