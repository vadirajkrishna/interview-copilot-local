# Interview Coach: Real-Time, Local Interview Assistance Without Taking Over the Answer

## Introduction

Technical interviews are noisy, fast, and cognitively expensive. A candidate has to listen carefully, identify the actual question, choose the right structure, remember the important concepts, and answer naturally, all in a few seconds.

Interview Coach is built for that exact moment. It listens to the interview audio, extracts the interviewer’s Data Science, Machine Learning, AI, or System Design question, and shows a compact coaching card with important pointers. The goal is not to generate an answer for the candidate or handhold them through the interview. The goal is to provide timely reminders so the candidate can cover the important parts of their own answer.

The app also helps after the interview. It extracts question-answer pairs from noisy transcripts, stores them in SQLite, and evaluates the candidate’s answers with structured feedback. That makes it useful both as a live coaching tool and as a practice-review system.

The full setup is designed to run locally on a Mac. This keeps latency low, avoids sending interview audio to a cloud service, and makes the experience feel private and responsive.

## Architecture

At a high level, Interview Coach has two paths: a fast live coaching path and a slower post-session evaluation path.

```text
Live audio
  -> Speech-to-text
  -> Fast question boundary detection
  -> Fine-tuned topic/pattern classifier
  -> Coaching hint generator
  -> Floating coaching cards

Full transcript
  -> Q&A extraction
  -> SQLite persistence
  -> Evaluation agent
  -> Feedback table / CSV export
```

The live path is optimized for speed. It tries to show the coaching card immediately after the interviewer asks the question, before the candidate has already started answering. The post-session path is optimized for accuracy and reflection, so it can spend more time cleaning up the transcript and evaluating the answer.

## Models Used

The project uses a multi-model approach. Instead of forcing one model to handle every task, each model has a focused responsibility.

| Model | Approx Size | Purpose |
| --- | ---: | --- |
| `mlx-community/whisper-tiny` | ~39M parameters | Fast streaming transcription for live audio |
| `mlx-community/whisper-small-mlx` | ~244M parameters | Higher-quality local transcription when needed |
| `vadirajkrishna/interview-coach-3b` | 3B base model plus LoRA adapter | Fine-tuned topic and pattern detection |
| `Qwen/Qwen2.5-3B-Instruct` | ~3B parameters | General reasoning, coaching hint generation, transcript cleanup, and evaluation |
| SQLite | Local database | Session, exchange, and evaluation storage |

This split is important. The fine-tuned model is used for what it is good at: quickly identifying the type and pattern of the question. The general instruction model is used when more flexible reasoning is needed, such as generating hints or extracting structured Q&A from a noisy transcript.

## Fine-Tuning Approach

The fine-tuned model was trained to map interview questions to a type and a set of coarse answering steps. For example:

```json
{
  "prompt": "How would you handle class imbalance in a fraud detection model?",
  "completion": "Type: Data Science\nSteps: Check base rate -> Choose right metric -> Handle imbalance -> Tune threshold -> Evaluate"
}
```

This helped in two ways.

First, the coaching system became more reliable at deciding whether a transcript segment was relevant. Greetings, logistics, and generic discussion should not trigger a coaching card. The fine-tuned model helps classify only meaningful DS, ML, AI, or System Design questions.

Second, it gave the live system a fast topic/pattern signal. The app does not need to ask a larger model to deeply reason about the question before showing anything. It can quickly classify the question and then use that classification as context for generating better hints.

The final design uses the fine-tuned 3B model for type and pattern detection, not for the final coaching bullets. The displayed hints are generated separately so they can be more specific and useful.

## Challenge 1: Making the Coaching Card Appear Fast Enough

The most critical product challenge was timing.

If the coaching card appears after the candidate has already started answering, it is too late. The card has to appear right after the interviewer asks the question and before the candidate has committed to an answer structure.

The first version used a general LLM to extract the clean question before showing the card. That was accurate, but too slow. The fix was to split the live path from the post-processing path.

For the live path, the app now uses:

- A recent transcript window instead of the full transcript.
- Fast question boundary cleanup.
- A warm-loaded fine-tuned topic/pattern model.
- A separate coaching-hints prompt only after the question is detected.
- Persistent floating cards so earlier coaching cards do not disappear.

The live extractor intentionally does not try to produce a perfect transcript. It aims to identify the current question quickly enough to help the candidate. Accuracy cleanup happens later in `Process Text`.

This tradeoff made the coaching card feel much faster and more useful during an actual interview.

## Challenge 2: Extracting Questions and Answers From Noisy Conversations

Live transcripts are messy. A single transcript may contain:

- Interviewer greetings.
- Candidate acknowledgements.
- Repeated STT fragments.
- Half-finished questions.
- Candidate answers starting before punctuation is clear.
- Transitions like “Good, next question”.
- Multiple questions and answers in one block.

One recurring issue was that the candidate’s answer was being included inside the question. Another issue was that the extracted answer was sometimes shortened too aggressively.

The solution was to separate extraction responsibilities:

- The live coaching path extracts a fast question candidate only for the card.
- The post-session path uses a structured LLM prompt to extract all Q&A exchanges chronologically.
- The extraction prompt explicitly says to preserve the candidate’s full answer, including definitions, reasoning, examples, caveats, and explanatory setup.
- The answer ends only when the next interviewer question or transition begins.

This makes the session log more useful. It does not just store the shortest direct answer; it stores what the candidate actually said, lightly cleaned for transcription noise.

## Challenge 3: Avoiding Duplicate or Noisy Coaching Cards

Another challenge was duplicate cards. Live transcription can produce slightly different versions of the same question:

```text
How does linear regression work?
Can you briefly explain how linear regression works?
Can you explain how linear regression works?
```

Exact string matching was not enough. The app now creates a dedupe key from the important words in the question and compares similarity between cards. That prevents repeated STT variants from creating multiple coaching cards for the same question.

The app also filters out non-target topics. If the detected type is too generic or not relevant to DS, ML, AI, or System Design, the live card keeps listening instead of showing noise.

## Challenge 4: Making the Evaluation Fair

Evaluation had a subtle failure mode. The evaluator generated a benchmark answer so the candidate could learn, but the feedback sometimes looked like it was evaluating the benchmark rather than the candidate’s actual answer.

The fix was to make the evaluator contract explicit:

- The benchmark answer is only for learning.
- The hiring band and feedback must be based only on the candidate answer.
- Empty candidate answers should not receive credit.
- Generic benchmark-style feedback is rejected and falls back to a local candidate-answer heuristic.

This made the evaluation more faithful to what the candidate actually said.

## Local-First Runtime

The entire system can run locally on a Mac. That is a major part of the design.

Local execution gives three practical benefits:

- Privacy: interview audio and transcripts do not need to leave the machine.
- Latency: the coaching card can appear quickly enough to be useful.
- Control: models, prompts, and fine-tuned adapters can be swapped without changing the whole app.

The local setup uses Python, Gradio, SQLite, Hugging Face Transformers, PEFT, and MLX Whisper. For system audio capture on macOS, the app can use a local audio routing setup such as BlackHole.

## What Makes the Design Agentic

The app is agentic because it is not a single prompt wrapped in a UI. It is a pipeline of specialized steps, each with a clear role and state handoff.

The main agents and nodes are:

- Audio/transcription node: converts live audio into text.
- Question extraction logic: identifies the likely interviewer question boundary.
- Topic/pattern agent: classifies the question using the fine-tuned model.
- Coaching hint generator: creates short, useful pointers for the candidate.
- Persistence layer: stores sessions, exchanges, and evaluations in SQLite.
- Evaluation agent: reviews saved Q&A exchanges and produces structured feedback.

This modular design made it easier to improve one part without breaking the rest. For example, the live coaching path could be optimized for speed while the post-session extractor stayed more careful and LLM-driven.

## Conclusion

Interview Coach is designed around a simple idea: candidates do not need a model to answer for them, but they can benefit from timely reminders that help them structure their own thinking.

The project combines local speech-to-text, a fine-tuned 3B model, a general instruction model, SQLite persistence, and a Gradio interface into a practical interview practice system. The hardest part was not simply building a chatbot. It was making the coaching card appear at the right moment, extracting useful Q&A from noisy speech, and evaluating the candidate’s actual answer fairly.

The result is a local-first tool that helps during the interview and becomes a feedback system after the interview.
