from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "interviews.db"
GENERAL_LLM_MODEL = os.environ.get("INTERVIEW_COACH_GENERAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MODEL_PATH = GENERAL_LLM_MODEL
EVALUATION_MODEL_PATH = GENERAL_LLM_MODEL
WHISPER_MODEL = "mlx-community/whisper-small-mlx"
STREAMING_WHISPER_MODEL = "mlx-community/whisper-tiny"
HF_SPACE_MODE = bool(os.environ.get("SPACE_ID")) or os.environ.get("INTERVIEW_COACH_RUNTIME") == "space"
if HF_SPACE_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
STT_BACKEND = os.environ.get("INTERVIEW_COACH_STT_BACKEND", "transformers" if HF_SPACE_MODE else "mlx")
HF_WHISPER_MODEL = os.environ.get("INTERVIEW_COACH_HF_WHISPER_MODEL", "openai/whisper-small")
APP_HOST = os.environ.get("INTERVIEW_COACH_HOST", "0.0.0.0" if HF_SPACE_MODE else "127.0.0.1")
APP_PORT = 7860
<<<<<<< HEAD
TOPIC_PATTERN_MODEL = os.environ.get("INTERVIEW_COACH_TOPIC_PATTERN_MODEL", "build-small-hackathon/interview-coach-3b")
=======
TOPIC_PATTERN_MODEL = os.environ.get("INTERVIEW_COACH_TOPIC_PATTERN_MODEL", "vadirajkrishna/interview-coach-3b")
>>>>>>> c0f39ad (initial commit - InterviewCopilotLocal)
TOPIC_PATTERN_BASE_MODEL = os.environ.get("INTERVIEW_COACH_TOPIC_PATTERN_BASE_MODEL", GENERAL_LLM_MODEL)
HF_LOCAL_FILES_ONLY = os.environ.get("INTERVIEW_COACH_HF_LOCAL_FILES_ONLY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
USE_TOPIC_PATTERN_MODEL = os.environ.get("INTERVIEW_COACH_USE_TOPIC_PATTERN_MODEL", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
