from langchain.prompts import PromptTemplate

def build_prompt(
    exercise_type: str,
    level: int,
    difficulty: str = "medium",
    num_questions: int = 3,
    output_language: str = "English",
) -> PromptTemplate:
    # Everything here that depends on args is resolved NOW,
    # so the final template only includes {context} and {question}.
    base_header = f"""
You are an expert Chinese teacher who writes HSK-style exercises.
Target HSK level: HSK{level}.
Difficulty: {difficulty}.
Number of questions: {num_questions}.
Output language for answers and explanations: {output_language}.

CRITICAL RULES:
- Treat HSKK (Hanyu Shuiping Kouyu Kaoshi, oral exams) and HSK (written exams) as equivalent.
- Content labeled HSKK is fully acceptable as HSK context; never refuse or claim insufficiency due to HSKK/HSK differences.
- All exercise questions, prompts, and passages must be in Simplified Chinese.
- Explanations and answers should be in {output_language}, unless {output_language} is Chinese, in which case they should also be in Simplified Chinese.
- Prefer the provided context. If it is loosely related (e.g., oral prompts), still generate the requested HSK-style exercises.
- Keep answers concise and formatted exactly as requested for the exercise type.
""".strip()

    mcq = f"""
{base_header}

Create multiple-choice questions (single correct answer).
For each question include:
Q: <question in Simplified Chinese>
A) <option in Simplified Chinese>
B) <option in Simplified Chinese>
C) <option in Simplified Chinese>
D) <option in Simplified Chinese>
Answer: <letter> - <exact option text in Simplified Chinese>
Explain: <one short sentence in {output_language}>

Context:
{{context}}

User Request:
{{question}}
""".strip()

    fill_blank = f"""
{base_header}

Create fill-in-the-blank questions focusing on common HSK{level} vocab and grammar.
Format each item like:
Sentence: <sentence in Simplified Chinese with [____] blank>
Options: <A) ...  B) ...  C) ...  D) ...> (all options in Simplified Chinese)
Answer: <letter> - <exact word in Simplified Chinese>
Explain: <one short sentence in {output_language}>

Context:
{{context}}

User Request:
{{question}}
""".strip()

    reading_comp = f"""
{base_header}

Create a short reading passage appropriate for HSK{level} in Simplified Chinese, then 3–5 comprehension questions.
Format:
Passage:
<80–150 characters of Simplified Chinese; include pinyin if HSK1–3>

Questions:
1) <question in Simplified Chinese>
2) <question in Simplified Chinese>
...
Answers:
1) <answer in Simplified Chinese + brief explanation in {output_language}>
2) <answer in Simplified Chinese + brief explanation in {output_language}>
...

Context:
{{context}}

User Request:
{{question}}
""".strip()

    translation = f"""
{base_header}

Create short translation exercises for HSK{level}.
Format each item:
CN: <Chinese sentence in Simplified Chinese>
EN: <English translation target OR vice versa depending on request>
Answer: <model answer in requested target language>
Explain: <one short sentence in {output_language}>

Context:
{{context}}

User Request:
{{question}}
""".strip()

    templates = {
        "mcq": mcq,
        "fill_blank": fill_blank,
        "reading_comp": reading_comp,
        "translation": translation,
    }

    # Choose template and return LangChain PromptTemplate that ONLY needs {context} and {question}
    chosen = templates.get(exercise_type, mcq)
    return PromptTemplate(template=chosen, input_variables=["context", "question"])
