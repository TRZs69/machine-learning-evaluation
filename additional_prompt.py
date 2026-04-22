# prompts.py

COURSE = """
MATERI REFERENSI :

1. Pengantar HCI: Interaksi manusia & komputer. Fokus: usability, usefulness, satisfaction.

2. Usability Heuristics (Nielsen): 10 prinsip (Visibility, Match real world, User control, Consistency, Error prevention, Recognition, Flexibility, Aesthetic, Help users, Documentation).

3. Cognitive Load Theory: Intrinsic (kompleksitas materi), Extraneous (desain buruk), Germane (proses belajar). Desain harus minimize extraneous load.

4. User-Centered Design (UCD): Iteratif melibatkan user (Research -> Design -> Test -> Iterate).

5. Fitts's Law: Waktu gerak pointer tergantung jarak & ukuran target. Target besar & dekat = lebih cepat.

6. Norman's Action Cycle: 7 tahap (Goal -> Intention -> Action -> Execute -> Perceive -> Interpret -> Evaluate). Gap Execution & Evaluation.

7. Mental Model: Representasi user tentang cara kerja sistem. Desain harus match dengan mental model user.

8. Accessibility (WCAG): Perceivable, Operable, Understandable, Robust.

9. User Experience (UX): Meliputi emosi, estetika, dan nilai praktis dari penggunaan sistem. 
"""

INSTRUCTION = """
You are an Educational Evaluator. Rate the response 1-5.

5 = Accurate to reference material, gives concrete examples, personalized to student level, well-structured

4 = Accurate and helpful, minor details may be missing, good personalization

3 = Factually correct but generic, lacks depth or personalization

2 = Has factual errors, not personalized, too brief or unfocused

1 = Incorrect, misleading, or harmful

Be fair, honest, and discriminating.
"""