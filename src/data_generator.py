"""
Clean, High-Fidelity Academic Student Feedback Dataset Generator.
Generates multi-course, multi-semester student reviews across 5 distinct emotion categories
with semantic emotion-coherent phrasing and longitudinal semester trends.
"""

import os
import random
import pandas as pd

COURSES = {
    "CS101": "Introduction to Computer Science",
    "CS202": "Data Structures and Algorithms",
    "MATH105": "Linear Algebra and Calculus",
    "PHY101": "Engineering Physics",
    "ENG210": "Technical Communication"
}

SEMESTERS = [
    "Spring 2024",
    "Fall 2024",
    "Spring 2025",
    "Fall 2025",
    "Spring 2026"
]

EMOTIONS = ["Satisfaction", "Confusion", "Frustration", "Boredom", "Stress"]

EMOTION_TO_SENTIMENT = {
    "Satisfaction": "Positive",
    "Confusion": "Negative",
    "Frustration": "Negative",
    "Boredom": "Neutral",
    "Stress": "Negative"
}

ACADEMIC_COMMENTS = {
    "Satisfaction": [
        "The professor explained complex topics with crystal clarity and great enthusiasm.",
        "Loved the hands-on coding labs! The practical work made the theory easy to digest.",
        "Teaching assistants were exceptionally supportive and patient during weekly office hours.",
        "The course structure was logical, well-paced, and very rewarding.",
        "Clear slides, transparent grading rubrics, and very insightful lecture demonstrations.",
        "One of the best classes I have taken at university. The instructor is truly passionate.",
        "The assignments challenged us to think critically while offering fair guidelines.",
        "Great feedback on homework submissions, helped me improve steadily each week.",
        "Interactive classroom discussions kept everyone motivated and eager to learn.",
        "The textbook and recommended readings aligned seamlessly with the exam content.",
        "I gained immense confidence in problem-solving and programming from this course.",
        "The instructor took time to answer all questions thoroughly without rushing.",
        "Real-world case studies made abstract concepts immediately practical and exciting.",
        "Fair exams that accurately reflected what was practiced during lectures.",
        "Well organized canvas modules and prompt email communications throughout the term.",
        "Brilliant lecture delivery! Even the most challenging algorithms were broken down simply.",
        "The final project gave me something genuinely impressive to put on my portfolio.",
        "Highly engaging presentation style with memorable intuitive analogies.",
        "The grading criteria were fair, transparent, and returned in a timely manner.",
        "Outstanding professor who cares deeply about student learning and success.",
        "Awesome class, I learned so much and felt truly supported throughout the term.",
        "Really appreciated the practical live coding sessions and clear explanations.",
        "The professor was approachable, knowledgeable, and always encouraged questions.",
        "Top-notch course materials and very well structured weekly homework sets."
    ],
    "Confusion": [
        "The assignment specifications were extremely vague and contradicted the lecture slides.",
        "Transitions between topics were abrupt and left most of the class baffled.",
        "I could not understand how the mathematical proofs connected to the practical code.",
        "The midterm exam format was totally unexpected and differed from the study guide.",
        "Unclear grading criteria on the group project made it confusing to prepare deliverables.",
        "Pacing was way too fast in the second half of the term, leaving basic concepts behind.",
        "The problem sets assumed advanced prerequisites that were never covered or reviewed.",
        "Homework questions were worded ambiguously, causing unnecessary confusion.",
        "Lecture slides had dense code snippets with no commentary or explanation.",
        "Not sure what will be covered on the final examination, the syllabus is contradictory.",
        "Different teaching assistants gave conflicting advice on how to solve the project.",
        "The professor jumped straight into complicated edge cases before explaining the fundamentals.",
        "Notation used in class differed completely from the textbook equations, very confusing.",
        "Rubrics were posted only after submission, making expectations impossible to gauge.",
        "I struggled to follow the lectures because the board work was chaotic and unreadable.",
        "Concepts were introduced without context or intuition, leaving us lost.",
        "Project guidelines kept changing verbally during class with no written updates.",
        "The review session did not address questions on the hardest problem sets.",
        "Uncertain whether we are allowed to use external libraries due to vague policies.",
        "I never knew what the instructor expected on open-ended essay questions.",
        "Assignment instructions were vague and contradicted what was said in lecture slides.",
        "The textbook proofs were convoluted and the lecture explanations made it more confusing.",
        "I felt completely lost during the second half because concepts were rushed without review."
    ],
    "Frustration": [
        "Grading was delayed by six weeks and returned with zero constructive feedback.",
        "The laboratory computers crashed repeatedly and essential software licenses were expired.",
        "The professor dismissed student questions during class and seemed annoyed by inquiries.",
        "Unfair exam with several trick questions that were never taught in lectures or labs.",
        "Workload was totally disproportionate to the credit hours assigned to this course.",
        "Communication was atrocious, deadlines were shifted at the eleventh hour with no notice.",
        "Office hours were frequently cancelled without rescheduling, leaving us stranded.",
        "Team members contributed zero effort and the instructor refused to intervene on peer evals.",
        "Frustrating lack of guidance for the term paper, expectations were a moving target.",
        "Grading penalties were applied retroactively for formatting rules never mentioned in the rubric.",
        "Teaching assistants were rude and unhelpful when asked for clarification on test scores.",
        "Auto-grader had fatal bugs that caused valid submissions to receive zero marks.",
        "The instructor was completely inaccessible via email and ignored discussion board posts.",
        "Felt like the course was deliberately designed to trick students rather than educate.",
        "Disorganized course coordination between the lead instructor and the laboratory sections.",
        "No sample exams or practice problems were provided despite repeated requests.",
        "Feedback on previous assignments was returned only after the final exam had already taken place.",
        "Lecture recordings were frequently muted or corrupted and never re-uploaded.",
        "Disregard for student feedback and refusal to acknowledge clear mistakes on exam answer keys.",
        "Administrative chaos from week one until final grades were posted.",
        "Grading was delayed for weeks and the exam had questions never covered in class.",
        "Extremely frustrating experience with unhelpful TAs and broken auto-grading test cases."
    ],
    "Boredom": [
        "The lectures were just the professor reading verbatim off PowerPoint slides in a monotone drone.",
        "Extremely dry and uninspiring presentation style that made staying awake a chore.",
        "Repetitive exercises with almost zero intellectual challenge or novel insights.",
        "Attending class felt pointless because every word was just recited from the textbook.",
        "Three-hour lecture blocks without breaks or interactive questions were mind-numbing.",
        "Lacked modern relevance, the curriculum felt outdated and disconnected from industry.",
        "No passion or enthusiasm conveyed for the subject matter throughout the entire semester.",
        "Slow-paced discussions that dragged on basic definitions for weeks.",
        "Slide decks were crammed with walls of text and read in an unenthusiastic tone.",
        "Zero hands-on activities, just passive listening to endless theoretical ramblings.",
        "Very monotonous delivery, every single class followed the exact same repetitive formula.",
        "Felt like a high school level curriculum stretched artificially across an entire term.",
        "Lecturer seemed bored by their own material, which made it impossible for students to care.",
        "Unengaging teaching methods that failed to ignite any curiosity or interest.",
        "Content was excessively rudimentary with no depth or stimulating discussions.",
        "Just copying equations from notes to the chalkboard without explaining why they matter.",
        "The material could have been learned in three days of self-study rather than four months.",
        "Long uninterrupted monologues with no opportunity for questions or debate.",
        "Uninspired delivery that drained the excitement from what should have been a fun topic.",
        "Very bland, routine, and monotonous classroom experience.",
        "The lectures were very boring, just reading slides verbatim with zero student interaction."
    ],
    "Stress": [
        "Having three major project deadlines and an exam in the exact same week was overwhelming.",
        "Constant anxiety throughout the semester caused by surprise quizzes and harsh deductions.",
        "The brutal workload severely compromised my mental health, physical wellness, and sleep.",
        "Unrealistic exam time constraints where barely anyone could finish more than half the test.",
        "Felt relentlessly behind despite spending over 30 hours a week solely on this course.",
        "Zero breathing room between assignments, new deliverables due every 48 hours without pause.",
        "Extreme panic over minor syntax errors resulting in automatic failing grades on programming labs.",
        "High pressure atmosphere that felt punitive and hostile rather than supportive.",
        "Impossible workload that forced students to sacrifice other classes just to barely pass.",
        "Excessive weekly reading assignments consisting of hundreds of pages of dense jargon.",
        "Sleep deprivation and chronic burnout became the norm for everyone in this section.",
        "The threat of failing was constantly held over our heads rather than encouragement.",
        "Sudden difficult pop quizzes with no advance warning or preparation window.",
        "Harsh curving policies that pitted students against each other in toxic competition.",
        "I felt on the verge of a panic attack before every single laboratory practical.",
        "No extensions granted even under documented illness and emergencies.",
        "Absurdly heavy weekly homework sets requiring days of continuous non-stop work.",
        "The stress and exhaustion made it nearly impossible to retain any actual knowledge.",
        "Suffocating amount of deliverables with zero time to absorb or review previous topics.",
        "Dreading coming to class every week due to high stakes interrogations and pop quizzes.",
        "Having three major deadlines in the exact same week was completely unmanageable."
    ]
}

COHERENT_AFFIXES = {
    "Satisfaction": {
        "prefixes": ["", "Overall, ", "In my experience, ", "Looking back, ", "To be fair, ", "Frankly, "],
        "suffixes": ["", " Highly recommended!", " Would definitely take again.", " Truly appreciated.", " Best experience ever.", " Fantastic professor!"]
    },
    "Confusion": {
        "prefixes": ["", "Honestly, ", "In my experience, ", "Looking back, ", "To be fair, ", "Throughout the term, "],
        "suffixes": ["", " Very confusing and unclear.", " Left most students lost.", " Needs much better explanation.", " Hope expectations are clarified."]
    },
    "Frustration": {
        "prefixes": ["", "Frankly, ", "In my opinion, ", "Unfortunately, ", "Looking back, ", "To be honest, "],
        "suffixes": ["", " Very disappointed with the disorganization.", " Completely unorganized.", " Never taking this again.", " Needs a serious overhaul."]
    },
    "Boredom": {
        "prefixes": ["", "Honestly, ", "In my opinion, ", "Personally, ", "Looking back, "],
        "suffixes": ["", " Very dull and hard to stay awake.", " Needs more interactive discussion.", " Monotonous delivery.", " Felt like a chore."]
    },
    "Stress": {
        "prefixes": ["", "Honestly, ", "In my experience, ", "Personally, ", "Without a doubt, "],
        "suffixes": ["", " Caused extreme burnout and anxiety.", " Unmanageable workload.", " Severely impacted my mental health.", " Way too stressful."]
    }
}

def generate_dataset(n_samples=3000, output_csv="data/student_feedback.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = []
    
    course_biases = {
        "CS101": {
            "Spring 2024": {"Satisfaction": 0.45, "Confusion": 0.25, "Frustration": 0.10, "Boredom": 0.10, "Stress": 0.10},
            "Fall 2024":   {"Satisfaction": 0.40, "Confusion": 0.30, "Frustration": 0.10, "Boredom": 0.10, "Stress": 0.10},
            "Spring 2025": {"Satisfaction": 0.50, "Confusion": 0.20, "Frustration": 0.10, "Boredom": 0.10, "Stress": 0.10},
            "Fall 2025":   {"Satisfaction": 0.55, "Confusion": 0.15, "Frustration": 0.10, "Boredom": 0.10, "Stress": 0.10},
            "Spring 2026": {"Satisfaction": 0.60, "Confusion": 0.15, "Frustration": 0.08, "Boredom": 0.07, "Stress": 0.10}
        },
        "CS202": {
            "Spring 2024": {"Satisfaction": 0.35, "Confusion": 0.25, "Frustration": 0.15, "Boredom": 0.05, "Stress": 0.20},
            "Fall 2024":   {"Satisfaction": 0.25, "Confusion": 0.25, "Frustration": 0.20, "Boredom": 0.05, "Stress": 0.25},
            "Spring 2025": {"Satisfaction": 0.30, "Confusion": 0.20, "Frustration": 0.18, "Boredom": 0.07, "Stress": 0.25},
            "Fall 2025":   {"Satisfaction": 0.20, "Confusion": 0.20, "Frustration": 0.32, "Boredom": 0.08, "Stress": 0.20},
            "Spring 2026": {"Satisfaction": 0.38, "Confusion": 0.20, "Frustration": 0.15, "Boredom": 0.07, "Stress": 0.20}
        },
        "MATH105": {
            "Spring 2024": {"Satisfaction": 0.25, "Confusion": 0.35, "Frustration": 0.15, "Boredom": 0.10, "Stress": 0.15},
            "Fall 2024":   {"Satisfaction": 0.30, "Confusion": 0.30, "Frustration": 0.15, "Boredom": 0.10, "Stress": 0.15},
            "Spring 2025": {"Satisfaction": 0.20, "Confusion": 0.42, "Frustration": 0.15, "Boredom": 0.08, "Stress": 0.15},
            "Fall 2025":   {"Satisfaction": 0.35, "Confusion": 0.25, "Frustration": 0.15, "Boredom": 0.10, "Stress": 0.15},
            "Spring 2026": {"Satisfaction": 0.40, "Confusion": 0.22, "Frustration": 0.13, "Boredom": 0.10, "Stress": 0.15}
        },
        "PHY101": {
            "Spring 2024": {"Satisfaction": 0.30, "Confusion": 0.20, "Frustration": 0.15, "Boredom": 0.20, "Stress": 0.15},
            "Fall 2024":   {"Satisfaction": 0.25, "Confusion": 0.15, "Frustration": 0.15, "Boredom": 0.30, "Stress": 0.15},
            "Spring 2025": {"Satisfaction": 0.35, "Confusion": 0.20, "Frustration": 0.15, "Boredom": 0.15, "Stress": 0.15},
            "Fall 2025":   {"Satisfaction": 0.30, "Confusion": 0.20, "Frustration": 0.15, "Boredom": 0.20, "Stress": 0.15},
            "Spring 2026": {"Satisfaction": 0.38, "Confusion": 0.18, "Frustration": 0.14, "Boredom": 0.15, "Stress": 0.15}
        },
        "ENG210": {
            "Spring 2024": {"Satisfaction": 0.40, "Confusion": 0.15, "Frustration": 0.15, "Boredom": 0.20, "Stress": 0.10},
            "Fall 2024":   {"Satisfaction": 0.45, "Confusion": 0.15, "Frustration": 0.12, "Boredom": 0.18, "Stress": 0.10},
            "Spring 2025": {"Satisfaction": 0.50, "Confusion": 0.12, "Frustration": 0.10, "Boredom": 0.18, "Stress": 0.10},
            "Fall 2025":   {"Satisfaction": 0.48, "Confusion": 0.12, "Frustration": 0.12, "Boredom": 0.18, "Stress": 0.10},
            "Spring 2026": {"Satisfaction": 0.52, "Confusion": 0.12, "Frustration": 0.10, "Boredom": 0.16, "Stress": 0.10}
        }
    }

    per_combo = max(20, n_samples // (len(COURSES) * len(SEMESTERS)))

    for course_id, course_name in COURSES.items():
        for sem in SEMESTERS:
            weights = course_biases[course_id][sem]
            emotions_list = list(weights.keys())
            probs = list(weights.values())

            for _ in range(per_combo):
                chosen_emotion = random.choices(emotions_list, weights=probs, k=1)[0]
                base_text = random.choice(ACADEMIC_COMMENTS[chosen_emotion])
                
                affix = COHERENT_AFFIXES[chosen_emotion]
                pfx = random.choice(affix["prefixes"])
                sfx = random.choice(affix["suffixes"])
                full_text = f"{pfx}{base_text}{sfx}".strip()

                sentiment = EMOTION_TO_SENTIMENT[chosen_emotion]
                rows.append({
                    "course_id": course_id,
                    "course_name": course_name,
                    "semester": sem,
                    "feedback_text": full_text,
                    "emotion": chosen_emotion,
                    "sentiment": sentiment
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Generated {len(df)} coherent student feedback samples saved to {output_csv}")
    return df

if __name__ == "__main__":
    generate_dataset()
