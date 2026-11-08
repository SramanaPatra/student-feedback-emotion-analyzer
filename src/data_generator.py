"""
Synthetic & Realistic Academic Student Feedback Dataset Generator.
Generates multi-course, multi-semester feedback text with emotion and sentiment labels.
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

FEEDBACK_TEMPLATES = {
    "Satisfaction": [
        "The professor explained complex topics with crystal clarity. Loved the hands-on labs!",
        "Excellent course structure and very approachable teaching assistants.",
        "The lectures were engaging and the real-world examples made everything click.",
        "One of the best classes I have taken this semester. The instructor is truly passionate.",
        "Clear slides, fair grading rubrics, and very insightful office hours.",
        "The practical assignments really helped solidify the theoretical concepts.",
        "Fantastic delivery and very well organized weekly schedule.",
        "I really appreciated how the professor took time to answer all questions patiently.",
        "The project helped me build confidence and practical coding skills.",
        "Great feedback on homework submissions, helped me improve week over week."
    ],
    "Confusion": [
        "The transitions between lecture topics were very abrupt and difficult to follow.",
        "Assignment instructions were vague and contradicted what was said in class.",
        "I am still confused about how the theoretical proofs connect to the code.",
        "The exam format was completely unexpected and differed from the review sheet.",
        "Unclear grading criteria on the group project made it confusing to prepare.",
        "Too many prerequisites assumed without a proper refresher or introductory reading.",
        "Pacing was way too fast in the second half of the semester, lost track of concepts.",
        "The textbook chapters assigned did not align well with the lecture slides.",
        "Homework questions were worded ambiguously, causing unnecessary misunderstanding.",
        "Not sure what will be covered on the final test, study guide is too generic."
    ],
    "Frustration": [
        "Grading was extremely delayed and returned with zero constructive comments.",
        "The lab computers crashed repeatedly and the software licenses were expired.",
        "Professor ignored student questions during class and seemed annoyed.",
        "Completely unfair midterm exam, questions were not taught in lectures at all.",
        "The workload was unreasonable compared to the credit hours assigned to this course.",
        "Communication on announcements was terrible, deadlines kept shifting without notice.",
        "Office hours were often cancelled at the last minute with no rescheduling.",
        "Group members did no work and the instructor refused to take action on peer reviews.",
        "Frustrating lack of guidance for the term project, we were left entirely on our own.",
        "Grading rubrics were changed after we already submitted our final papers."
    ],
    "Boredom": [
        "The lectures were just reading off slides verbatim with monotone delivery.",
        "Very repetitive exercises with little challenge or new information.",
        "Class attendance felt pointless since everything was just regurgitated from the book.",
        "The material was presented in a dry and uninspiring way.",
        "Not enough interactive discussions or practical applications to keep us engaged.",
        "Three-hour lectures with no breaks made it extremely difficult to stay attentive.",
        "Felt like a high school level class stretched out over an entire semester.",
        "Monotonous speech and very slow pacing made the lectures hard to stay awake through.",
        "Content felt outdated and disconnected from modern industry practices.",
        "Lacked enthusiasm and variety in teaching methodologies."
    ],
    "Stress": [
        "Having three major project milestones due in the exact same week was overwhelming.",
        "High anxiety throughout the semester due to surprise quizzes and harsh grading.",
        "The excessive workload severely impacted my mental health and sleep schedule.",
        "Impossible time limit on exams, barely anyone was able to finish half the questions.",
        "Felt constantly behind despite spending 25+ hours a week solely on this course.",
        "No breathing room between assignments, one due every 48 hours.",
        "The strict penalty for slight formatting mistakes caused unnecessary panic.",
        "Constant pressure and zero flexibility for medical emergencies or personal issues.",
        "Anxiety-inducing evaluation style that felt more punitive than educational.",
        "Overwhelming amount of weekly readings that could not realistically be completed."
    ]
}

EMOTION_TO_SENTIMENT = {
    "Satisfaction": "Positive",
    "Confusion": "Negative",
    "Frustration": "Negative",
    "Boredom": "Neutral",
    "Stress": "Negative"
}

def generate_dataset(n_samples=1500, output_csv="data/student_feedback.csv"):
    rows = []
    
    # Define trends over semesters for specific courses to show meaningful patterns
    # e.g., CS101 improves satisfaction over time
    # MATH105 has a spike in confusion in Fall 2025
    course_list = list(COURSES.keys())
    
    for _ in range(n_samples):
        course = random.choice(course_list)
        semester = random.choice(SEMESTERS)
        
        # Biased distribution depending on course & semester to create realistic trends
        if course == "CS101" and semester in ["Spring 2025", "Fall 2025", "Spring 2026"]:
            # Better satisfaction in later semesters
            weights = [0.55, 0.15, 0.10, 0.10, 0.10]
        elif course == "MATH105" and semester == "Fall 2025":
            # Confusion spike
            weights = [0.15, 0.45, 0.20, 0.05, 0.15]
        elif course == "CS202" and semester in ["Fall 2024", "Fall 2025"]:
            # High stress in fall project semester
            weights = [0.20, 0.15, 0.20, 0.10, 0.35]
        elif course == "ENG210":
            # High satisfaction and boredom
            weights = [0.40, 0.10, 0.10, 0.30, 0.10]
        else:
            weights = [0.30, 0.20, 0.20, 0.15, 0.15]
            
        emotion = random.choices(list(FEEDBACK_TEMPLATES.keys()), weights=weights)[0]
        base_text = random.choice(FEEDBACK_TEMPLATES[emotion])
        
        # Add slight natural variation
        prefixes = ["Overall, ", "In my opinion, ", "To be honest, ", "Personally, ", "", "", ""]
        suffixes = [" Would take again.", " Needs serious improvement.", " Hope changes are made.", "", "", ""]
        
        text = random.choice(prefixes) + base_text + random.choice(suffixes)
        sentiment = EMOTION_TO_SENTIMENT[emotion]
        
        rows.append({
            "course_id": course,
            "course_name": COURSES[course],
            "semester": semester,
            "feedback_text": text.strip(),
            "emotion": emotion,
            "sentiment": sentiment
        })
        
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Generated {len(df)} feedback entries at {output_csv}")
    return df

if __name__ == "__main__":
    generate_dataset()
