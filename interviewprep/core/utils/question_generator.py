import random

# Predefined questions for job roles
QUESTION_BANK = {
    'backend developer': [
        "Explain the concept of RESTful APIs.",
        "What are the differences between SQL and NoSQL databases?",
        "How do you optimize a slow database query?",
        "What is middleware in web development?",
        "Explain caching strategies in backend development."
    ],
    'frontend developer': [
        "What is the Virtual DOM?",
        "Explain the difference between HTML and JSX.",
        "What are promises and async/await in JavaScript?",
        "What are key features of React?",
        "How do you optimize website performance?"
    ],
    'data scientist': [
        "What is the difference between supervised and unsupervised learning?",
        "Explain overfitting and how to prevent it.",
        "What are some common data preprocessing techniques?",
        "Describe the bias-variance tradeoff.",
        "How do you select important features?"
    ],
}

def generate_questions_based_on_role(role, num_questions=5):
    questions = QUESTION_BANK.get(role.lower(), [])
    return random.sample(questions, min(num_questions, len(questions)))
