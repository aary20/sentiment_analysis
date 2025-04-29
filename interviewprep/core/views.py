from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import os
from .forms import ResumeUploadForm
from pyresparser import ResumeParser
from .utils.question_generator import generate_questions_based_on_role
from .analysis.grammar_check import check_grammar
from .analysis.keyword_check import check_keywords
from .analysis.score_calculator import calculate_score
from django.shortcuts import render, redirect
from .models import UserAnswer
from django.contrib.auth.decorators import login_required


def home(request):
    return render(request, 'core/home.html')
# Create your views here.

@login_required
def dashboard(request):
    user_answers = UserAnswer.objects.filter(user=request.user).order_by('-timestamp')

    return render(request, 'core/dashboard.html', {'user_answers': user_answers})

def register_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('register')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect('register')

        user = User.objects.create_user(username=username, password=password)
        login(request, user)
        return redirect('dashboard')

    return render(request, 'register.html')


def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password.")
            return redirect('login')

    return render(request, 'login.html')


def logout_user(request):
    logout(request)
    return redirect('login')

def upload_resume(request):
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            resume_obj = form.save(commit=False)
            resume_obj.user = request.user
            resume_obj.save()

            # Analyze the uploaded resume
            resume_path = resume_obj.resume.path
            data = ResumeParser(resume_path).get_extracted_data()

            # Simple Analysis Example
            analysis = ""
            if data:
                if 'skills' in data and data['skills']:
                    skills = ", ".join(data['skills'])
                    analysis += f"Skills found: {skills}\n"
                else:
                    analysis += "No skills detected. Add more skills.\n"

                if 'experience' in data and data['experience']:
                    if data['experience'] < 2:
                        analysis += "Less experience detected. Try gaining more projects or internships.\n"
                else:
                    analysis += "Experience not detected. Mention your work clearly.\n"

            return render(request, 'users/resume_analysis.html', {'data': data, 'analysis': analysis})

    else:
        form = ResumeUploadForm()
    return render(request, 'users/resume_upload.html', {'form': form}) 

def generate_questions(request):
    questions = []
    if request.method == 'POST':
        selected_role = request.POST.get('role')
        if selected_role:
            questions = generate_questions_based_on_role(selected_role)
            request.session['generated_questions'] = questions
            return redirect('mock_interview')  # Go to mock interview

    return render(request, 'core/generate_questions.html', {'questions': questions})
def mock_interview(request):
    questions = request.session.get('generated_questions', [])
    total_questions = len(questions)
    current_question_index = request.session.get('current_question_index', 0)

    if request.method == 'POST':
        user_answers = request.session.get('user_answers', [])
        user_answer = request.POST.get('answer')
        user_answers.append(user_answer)
        request.session['user_answers'] = user_answers

        current_question_index += 1
        request.session['current_question_index'] = current_question_index

        if current_question_index >= total_questions:
            return redirect('thank_you')  # After last question, go to thank you page

    if current_question_index < total_questions:
        question = questions[current_question_index]
    else:
        question = None

    context = {
        'question': question,
        'current_index': current_question_index + 1,
        'total_questions': total_questions
    }
    return render(request, 'core/mock_interview.html', context)

@login_required
def view_report(request, id):
    answer = UserAnswer.objects.get(id=id, user=request.user)
    return render(request, 'core/report.html', {'answer': answer})


def reports(request):
    return render(request, 'core/reports.html')


EXPECTED_KEYWORDS = ["Python", "Django", "API", "Database", "Authentication"]

@login_required
def analyze_answer(request):
    if request.method == "POST":
        answer = request.POST.get("answer")
        question = request.POST.get("question")  # Pass from form

        grammar_errors, grammar_matches = check_grammar(answer)
        found_keywords, missing_keywords = check_keywords(answer, EXPECTED_KEYWORDS)
        final_score, grammar_score, keyword_score = calculate_score(
            grammar_errors, found_keywords, len(EXPECTED_KEYWORDS)
        )

        # Save to DB
        UserAnswer.objects.create(
            user=request.user,
            question=question,
            answer=answer,
            grammar_errors=grammar_errors,
            grammar_score=grammar_score,
            keyword_score=keyword_score,
            final_score=final_score
        )

        context = {
            'answer': answer,
            'question': question,
            'grammar_errors': grammar_errors,
            'grammar_matches': grammar_matches,
            'found_keywords': found_keywords,
            'missing_keywords': missing_keywords,
            'final_score': final_score,
            'grammar_score': grammar_score,
            'keyword_score': keyword_score,
        }

        return render(request, 'core/feedback.html', context)
    
def thank_you(request):
    return render(request, 'core/thank_you.html')

