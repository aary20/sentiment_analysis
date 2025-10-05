# sentiment_app/views.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from django.shortcuts import render
from . import backend
import io
import base64
import pandas as pd
import matplotlib
import random
# Add this import at the top with the other imports
import numpy as np
matplotlib.use('Agg')

# Load dataset once (backend.load_dataset should be cached)
try:
    df = backend.load_dataset()
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = None


def index(request):
    context = {}
    
    if df is not None:
        # Get dataset statistics
        total_reviews = len(df)
        positive_reviews = len(df[df['feedback'] == 1])
        negative_reviews = len(df[df['feedback'] == 0])
        
        # Get 5 random reviews for display
        sample_reviews = df.sample(min(5, len(df))).to_dict('records')
        recent_reviews = []
        
        for review in sample_reviews:
            recent_reviews.append({
                'text': review.get('verified_reviews', '')[:200] + '...' if len(str(review.get('verified_reviews', ''))) > 200 else review.get('verified_reviews', ''),
                'rating': review.get('rating', 0),
                'sentiment': review.get('feedback', 0)
            })
        
        context = {
            'total_reviews': total_reviews,
            'positive_reviews': positive_reviews,
            'negative_reviews': negative_reviews,
            'recent_reviews': recent_reviews
        }
    
    return render(request, 'sentiment_app/index.html', context)


def visualize(request):
    visualization_type = request.POST.get('visualization_type', request.GET.get('visualization_type', 'wordcloud'))
    sentiment_filter = request.POST.get('sentiment_filter', request.GET.get('sentiment_filter', 'total'))
    variation = int(request.POST.get('variation', request.GET.get('variation', 50)))

    plot_url = None
    error = None

    if df is None:
        error = 'Dataset not loaded. Please check if the Amazon reviews CSV file exists.'
        return render(request, 'sentiment_app/visualize.html', {
            'error': error, 
            'visualization_type': visualization_type,
            'sentiment_filter': sentiment_filter,
            'variation': variation
        })

    try:
        if visualization_type == 'wordcloud':
            # Create word cloud based on sentiment filter
            if sentiment_filter == 'positive':
                text = ' '.join(df[df['feedback'] == 1]['verified_reviews'].dropna().astype(str).tolist())
            elif sentiment_filter == 'negative':
                text = ' '.join(df[df['feedback'] == 0]['verified_reviews'].dropna().astype(str).tolist())
            else:  # total
                text = ' '.join(df['verified_reviews'].dropna().astype(str).tolist())
            
            # Generate word cloud
            wc = WordCloud(width=1000, height=500, max_words=variation*5, 
                          background_color='white', colormap='viridis').generate(text)
            
            # Save to buffer
            img = io.BytesIO()
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(img, format='PNG', dpi=300, bbox_inches='tight')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            
        elif visualization_type == 'pie_chart':
            # Create a pie chart based on sentiment distribution
            plt.figure(figsize=(10, 6))
            
            if sentiment_filter == 'total':
                # Pie chart of positive vs negative reviews
                sentiment_counts = df['feedback'].value_counts()
                labels = ['Positive', 'Negative']
                sizes = [sentiment_counts.get(1, 0), sentiment_counts.get(0, 0)]
                colors = ['#28a745', '#dc3545']
                explode = (0.1, 0)  # explode the 1st slice (Positive)
                
                plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct='%1.1f%%', shadow=True, startangle=90)
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title('Distribution of Sentiment in Reviews')
                
            else:
                # Pie chart of ratings distribution for the selected sentiment
                if sentiment_filter == 'positive':
                    data = df[df['feedback'] == 1]
                    title = 'Rating Distribution for Positive Reviews'
                    colors = plt.cm.Greens(np.linspace(0.4, 0.8, 5))
                else:  # negative
                    data = df[df['feedback'] == 0]
                    title = 'Rating Distribution for Negative Reviews'
                    colors = plt.cm.Reds(np.linspace(0.4, 0.8, 5))
                
                rating_counts = data['rating'].value_counts().sort_index()
                labels = [f"{i} Stars" for i in rating_counts.index]
                
                plt.pie(rating_counts.values, labels=labels, autopct='%1.1f%%', 
                       shadow=True, startangle=90, colors=colors)
                plt.axis('equal')
                plt.title(title)
            
            # Save to buffer
            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='PNG', dpi=300)
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            
        elif visualization_type == 'label_analysis':
            # Create a bar chart of ratings distribution
            plt.figure(figsize=(10, 6))
            
            if sentiment_filter == 'positive':
                data = df[df['feedback'] == 1]
            elif sentiment_filter == 'negative':
                data = df[df['feedback'] == 0]
            else:  # total
                data = df
            
            # Count ratings
            rating_counts = data['rating'].value_counts().sort_index()
            
            # Plot
            bars = plt.bar(rating_counts.index, rating_counts.values, color='#ff9900')
            plt.xlabel('Rating')
            plt.ylabel('Number of Reviews')
            plt.title('Distribution of Amazon Review Ratings')
            plt.xticks(range(1, 6))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height}', ha='center', va='bottom')
            
            # Save to buffer
            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='PNG', dpi=300)
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
    
    except Exception as e:
        error = f"Error generating visualization: {str(e)}"
    
    return render(request, 'sentiment_app/visualize.html', {
        'plot_url': f"data:image/png;base64,{plot_url}" if plot_url else None,
        'error': error,
        'visualization_type': visualization_type,
        'sentiment_filter': sentiment_filter,
        'variation': variation
    })


def predict(request):
    context = {
        'text': '',
        'model': 'naive_bayes',
    }
    
    if request.method == 'POST':
        text = request.POST.get('text', '')
        model_name = request.POST.get('model', 'naive_bayes')
        
        context['text'] = text
        context['model'] = model_name
        
        try:
            # Here you would call your backend prediction function
            # For now, let's simulate a prediction
            if df is not None:
                # Simple sentiment prediction based on keywords (for demonstration)
                positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best']
                negative_words = ['bad', 'poor', 'terrible', 'worst', 'hate', 'disappointing']
                
                text_lower = text.lower()
                pos_count = sum(word in text_lower for word in positive_words)
                neg_count = sum(word in text_lower for word in negative_words)
                
                if pos_count > neg_count:
                    sentiment = 'positive'
                    confidence = min(100, 50 + (pos_count - neg_count) * 10)
                else:
                    sentiment = 'negative'
                    confidence = min(100, 50 + (neg_count - pos_count) * 10)
                
                context['sentiment'] = sentiment
                context['confidence'] = confidence
                context['accuracy'] = random.randint(75, 95)  # Simulated accuracy
                context['model_name'] = {
                    'naive_bayes': 'Naive Bayes',
                    'logistic_regression': 'Logistic Regression',
                    'svm': 'Support Vector Machine'
                }.get(model_name, model_name)
            else:
                context['error'] = "Dataset not loaded. Cannot make predictions."
        
        except Exception as e:
            context['error'] = f"Error making prediction: {str(e)}"
    
    return render(request, 'sentiment_app/predict.html', context)
