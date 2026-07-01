from django.shortcuts import render
from textblob import TextBlob


def classify_sentiment(polarity):
    """Classify polarity score into Positive / Negative / Neutral."""
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


def classify_subjectivity(subjectivity):
    """Classify subjectivity score into a readable label."""
    if subjectivity < 0.3:
        return 'Mostly Factual'
    elif subjectivity < 0.6:
        return 'Mixed'
    else:
        return 'Highly Opinionated'


def get_sentence_sentiments(text):
    """Break text into sentences with individual polarity, for detail view."""
    blob = TextBlob(text)
    results = []
    for sentence in blob.sentences:
        results.append({
            'text': str(sentence),
            'polarity': round(sentence.sentiment.polarity, 3),
            'label': classify_sentiment(sentence.sentiment.polarity),
        })
    return results


def analyze_view(request):
    context = {}

    if request.method == 'POST':
        user_text = request.POST.get('user_text', '').strip()

        if user_text:
            blob = TextBlob(user_text)
            polarity = round(blob.sentiment.polarity, 3)
            subjectivity = round(blob.sentiment.subjectivity, 3)
            sentiment_label = classify_sentiment(polarity)
            subjectivity_label = classify_subjectivity(subjectivity)

            # For the polarity gauge: map [-1, 1] to [0, 100]
            polarity_percent = round((polarity + 1) / 2 * 100, 1)
            subjectivity_percent = round(subjectivity * 100, 1)

            word_count = len(user_text.split())
            sentence_data = get_sentence_sentiments(user_text)

            context.update({
                'submitted': True,
                'user_text': user_text,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment_label': sentiment_label,
                'subjectivity_label': subjectivity_label,
                'polarity_percent': polarity_percent,
                'subjectivity_percent': subjectivity_percent,
                'word_count': word_count,
                'sentence_data': sentence_data,
            })
        else:
            context['error'] = 'Please enter some text to analyze.'

    return render(request, 'analyzer/index.html', context)
