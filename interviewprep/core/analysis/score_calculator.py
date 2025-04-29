def calculate_score(grammar_errors, found_keywords, total_keywords):
    grammar_score = max(0, 100 - (grammar_errors * 5))  # Lose 5 marks per error
    if total_keywords > 0:
        keyword_score = (len(found_keywords) / total_keywords) * 100
    else:
        keyword_score = 100

    final_score = (grammar_score * 0.6) + (keyword_score * 0.4)  # Weighted average
    return round(final_score, 2), grammar_score, keyword_score
