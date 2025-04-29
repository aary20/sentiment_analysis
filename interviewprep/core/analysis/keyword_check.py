def check_keywords(answer, expected_keywords):
    found_keywords = []
    missing_keywords = []

    answer_lower = answer.lower()

    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    return found_keywords, missing_keywords
