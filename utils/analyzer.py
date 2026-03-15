def extract_drawbacks(text):

    text_lower = text.lower()

    if "however" in text_lower or "but" in text_lower:
        return "The paper discusses certain limitations or challenges."

    return "Specific limitations are not clearly mentioned."


def suggest_solution():

    return "Future research can improve this work using larger datasets and advanced AI models."


def future_scope():

    return "This research can be extended using deep learning, real-time systems, and large-scale datasets."


def trending_indicator(date):

    try:
        year = int(date.split("-")[0])
    except:
        return "Standard Research Topic"

    if year >= 2020:
        return "🔥 Trending Research Area"
    elif year >= 2010:
        return "📈 Growing Research Topic"
    else:
        return "📚 Classical Research Area"
