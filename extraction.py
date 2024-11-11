import pandas as pd


def get_hum_reviews(data_frame):
    keys = ['review#1', 'review#2', 'review#3', '']
    return [data_frame[key] for key in keys]


def get_llm_reviews(data_frame):
    keys = 'claude_opus', 'gpt4', 'gemini_pro_1.5'
    return [data_frame[key] for key in keys]


def get_example_comparison():
    return {{
        "A1-B2": {
            {"rationale": "<explain why A1 and B2 are nearly identical>", "similarity": "<5-10, only an integer>"}},
    }}


def hit_rate(set_a, set_b):
    """ Calculates Hit Rate.

    Hit Rate = |A∩B| / |A|

    Returns the proportion of comments in set A that match those in set B.
    Pairwise overlap of both GPT-4 vs. Human and Human vs. Human.

    Expectation:
    - One third (30.85%) of GPT-4 raised comments overlap with the comments from an individual reviewer.

    """
    matches = sum(set_a['comments'].isin(set_b['comments']))
    return matches / len(set_a['comments'])


if __name__ == '__main__':

    # Get papers which have both human and LLM reviews
    raw_llm_data = pd.read_json('./data/ReviewCritique_LLM.jsonl', lines=True)
    raw_human_data = pd.read_json('./data/ReviewCritique.jsonl', lines=True)
    human_data = raw_human_data[raw_human_data['title'].isin(raw_llm_data['title'])]
    llm_data = raw_llm_data[raw_llm_data['title'].isin(raw_human_data['title'])]

    # Ensure Same Contents
    papers = list(llm_data['title'])
    len(human_data)
    len(llm_data)

    # Get Human & LLM Reviews for each paper
    reviews = {paper: {} for paper in papers}
    for i, paper in enumerate(papers):
        print(f'Paper: {paper}')
        reviews[paper].update({'human': get_hum_reviews(human_data.iloc[i])})
        reviews[paper].update({'llm': get_llm_reviews(llm_data.iloc[i])})

    # Perform Cross Correlation between all Humans & all LLMS
    comparison_scores = {paper: {} for paper in papers}
    for review in reviews.items():
        for author in range(6):
            comparison_scores[review[0]] = review[1]['human'][author]

    """ This is unfinished and most likely unnecessary. """

