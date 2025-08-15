import os
import random
import re
import sys
from collections import defaultdict
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    res = defaultdict(int)
    #base value assigned to each page that the given page links to calculated with damping_factor/len(pages_listed)
    connected_pages = corpus[page]
    if len(connected_pages) == 0:
        connected_pages = set(corpus.keys())
    base_value = damping_factor/len(connected_pages)
    for connected_page in connected_pages:
        res[connected_page]=base_value
    
    #each page is incremented by (1-damping_factor)/len(all_pages)
    probability_of_all = (1-damping_factor)/len(corpus)
    for p, _ in corpus.items():
        res[p]+=probability_of_all

    return dict(res)

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #transition model gives probabilities at each turn
    #first choose random page from corpus, then call transition model, then repeat n times
    # random_page = list(corpus.keys())[random.randint(0,len(corpus)-1)]
    pages = list(corpus.keys())
    counts = {p: 0 for p in pages}
    
    current = random.choice(pages)
    counts[current] += 1
    
    for _ in range(1,n):
        probs = transition_model(corpus, current, damping_factor)
        weights = [probs[p] for p in pages]
        current = random.choices(pages, weights=weights, k=1)[0]
        counts[current] += 1

    return {p: counts[p] / n for p in pages}        
    
    # probabilities_for_each_page = transition_model(corpus, random_page, damping_factor)
    # chosen_page = np.random.choice(
    #     [page for page in probabilities_for_each_page.keys()],
    #     p=[probability for probability in probabilities_for_each_page.values()]
    # )
    
    # #choose a page with these probabilities
    # for _ in range(1,n):
    #     probabilities_for_each_page = transition_model(corpus, chosen_page, damping_factor)
    #     chosen_page = np.random.choice(
    #         [page for page in probabilities_for_each_page.keys()],
    #         p=[probability for probability in probabilities_for_each_page.values()]
    #     )
    #     #repeat process
    # return probabilities_for_each_page

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pages = list(corpus.keys())
    ranks = {p: 1 / N for p in pages}

    while True:
        new_ranks = {}
        for p in pages:
            base = (1 - damping_factor) / N
            link_sum = 0.0

            for q in pages:
                links = corpus[q]
                if len(links) == 0:
                    link_sum += ranks[q] / N
                elif p in links:
                    link_sum += ranks[q] / len(links)

            new_ranks[p] = base + damping_factor * link_sum

        if max(abs(new_ranks[p] - ranks[p]) for p in pages) <= 0.001:
            ranks = new_ranks
            break
        ranks = new_ranks

    total = sum(ranks.values())
    if total != 0:
        ranks = {p: ranks[p] / total for p in pages}
    return ranks


if __name__ == "__main__":
    main()
