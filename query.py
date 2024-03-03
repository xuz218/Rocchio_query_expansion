import sys
import requests
import nltk
import math

nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from collections import Counter


class GoogleSearchAPI:
    def __init__(self, api_key, engine_id, precision, query):
        self.api_key = api_key
        self.engine_id = engine_id
        self.precision = float(precision)
        self.query = query

    def call_google_api(self):
        """
        Retrieve information from google search engine.

        :param api_key, engine_id, query
        :return results: the original results from google search
        """
        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.engine_id}&q={self.query}"
        response = requests.get(url)
        results = response.json()
        return results

    def getFormattedData(self, results):
        """
        Formatted the retrieved information and split into title, link, snippet.

        :param results : the original results from google search
        :return formatted_results :  list of dictionaries containing unlabeled search results and separated by title, link, snippet.
        """
        formatted_results = {"items": []}
        for item in results["items"]:
            formatted_result = {
                "title": item["title"],
                "link": item["link"],
                "snippet": item["snippet"]
            }
            formatted_results["items"].append(formatted_result)
        return formatted_results

    def tokenize(self, text):
        english_stopwords = stopwords.words('english')
        token = re.findall(r'\w+', text.lower())
        tokens_wo_stopwords = [t for t in token if len(t) >= 2 and t not in english_stopwords]
        return tokens_wo_stopwords

    def getLabeledData(self, formatted_results):
        """
        Retrieve the feedback from users with labeled results.

        :param formatted_results: list of dictionaries containing unlabeled search results
        :return labeled_results: list of dictionaries containing labeled search results
        """

        labeled_results = []
        print("Recognized True Labels: y, Y, t, T")
        print("Recognized False Labels: n, N, f, F")
        print("-----------------")
        for item in formatted_results["items"]:
            print("Title:", item["title"])
            print("Link:", item["link"])
            print("Summary:", item["snippet"])
            relevance = "T"

            # Prompt for the user's label (in a real command-line environment)
            label_input = input("Is this item relevant? (true/false): ").lower()
            # Check if input starts with 't' or 'y'
            if label_input[0] not in ['t', 'y', 'Y', 'T']:
                relevance = "F"
            print("-----------------")

            labeled_results.append({**item, "relevance": relevance})

        for doc in labeled_results:
            tokens = self.tokenize(doc["snippet"])
            # Count the frequency of each term in the document
            doc["df"] = Counter(tokens)

        return labeled_results

    def getPrecisionScore(self, labeled_results):
        """
        Calculate precision score based on labeled results.

        :param labeled_results: list of dictionaries containing labeled search results
        :return: precision score
        """
        precision_score = sum(1 for item in labeled_results if item['relevance'] == 'T') / len(labeled_results)
        return precision_score

    def build_inverted_list(self, labeled_results):
        inverted_list = {}
        for idx, item in enumerate(labeled_results):
            title_tokens = self.tokenize(item["title"])
            summary_tokens = self.tokenize(item["snippet"])

            tokens = set(title_tokens + summary_tokens)

            for token in tokens:
                if token not in inverted_list:
                    inverted_list[token] = set()
                inverted_list[token].add(idx)

        return inverted_list

    def rocchio(self, invertedList, labeled_results):
        # Constants for the Rocchio algorithm
        alpha = 1
        beta = 0.75
        gamma = 0.15

        query_weights = {term: 1.0 for term in self.query.split()}
        weights = {term: 0.0 for term in invertedList}

        relevant_doc = {}
        irrelevant_doc = {}

        numRel = 0
        numNonrel = 0

        # Accumulate document frequencies for terms in relevant and non-relevant documents
        for item in labeled_results:
            if item["relevance"] == 'T':
                numRel += 1
                for term, freq in item["df"].items():
                    relevant_doc[term] = relevant_doc.get(term, 0) + freq
            else:
                numNonrel += 1
                for term, freq in item["df"].items():
                    irrelevant_doc[term] = irrelevant_doc.get(term, 0) + freq

        # Rocchio algorithm
        for term, docs in invertedList.items():
            idf = math.log((float(len(labeled_results)) / (1 + len(docs))), 10)  # Added 1 to avoid division by zero

            for idx in docs:
                if labeled_results[idx]['relevance'] == 'T':
                    weights[term] += beta * idf * (float(relevant_doc.get(term, 0)) / (numRel if numRel > 0 else 1))
                else:
                    weights[term] -= gamma * idf * (
                                float(irrelevant_doc.get(term, 0)) / (numNonrel if numNonrel > 0 else 1))

            if term in query_weights:
                query_weights[term] = alpha * query_weights[term] + weights[term]
            elif weights[term] > 0:
                query_weights[term] = weights[term]

        return query_weights

    def expand_query_list(self, query_weights):
        """
        Project next two search query in accordance with query weights

        :param query_weights: weights for words
        :return: augmented_words
        """
        original_query = self.query.split()
        n = len(original_query)
        augmented_words = [word for word, score in sorted(query_weights.items(), key=lambda x: x[1], reverse=True)[:n+2]]
        return augmented_words


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Please enter: python3 query.py <API_KEY> <ENGINE_ID> <PRECISION> <QUERY>")
        sys.exit()

    _, api_key, engine_id, precision, query = sys.argv
    if float(precision) <= 0.000001:
        print("The precision required is 0. Program terminated.")
        sys.exit()
    search_api = GoogleSearchAPI(api_key, engine_id, float(precision), query)
    results = search_api.call_google_api()
    formatted_results = search_api.getFormattedData(results)
    labeled_results = search_api.getLabeledData(formatted_results)
    precision_score = search_api.getPrecisionScore(labeled_results)
    print(precision_score)

    while precision_score < float(precision):
        if float(precision_score) <= 0.000001:
            print("The precision reaches 0. Program terminated.")
            sys.exit()
        inverted_list = search_api.build_inverted_list(labeled_results)
        query_weights = search_api.rocchio(inverted_list, labeled_results)
        new_query = search_api.expand_query_list(query_weights)
        print(f"The new query is {new_query}")
        search_api.query = " ".join(new_query)

        results = search_api.call_google_api()
        formatted_results = search_api.getFormattedData(results)
        labeled_results = search_api.getLabeledData(formatted_results)
        precision_score = search_api.getPrecisionScore(labeled_results)
        print(precision_score)

    print("Threshold reached.")
    sys.exit()
