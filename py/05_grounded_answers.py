"""
Lesson 5: grounding, citations and "I don't know".

Retrieval found the right chunks. Now the prompt must make the model use them,
and only them:

    grounding   "Answer using ONLY the numbered sources below" + the sources in the message
    citations   numbered sources, the model writes [1] [2]; your CODE checks every number
    no answer   two layers:
                  1. a retrieval gate: if no chunk is similar enough, do not call the model at all
                  2. the prompt: if the sources do not say it, reply exactly NO_ANSWER

Run:  python 05_grounded_answers.py
      python 05_grounded_answers.py "your question"
"""
import sys

from azure_client import chat
from rag import MIN_SIMILARITY, NO_ANSWER, cited_numbers, generate, search


def answer_with_citations(question: str) -> str:
    """The full pipeline: retrieve, gate, generate, check the citations."""
    matches = search(question, top=3)
    best = max((m.similarity or 0 for m in matches), default=0)
    print(f"\nQ: {question}")
    print(f"   best similarity {best:.3f} (gate {MIN_SIMILARITY})")
    if best < MIN_SIMILARITY:
        print(f"A: {NO_ANSWER}   <- layer 1: the gate. No chunk is about this; the model was not called.")
        return NO_ANSWER

    answer = generate(question, matches)
    print(f"A: {answer}")
    if answer.strip() == NO_ANSWER:
        print("   <- layer 2: the model found nothing in the sources and said so.")
        return answer

    cited = cited_numbers(answer)
    for n in cited:
        if 1 <= n <= len(matches):
            print(f"   [{n}] {matches[n - 1].source} | {matches[n - 1].title}")
        else:
            print(f"   [{n}] DOES NOT EXIST: only {len(matches)} sources were sent. Treat the answer as suspect.")
    if not cited:
        print("   No citations: the prompt asked for them. Show it with a warning, or ask again.")
    return answer


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for question in sys.argv[1:]:
            answer_with_citations(question)
        sys.exit()

    print("=== 1. Without and with grounding ===")
    question = "My order arrived 5 days late. What compensation do I get?"
    print(f"Q: {question}")
    print(f"No sources:   {chat([{'role': 'user', 'content': f'I shop at ShopEase. {question} Answer in two sentences.'}])}")
    print(f"With sources: {generate(question, search(question, top=3))}")
    print("Without sources the model gives a plausible, generic answer. Only the ShopEase documents know 'Rs. 150 voucher'.")

    print()
    print("=== 2. Citations you can check ===")
    answer_with_citations("I am a Plus member. Do I pay for express delivery, and what would it cost otherwise?")
    answer_with_citations("How long is the warranty on a fridge compressor?")

    print()
    print("=== 3. When the answer is not in the documents ===")
    answer_with_citations("What is the capital of France?")             # off topic: the gate should stop it
    answer_with_citations("Who is the CEO of ShopEase?")                # on topic, not in the documents
    answer_with_citations("Can I pay with PayPal?")                     # the documents list what you CAN use
    print("\nDiscuss the PayPal answer: is 'PayPal is not in the list of payment methods' an answer or an 'I don't know'?")
    print("Decide what you want, then write it into the prompt and into your test set (lesson 6).")
