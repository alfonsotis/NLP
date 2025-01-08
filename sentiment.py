import pandas as pd
from transformers import pipeline

def main():
    text = """Dear Amazon, last week I ordered an Optimus Prime action figure
    from your online store in Germany. Unfortunately, when I opened the package,
    I discovered to my horror that I had been sent an action figure of Megatron
    instead! As a lifelong enemy of the Decepticons, I hope you can understand my
    dilemma. To resolve the issue, I demand an exchange of Megatron for the
    Optimus Prime figure I ordered. Enclosed are copies of my records concerning
    this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    outputs = ner_tagger(text)
    df = pd.DataFrame(outputs)
    print(df)

    reader = pipeline("question-answering")
    question = "What does the customer want?"
    # question = "What type of events were hosted on weekends?"
    outputs = reader(question=question, context=text)
    df = pd.DataFrame([outputs])
    print(df)

    # summarizer = pipeline("summarization")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    outputs = summarizer(text, max_length=45, min_length=30, clean_up_tokenization_spaces=True)
    print(outputs[0]['summary_text'])

if __name__ == "__main__":
    main()


