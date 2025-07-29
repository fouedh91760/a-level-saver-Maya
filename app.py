from flask import Flask, request, jsonify
import openai
import csv
import os
import tiktoken

# Initialize Flask
app = Flask(__name__)

# Load your OpenAI API key (you can move this to environment variables later)
openai.api_key = os.getenv("OPENAI_API_KEY")
  # replace this with your real key

# Load jc_corpus
with open("jc_corpus.txt", "r", encoding="utf-8") as f:
    jc_corpus = f.read()

# Split corpus into chunks
def split_corpus_into_chunks(corpus, max_tokens=300):
    enc = tiktoken.get_encoding("cl100k_base")
    words, chunks, chunk = corpus.split(), [], []
    for word in words:
        chunk.append(word)
        if len(enc.encode(" ".join(chunk))) >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk: chunks.append(" ".join(chunk))
    return chunks

# Find best chunk for query
def find_best_chunk(question, chunks):
    messages = [
        {"role": "system", "content": "Find the most relevant chunk for this query."},
        {"role": "user", "content": f"Query: {question}\n\nChunks:\n" + "\n---\n".join(chunks)}
    ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0)
    return response.choices[0].message["content"]

# Route: Home
@app.route("/")
def home():
    return "✅ A-Level Saver backend is running!"

# Route: Chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    # Collect all user fields from frontend
    name = data.get("name", "")
    interest = data.get("interest", "")
    gcse = data.get("gcse", "")
    thinking_subjects = data.get("thinking_subjects", "")
    style = data.get("style", "")
    personality = data.get("personality", "")
    ambition = data.get("ambition", "")
    worry = data.get("worry", "")
    a_level_plan = data.get("a_level_plan", "")
    activities = data.get("activities", "")
    dream_uni = data.get("dream_uni", "")
    epq_interest = data.get("epq_interest", "")
    btec_interest = data.get("btec_interest", "")

    # Build profile
    profile = f"""
Name: {name}
Enjoys: {interest}
GCSE Subjects: {gcse}
Already thinking about: {thinking_subjects}
Learning Style: {style}
Personality: {personality}
Ambition: {ambition}
Worries: {worry}
A-Level Plan: {a_level_plan}
Extracurriculars: {activities}
Dream Uni: {dream_uni}
Interested in EPQ: {epq_interest}
Interested in BTEC: {btec_interest}
"""

    # Save profile (append to CSV)
    csv_file = "saved_profiles.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["name", "interest", "gcse", "thinking_subjects", "style", "personality", "ambition",
                             "worry", "a_level_plan", "activities", "dream_uni", "epq_interest", "btec_interest"])
        writer.writerow([name, interest, gcse, thinking_subjects, style, personality, ambition, worry,
                         a_level_plan, activities, dream_uni, epq_interest, btec_interest])

    # RAG
    chunks = split_corpus_into_chunks(jc_corpus)
    best_chunk = find_best_chunk(profile, chunks)

    # Final GPT message
    messages = [
        {
            "role": "system",
            "content": f"""You are A-Level Saver — a warm, helpful, and supportive assistant helping a student choose A-Levels and explore future careers.

🎯 Structure your response like this:
🎯 Recommended A-Level Subjects:
- ...

💼 Matching Career Paths:
- ...

📚 Other Suggestions (EPQ, BTEC, extracurriculars):
- ...

💡 Final Thought:
End with encouragement.

📚 JC Context:
{jc_corpus}

🔍 Most relevant info:
{best_chunk}
"""
        },
        {"role": "user", "content": f"Here’s my info:\n{profile}\nWhat A-Levels would you recommend?"}
    ]

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
    reply = response.choices[0].message["content"]
    return jsonify({"reply": reply})

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
